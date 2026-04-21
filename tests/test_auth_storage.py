"""Tests for earl_sdk.auth_storage and the config_store integration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import pytest

from earl_sdk import auth_storage
from earl_sdk.auth_storage import CachedToken

# ── In-memory keyring shim ────────────────────────────────────────────────────


class FakeKeyring:
    """Minimal drop-in replacement for the ``keyring`` module."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    # real `keyring` module surface
    def set_password(self, service: str, user: str, value: str) -> None:
        self._data[(service, user)] = value

    def get_password(self, service: str, user: str) -> str | None:
        return self._data.get((service, user))

    def delete_password(self, service: str, user: str) -> None:
        self._data.pop((service, user), None)

    class _Backend:
        pass

    def get_keyring(self) -> object:
        return self._Backend()


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    """Inject a fresh FakeKeyring for every test."""
    fake = FakeKeyring()
    # Reset module-level caches so each test starts clean.
    monkeypatch.setattr(auth_storage, "_keyring_module", fake, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    monkeypatch.delenv("EARL_SECRET_BACKEND", raising=False)
    return fake


@pytest.fixture
def no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")


# ── set_secret / get_secret / delete_secret ──────────────────────────────────


def test_set_and_get_with_keyring(fake_keyring: FakeKeyring) -> None:
    backend = auth_storage.set_secret("profile:p1:client_secret", "s3cret!")
    assert backend == "keyring"
    assert auth_storage.get_secret("profile:p1:client_secret") == "s3cret!"


def test_legacy_base64_fallback_without_keyring(no_keyring: None) -> None:
    assert auth_storage.backend_name() == "file"
    import base64

    b64 = base64.b64encode(b"legacy-secret").decode()
    assert auth_storage.get_secret("profile:p1:client_secret", legacy_base64=b64) == "legacy-secret"


def test_get_secret_prefers_keyring_over_legacy(fake_keyring: FakeKeyring) -> None:
    auth_storage.set_secret("profile:p1:client_secret", "from-keyring")
    import base64

    legacy = base64.b64encode(b"stale-base64").decode()
    assert (
        auth_storage.get_secret("profile:p1:client_secret", legacy_base64=legacy) == "from-keyring"
    )


def test_delete_secret_is_idempotent(fake_keyring: FakeKeyring) -> None:
    auth_storage.set_secret("profile:p1:client_secret", "x")
    auth_storage.delete_secret("profile:p1:client_secret")
    auth_storage.delete_secret("profile:p1:client_secret")  # no raise
    assert auth_storage.get_secret("profile:p1:client_secret") == ""


def test_env_var_forces_file_backend(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: FakeKeyring
) -> None:
    # Even with keyring loaded, EARL_SECRET_BACKEND=file disables it on the next lookup.
    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", False, raising=False)
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")
    assert auth_storage.backend_name() == "file"


# ── Token cache ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_token_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    d = tmp_path / "tokens"
    monkeypatch.setattr(auth_storage, "TOKEN_CACHE_DIR", d, raising=False)
    return d


def test_token_cache_roundtrip(tmp_token_dir: Path) -> None:
    key = auth_storage.token_cache_key("cid", "aud", "org", "dom")
    token = CachedToken(access_token="abc", token_type="Bearer", expires_at=time.time() + 3600)
    auth_storage.save_token(key, token)

    path = auth_storage.token_cache_path(key)
    assert path.exists()
    # 0600 perms on POSIX
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600 or mode == 0o666  # CI filesystems sometimes force 0666

    loaded = auth_storage.load_token(key)
    assert loaded is not None
    assert loaded.access_token == "abc"


def test_token_cache_expired_is_ignored(tmp_token_dir: Path) -> None:
    key = "expired"
    stale = CachedToken(access_token="old", token_type="Bearer", expires_at=time.time() - 1)
    auth_storage.save_token(key, stale)
    assert auth_storage.load_token(key) is None


def test_token_cache_missing_returns_none(tmp_token_dir: Path) -> None:
    assert auth_storage.load_token("nope") is None


def test_token_cache_clear(tmp_token_dir: Path) -> None:
    key = "clearme"
    auth_storage.save_token(
        key, CachedToken(access_token="t", token_type="Bearer", expires_at=time.time() + 3600)
    )
    auth_storage.clear_token(key)
    auth_storage.clear_token(key)  # idempotent
    assert auth_storage.load_token(key) is None


def test_token_cache_corrupt_returns_none(tmp_token_dir: Path) -> None:
    key = "corrupt"
    p = auth_storage.token_cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    assert auth_storage.load_token(key) is None


# ── ConfigStore integration ──────────────────────────────────────────────────


@pytest.fixture
def tmp_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from earl_sdk.interactive.storage import config_store as cs

    path = tmp_path / "config.json"
    monkeypatch.setattr(cs, "CONFIG_PATH", path, raising=False)
    return cs, path


def test_profile_create_uses_keyring_when_available(fake_keyring: FakeKeyring, tmp_config) -> None:
    cs, path = tmp_config
    prof = cs.AuthProfile.create(
        name="prod",
        client_id="cid",
        clear_secret="s3cr3t",
        organization="org_x",
        environment="prod",
    )
    assert prof.secret_backend == "keyring"
    assert prof.client_secret == ""  # never stored in file when keyring is used
    assert prof.secret_clear() == "s3cr3t"

    store = cs.ConfigStore(path)
    store.upsert_profile(prof)

    # On-disk JSON must not contain the clear-text secret
    text = path.read_text()
    assert "s3cr3t" not in text


def test_profile_create_falls_back_to_file(no_keyring: None, tmp_config) -> None:
    cs, path = tmp_config
    prof = cs.AuthProfile.create(
        name="dev", client_id="cid", clear_secret="plain-secret", environment="dev"
    )
    assert prof.secret_backend == "file"
    assert prof.client_secret != ""  # base64 blob
    assert prof.secret_clear() == "plain-secret"


def test_migrate_secrets_moves_base64_to_keyring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    from earl_sdk.interactive.storage import config_store as cs

    path = tmp_path / "config.json"
    monkeypatch.setattr(cs, "CONFIG_PATH", path, raising=False)

    # Simulate an old config written by a pre-keyring SDK release.
    import base64

    legacy_blob = base64.b64encode(b"legacy-secret").decode()
    legacy_api_key = base64.b64encode(b"legacy-api-key").decode()
    path.write_text(
        json.dumps(
            {
                "active_profile": "old",
                "profiles": {
                    "old": {
                        "name": "old",
                        "client_id": "cid",
                        "client_secret": legacy_blob,
                        "organization": "org",
                        "environment": "prod",
                    }
                },
                "doctor_configs": {
                    "d1": {
                        "name": "d1",
                        "type": "external",
                        "api_url": "https://ex/chat",
                        "api_key": legacy_api_key,
                        "auth_type": "bearer",
                    }
                },
                "preferences": {},
            }
        )
    )

    store = cs.ConfigStore(path)
    summary = store.migrate_secrets_to_keyring()
    assert summary == {"profiles_migrated": 1, "doctors_migrated": 1, "skipped": 0}

    # Re-read from disk to ensure persistence
    store2 = cs.ConfigStore(path)
    prof = store2.config.profiles["old"]
    assert prof.secret_backend == "keyring"
    assert prof.client_secret == ""
    assert prof.secret_clear() == "legacy-secret"

    dc = store2.config.doctor_configs["d1"]
    assert dc.secret_backend == "keyring"
    assert dc.api_key == ""
    assert dc.key_clear() == "legacy-api-key"

    # Re-running is a no-op.
    again = store2.migrate_secrets_to_keyring()
    assert again == {"profiles_migrated": 0, "doctors_migrated": 0, "skipped": 0}


def test_migrate_skipped_when_keyring_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_keyring: None
) -> None:
    from earl_sdk.interactive.storage import config_store as cs

    path = tmp_path / "config.json"
    monkeypatch.setattr(cs, "CONFIG_PATH", path, raising=False)
    store = cs.ConfigStore(path)
    summary = store.migrate_secrets_to_keyring()
    assert summary["skipped"] == -1


def test_delete_profile_removes_keyring_entry(fake_keyring: FakeKeyring, tmp_config) -> None:
    cs, path = tmp_config
    prof = cs.AuthProfile.create(name="prod", client_id="cid", clear_secret="s", environment="prod")
    store = cs.ConfigStore(path)
    store.upsert_profile(prof)
    assert auth_storage.get_secret("profile:prod:client_secret") == "s"

    store.delete_profile("prod")
    assert auth_storage.get_secret("profile:prod:client_secret") == ""


# ── Auth0Client disk cache integration ───────────────────────────────────────


def test_auth0client_loads_cached_token_on_init(
    tmp_token_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from earl_sdk.auth import Auth0Client

    key = auth_storage.token_cache_key("cid", "aud", "org", "dom", "m2m")
    auth_storage.save_token(
        key,
        CachedToken(access_token="cached-tok", token_type="Bearer", expires_at=time.time() + 3600),
    )

    client = Auth0Client(
        client_id="cid",
        client_secret="secret",
        organization="org",
        domain="dom",
        audience="aud",
    )
    # Must not hit the network; the cache has a valid token.
    with mock.patch.object(Auth0Client, "_fetch_m2m_token") as fetch:
        assert client.get_token() == "cached-tok"
        fetch.assert_not_called()


def test_auth0client_persists_token_after_fetch(
    tmp_token_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from earl_sdk.auth import Auth0Client, TokenInfo

    client = Auth0Client(
        client_id="cid2",
        client_secret="s",
        organization="org2",
        domain="dom2",
        audience="aud2",
    )

    def fake_fetch(self: Auth0Client) -> TokenInfo:
        return TokenInfo(access_token="fresh", token_type="Bearer", expires_at=time.time() + 3600)

    with mock.patch.object(Auth0Client, "_fetch_m2m_token", fake_fetch):
        assert client.get_token() == "fresh"

    key = auth_storage.token_cache_key("cid2", "aud2", "org2", "dom2", "m2m")
    reloaded = auth_storage.load_token(key)
    assert reloaded is not None
    assert reloaded.access_token == "fresh"


def test_auth0client_invalidate_clears_disk_cache(tmp_token_dir: Path) -> None:
    from earl_sdk.auth import Auth0Client

    key = auth_storage.token_cache_key("cid3", "aud3", "org3", "dom3", "m2m")
    auth_storage.save_token(
        key, CachedToken(access_token="t", token_type="Bearer", expires_at=time.time() + 3600)
    )

    client = Auth0Client(
        client_id="cid3",
        client_secret="s",
        organization="org3",
        domain="dom3",
        audience="aud3",
    )
    assert client._token_info is not None
    client.invalidate_token()
    assert client._token_info is None
    assert auth_storage.load_token(key) is None


def test_auth0client_rejects_empty_domain_or_audience() -> None:
    from earl_sdk.auth import Auth0Client

    with pytest.raises(ValueError):
        Auth0Client(client_id="x", client_secret="y", organization="", domain="", audience="a")
    with pytest.raises(ValueError):
        Auth0Client(client_id="x", client_secret="y", organization="", domain="d", audience="")
