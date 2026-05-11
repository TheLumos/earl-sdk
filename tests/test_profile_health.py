"""Tests for the shared profile-health helpers used by both the command CLI
(`earl auth profile …`) and the interactive CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from earl_sdk import profile_health
from earl_sdk.interactive.storage import config_store as cs


@pytest.fixture
def tmp_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = tmp_path / "config.json"
    monkeypatch.setattr(cs, "CONFIG_PATH", path, raising=False)
    return path


@pytest.fixture
def no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    from earl_sdk import auth_storage

    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")


# ── build_client_kwargs ──────────────────────────────────────────────────────


def test_build_client_kwargs_m2m(no_keyring: None, tmp_config) -> None:
    prof = cs.AuthProfile.create(
        name="ci",
        client_id="cid",
        clear_secret="topsecret",
        organization="org_x",
        environment="prod",
    )
    kwargs = profile_health.build_client_kwargs(prof)
    assert kwargs["client_id"] == "cid"
    assert kwargs["client_secret"] == "topsecret"
    assert kwargs["auth_kind"] == "m2m"
    assert kwargs["organization"] == "org_x"
    assert "refresh_token" not in kwargs


def test_build_client_kwargs_pkce_uses_refresh(no_keyring: None, tmp_config) -> None:
    prof = cs.AuthProfile.create_pkce(
        name="pkce-dev-acme",
        client_id="public-native",
        clear_refresh_token="rt-abc",
        organization="org_x",
        organization_name="Acme",
        environment="dev",
    )
    kwargs = profile_health.build_client_kwargs(prof)
    assert kwargs["auth_kind"] == "pkce"
    assert kwargs["client_secret"] == ""  # browser profile must not look like M2M
    assert kwargs["refresh_token"] == "rt-abc"


def test_build_client_kwargs_device_uses_refresh(no_keyring: None, tmp_config) -> None:
    prof = cs.AuthProfile.create_device(
        name="device-dev-acme",
        client_id="public-native",
        clear_refresh_token="rt-xyz",
        environment="dev",
    )
    kwargs = profile_health.build_client_kwargs(prof)
    assert kwargs["auth_kind"] == "device"
    assert kwargs["client_secret"] == ""
    assert kwargs["refresh_token"] == "rt-xyz"


# ── test_profile / test_and_record ───────────────────────────────────────────


class _FakeClient:
    def __init__(self, *, result: Any = True, raise_exc: Exception | None = None) -> None:
        self.result = result
        self.raise_exc = raise_exc
        self.called = 0

    def test_connection(self) -> bool:
        self.called += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def test_test_profile_ok(monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    fake = _FakeClient(result=True)
    monkeypatch.setattr(profile_health, "build_client", lambda _p: fake)

    result = profile_health.test_profile(prof)
    assert result.ok is True
    assert result.error == ""
    assert fake.called == 1


def test_test_profile_returns_false_recorded_as_fail(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    monkeypatch.setattr(profile_health, "build_client", lambda _p: _FakeClient(result=False))

    result = profile_health.test_profile(prof)
    assert result.ok is False
    assert "False" in result.error


def test_test_profile_captures_transport_error(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    monkeypatch.setattr(
        profile_health,
        "build_client",
        lambda _p: _FakeClient(raise_exc=RuntimeError("DNS failed")),
    )

    result = profile_health.test_profile(prof)
    assert result.ok is False
    assert "DNS failed" in result.error


def test_test_profile_captures_init_error(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")

    def _boom(_p):
        raise ValueError("missing audience")

    monkeypatch.setattr(profile_health, "build_client", _boom)

    result = profile_health.test_profile(prof)
    assert result.ok is False
    assert "init failed" in result.error
    assert "missing audience" in result.error


def test_test_and_record_persists_health(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    store = cs.ConfigStore(tmp_config)
    store.upsert_profile(prof)

    monkeypatch.setattr(profile_health, "build_client", lambda _p: _FakeClient(result=True))

    result = profile_health.test_and_record(store, prof)
    assert result.ok is True

    fresh = cs.ConfigStore(tmp_config).config.profiles["p"]
    assert fresh.status_label() == "ok"
    assert fresh.last_tested_at != ""


def test_test_and_record_persists_failure(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    store = cs.ConfigStore(tmp_config)
    store.upsert_profile(prof)

    monkeypatch.setattr(
        profile_health,
        "build_client",
        lambda _p: _FakeClient(raise_exc=RuntimeError("403 Forbidden")),
    )
    result = profile_health.test_and_record(store, prof)
    assert result.ok is False

    fresh = cs.ConfigStore(tmp_config).config.profiles["p"]
    assert fresh.status_label() == "fail"
    assert "403 Forbidden" in fresh.last_test_error


def test_build_client_calls_earlclient_with_kwargs(
    monkeypatch: pytest.MonkeyPatch, no_keyring: None, tmp_config
) -> None:
    """Sanity check the lazy import path: build_client should call EarlClient
    exactly once with the resolved kwargs and forward the instance back."""
    prof = cs.AuthProfile.create(name="p", client_id="cid", clear_secret="s", environment="dev")
    sentinel = object()
    fake_client = mock.MagicMock(return_value=sentinel)

    import earl_sdk

    monkeypatch.setattr(earl_sdk, "EarlClient", fake_client)
    out = profile_health.build_client(prof)
    assert out is sentinel
    kwargs = fake_client.call_args.kwargs
    assert kwargs["client_id"] == "cid"
    assert kwargs["auth_kind"] == "m2m"
