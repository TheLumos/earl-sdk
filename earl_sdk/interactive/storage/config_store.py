"""Persistent local configuration: auth profiles, doctor configs, preferences.

Stored in ``~/.earl/config.json``.  Secrets are stored in the OS keyring when
available (``secret_backend == "keyring"``) and in a base64-obfuscated form in
the JSON as a fallback (``secret_backend == "file"``).  Either way, the
clear-text value is never written to ``config.json``.

See :mod:`earl_sdk.auth_storage` for the underlying backend abstraction.
"""

from __future__ import annotations

import base64
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ... import auth_storage
from ._atomic import atomic_write_text

EARL_DIR = Path.home() / ".earl"
CONFIG_PATH = EARL_DIR / "config.json"


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)  # rwx------
    except OSError:
        pass  # Windows or restricted fs


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # rw-------
    except OSError:
        pass


# ── Key helpers ───────────────────────────────────────────────────────────────


def _profile_secret_key(name: str) -> str:
    return f"profile:{name}:client_secret"


def _profile_refresh_key(name: str) -> str:
    return f"profile:{name}:refresh_token"


def _doctor_secret_key(name: str) -> str:
    return f"doctor:{name}:api_key"


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class AuthProfile:
    """Credentials for one EARL environment.

    Supports three authentication kinds:

    - ``auth_kind == "m2m"`` (default) — Auth0 client_credentials grant.
      ``client_id`` is the M2M app's ID and ``client_secret`` is set (either
      in the keyring or base64-obfuscated in this file). Used by servers/CI.
    - ``auth_kind == "pkce"`` — Authorization Code + PKCE loopback login,
      set up by ``earl login``. Preferred interactive path. ``client_id``
      is the public Native app's ID (no secret). Refresh-token storage is
      identical to the ``device`` kind.
    - ``auth_kind == "device"`` — OAuth Device Authorization Grant (headless
      fallback). ``client_id`` is the public Native app's ID (no secret).
      The refresh token lives under ``earl-sdk`` /
      ``profile:<name>:refresh_token`` in the keyring (or a base64-obfuscated
      copy is stored in the ``client_secret`` field when the keyring is
      unavailable — kept in that field to avoid expanding the stored schema).
    """

    name: str
    client_id: str
    client_secret: str = ""  # base64 secret OR base64 refresh_token (file backend)
    organization: str = ""
    environment: str = "staging"  # local | dev | staging | prod
    secret_backend: str = "file"
    # New in M2. Older configs without this field default to the M2M grant.
    auth_kind: str = "m2m"  # "m2m" | "pkce" | "device"

    # ── read ──

    def secret_clear(self) -> str:
        """Return the clear-text Auth0 client secret (M2M only)."""
        if self.auth_kind != "m2m":
            return ""
        if self.secret_backend == "keyring":
            return auth_storage.get_secret(_profile_secret_key(self.name))
        return auth_storage.get_secret(
            _profile_secret_key(self.name),
            legacy_base64=self.client_secret,
        )

    def refresh_token_clear(self) -> str:
        """Return the clear-text refresh token for browser-issued profiles."""
        if self.auth_kind not in ("device", "pkce"):
            return ""
        if self.secret_backend == "keyring":
            return auth_storage.get_secret(_profile_refresh_key(self.name))
        # File backend: refresh token is base64'd in the client_secret field.
        return auth_storage.get_secret(
            _profile_refresh_key(self.name),
            legacy_base64=self.client_secret,
        )

    # ── write helpers ──

    @staticmethod
    def obfuscate(clear: str) -> str:
        """Back-compat helper; prefer :meth:`create` for new profiles."""
        return base64.b64encode(clear.encode()).decode()

    @classmethod
    def create(
        cls,
        *,
        name: str,
        client_id: str,
        clear_secret: str,
        organization: str = "",
        environment: str = "staging",
    ) -> AuthProfile:
        """Create an M2M profile, storing *clear_secret* via the best backend."""
        backend = auth_storage.set_secret(_profile_secret_key(name), clear_secret)
        return cls(
            name=name,
            client_id=client_id,
            client_secret="" if backend == "keyring" else auth_storage.obfuscate(clear_secret),
            organization=organization,
            environment=environment,
            secret_backend=backend,
            auth_kind="m2m",
        )

    @classmethod
    def create_device(
        cls,
        *,
        name: str,
        client_id: str,
        clear_refresh_token: str,
        organization: str = "",
        environment: str = "staging",
    ) -> AuthProfile:
        """Create a device-flow profile and persist the refresh token securely."""
        return cls._create_browser(
            name=name,
            client_id=client_id,
            clear_refresh_token=clear_refresh_token,
            organization=organization,
            environment=environment,
            auth_kind="device",
        )

    @classmethod
    def create_pkce(
        cls,
        *,
        name: str,
        client_id: str,
        clear_refresh_token: str,
        organization: str = "",
        environment: str = "staging",
    ) -> AuthProfile:
        """Create a PKCE profile (browser + loopback login)."""
        return cls._create_browser(
            name=name,
            client_id=client_id,
            clear_refresh_token=clear_refresh_token,
            organization=organization,
            environment=environment,
            auth_kind="pkce",
        )

    @classmethod
    def _create_browser(
        cls,
        *,
        name: str,
        client_id: str,
        clear_refresh_token: str,
        organization: str,
        environment: str,
        auth_kind: str,
    ) -> AuthProfile:
        backend = auth_storage.set_secret(_profile_refresh_key(name), clear_refresh_token)
        return cls(
            name=name,
            client_id=client_id,
            client_secret=""
            if backend == "keyring"
            else auth_storage.obfuscate(clear_refresh_token),
            organization=organization,
            environment=environment,
            secret_backend=backend,
            auth_kind=auth_kind,
        )

    def update_refresh_token(self, new_refresh: str) -> None:
        """Replace the stored refresh token (after Auth0 rotation)."""
        if self.auth_kind not in ("device", "pkce"):
            return
        backend = auth_storage.set_secret(_profile_refresh_key(self.name), new_refresh)
        self.secret_backend = backend
        self.client_secret = "" if backend == "keyring" else auth_storage.obfuscate(new_refresh)


@dataclass
class DoctorConfig:
    """Saved doctor endpoint configuration."""

    name: str
    type: str  # "external" | "internal" | "client_driven"
    api_url: str = ""
    api_key: str = ""
    auth_type: str = "bearer"
    secret_backend: str = "file"

    def key_clear(self) -> str:
        """Return the clear-text API key (or empty)."""
        if self.type != "external":
            return ""
        if self.secret_backend == "keyring":
            return auth_storage.get_secret(_doctor_secret_key(self.name))
        if not self.api_key:
            return ""
        return auth_storage.get_secret(
            _doctor_secret_key(self.name),
            legacy_base64=self.api_key,
        )

    @staticmethod
    def obfuscate(clear: str) -> str:
        return base64.b64encode(clear.encode()).decode()

    @classmethod
    def create(
        cls,
        *,
        name: str,
        type: str,  # noqa: A002 — matches dataclass field name
        api_url: str = "",
        clear_api_key: str = "",
        auth_type: str = "bearer",
    ) -> DoctorConfig:
        if clear_api_key:
            backend = auth_storage.set_secret(_doctor_secret_key(name), clear_api_key)
            stored = "" if backend == "keyring" else auth_storage.obfuscate(clear_api_key)
        else:
            backend = "file"
            stored = ""
        return cls(
            name=name,
            type=type,
            api_url=api_url,
            api_key=stored,
            auth_type=auth_type,
            secret_backend=backend,
        )


@dataclass
class Preferences:
    default_pipeline: str = ""
    default_parallel: int = 3
    auto_save_runs: bool = True
    max_local_runs: int = 50


@dataclass
class AppConfig:
    profiles: dict[str, AuthProfile] = field(default_factory=dict)
    active_profile: str = ""
    doctor_configs: dict[str, DoctorConfig] = field(default_factory=dict)
    preferences: Preferences = field(default_factory=Preferences)
    # Public Auth0 Native-app client IDs for the Device Authorization Grant,
    # keyed by environment (``local``/``dev``/``staging``/``prod``). These are
    # NOT secrets — Native clients are public. Populate via
    # ``earl auth device-clients set --env <env> <client_id>``.
    device_client_ids: dict[str, str] = field(default_factory=dict)


# ── Persistence ───────────────────────────────────────────────────────────────


class ConfigStore:
    """Read/write ``~/.earl/config.json``."""

    def __init__(self, path: Path | None = None) -> None:
        # Resolve at call time so test fixtures that monkeypatch CONFIG_PATH
        # (or HOME) take effect. The module-level CONFIG_PATH default is
        # captured at import time and cannot be overridden after the fact.
        self._path = path if path is not None else CONFIG_PATH
        self._cfg: AppConfig | None = None

    # -- public api --

    def load(self) -> AppConfig:
        if self._cfg is not None:
            return self._cfg
        if not self._path.exists():
            self._cfg = AppConfig()
            return self._cfg
        try:
            raw = json.loads(self._path.read_text())
            self._cfg = self._deserialize(raw)
        except Exception:
            self._cfg = AppConfig()
        return self._cfg

    def save(self) -> None:
        cfg = self.load()
        _secure_dir(self._path.parent)
        atomic_write_text(self._path, json.dumps(self._serialize(cfg), indent=2) + "\n")
        _secure_file(self._path)

    @property
    def config(self) -> AppConfig:
        return self.load()

    # -- profiles --

    def get_active_profile(self) -> AuthProfile | None:
        cfg = self.load()
        return cfg.profiles.get(cfg.active_profile)

    def set_active_profile(self, name: str) -> None:
        cfg = self.load()
        if name not in cfg.profiles:
            raise KeyError(f"Profile '{name}' not found")
        cfg.active_profile = name
        self.save()

    def upsert_profile(self, profile: AuthProfile) -> None:
        cfg = self.load()
        cfg.profiles[profile.name] = profile
        if not cfg.active_profile:
            cfg.active_profile = profile.name
        self.save()

    def delete_profile(self, name: str) -> None:
        cfg = self.load()
        prof = cfg.profiles.pop(name, None)
        if prof is not None and prof.secret_backend == "keyring":
            auth_storage.delete_secret(_profile_secret_key(name))
            auth_storage.delete_secret(_profile_refresh_key(name))
        if cfg.active_profile == name:
            cfg.active_profile = next(iter(cfg.profiles), "")
        self.save()

    # -- doctor configs --

    def list_doctor_configs(self) -> list[DoctorConfig]:
        return list(self.load().doctor_configs.values())

    def upsert_doctor_config(self, dc: DoctorConfig) -> None:
        self.load().doctor_configs[dc.name] = dc
        self.save()

    def delete_doctor_config(self, name: str) -> None:
        cfg = self.load()
        dc = cfg.doctor_configs.pop(name, None)
        if dc is not None and dc.secret_backend == "keyring":
            auth_storage.delete_secret(_doctor_secret_key(name))
        self.save()

    # -- preferences --

    @property
    def preferences(self) -> Preferences:
        return self.load().preferences

    def save_preferences(self, prefs: Preferences) -> None:
        self.load().preferences = prefs
        self.save()

    # -- migration --

    def migrate_secrets_to_keyring(self) -> dict[str, int]:
        """Move any file-backed secrets into the OS keyring.

        Returns a summary dict with counts.  Safe to run repeatedly; profiles
        already using keyring are left untouched.
        """
        if auth_storage.backend_name() != "keyring":
            return {"profiles_migrated": 0, "doctors_migrated": 0, "skipped": -1}

        cfg = self.load()
        prof_moved = 0
        doc_moved = 0

        for prof in cfg.profiles.values():
            if prof.secret_backend == "keyring" or not prof.client_secret:
                continue
            clear = prof.secret_clear()
            if not clear:
                continue
            backend = auth_storage.set_secret(_profile_secret_key(prof.name), clear)
            if backend == "keyring":
                prof.secret_backend = "keyring"
                prof.client_secret = ""
                prof_moved += 1

        for dc in cfg.doctor_configs.values():
            if dc.secret_backend == "keyring" or not dc.api_key:
                continue
            clear = dc.key_clear()
            if not clear:
                continue
            backend = auth_storage.set_secret(_doctor_secret_key(dc.name), clear)
            if backend == "keyring":
                dc.secret_backend = "keyring"
                dc.api_key = ""
                doc_moved += 1

        if prof_moved or doc_moved:
            self.save()
        return {"profiles_migrated": prof_moved, "doctors_migrated": doc_moved, "skipped": 0}

    # -- (de)serialization --

    # -- device-code client IDs (public per-env Native apps) --

    def get_device_client_id(self, environment: str) -> str:
        return self.load().device_client_ids.get(environment, "")

    def set_device_client_id(self, environment: str, client_id: str) -> None:
        cfg = self.load()
        cfg.device_client_ids[environment] = client_id
        self.save()

    def clear_device_client_id(self, environment: str) -> None:
        cfg = self.load()
        cfg.device_client_ids.pop(environment, None)
        self.save()

    @staticmethod
    def _serialize(cfg: AppConfig) -> dict[str, Any]:
        return {
            "active_profile": cfg.active_profile,
            "profiles": {k: asdict(v) for k, v in cfg.profiles.items()},
            "doctor_configs": {k: asdict(v) for k, v in cfg.doctor_configs.items()},
            "preferences": asdict(cfg.preferences),
            "device_client_ids": dict(cfg.device_client_ids),
        }

    @staticmethod
    def _deserialize(raw: dict[str, Any]) -> AppConfig:
        profiles = {k: _build_profile(v) for k, v in raw.get("profiles", {}).items()}
        doctors = {k: _build_doctor(v) for k, v in raw.get("doctor_configs", {}).items()}
        prefs_raw = raw.get("preferences", {})
        prefs = Preferences(
            **{k: v for k, v in prefs_raw.items() if k in Preferences.__dataclass_fields__}
        )
        device_client_ids = dict(raw.get("device_client_ids", {}))

        # Back-compat: the legacy env label was ``test``; rename in-place to
        # ``staging``. Profile environments are migrated here; profile *names*
        # are left alone so existing ``--profile pkce-test-…`` invocations
        # keep working.
        for prof in profiles.values():
            if getattr(prof, "environment", "") == "test":
                prof.environment = "staging"
        if "test" in device_client_ids and "staging" not in device_client_ids:
            device_client_ids["staging"] = device_client_ids.pop("test")
        elif "test" in device_client_ids:
            # ``staging`` already populated — drop the stale ``test`` key.
            device_client_ids.pop("test", None)

        return AppConfig(
            profiles=profiles,
            active_profile=raw.get("active_profile", ""),
            doctor_configs=doctors,
            preferences=prefs,
            device_client_ids=device_client_ids,
        )


def _build_profile(raw: dict[str, Any]) -> AuthProfile:
    return AuthProfile(**{k: v for k, v in raw.items() if k in AuthProfile.__dataclass_fields__})


def _build_doctor(raw: dict[str, Any]) -> DoctorConfig:
    return DoctorConfig(**{k: v for k, v in raw.items() if k in DoctorConfig.__dataclass_fields__})
