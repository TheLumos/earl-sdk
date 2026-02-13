"""Persistent local configuration: auth profiles, doctor configs, preferences.

Stored in ``~/.earl/config.json``.  Secrets are base64-obfuscated (not
encrypted — this is a convenience CLI, not a vault).
"""

from __future__ import annotations

import base64
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ._atomic import atomic_write_text

EARL_DIR = Path.home() / ".earl"
CONFIG_PATH = EARL_DIR / "config.json"


def _secure_dir(path: Path) -> None:
    """Create directory with owner-only permissions (0o700)."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)  # rwx------
    except OSError:
        pass  # Windows or restricted fs


def _secure_file(path: Path) -> None:
    """Set file to owner-only read/write (0o600)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # rw-------
    except OSError:
        pass


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class AuthProfile:
    """Credentials for one EARL environment."""

    name: str
    client_id: str
    client_secret: str  # stored obfuscated
    organization: str = ""
    environment: str = "test"  # dev | test | prod

    def secret_clear(self) -> str:
        """Return the clear-text secret."""
        try:
            return base64.b64decode(self.client_secret.encode()).decode()
        except Exception:
            return self.client_secret

    @staticmethod
    def obfuscate(clear: str) -> str:
        return base64.b64encode(clear.encode()).decode()


@dataclass
class DoctorConfig:
    """Saved doctor endpoint configuration."""

    name: str
    type: str  # "external" | "internal" | "client_driven"
    api_url: str = ""
    api_key: str = ""  # stored obfuscated
    auth_type: str = "bearer"

    def key_clear(self) -> str:
        if not self.api_key:
            return ""
        try:
            return base64.b64decode(self.api_key.encode()).decode()
        except Exception:
            return self.api_key

    @staticmethod
    def obfuscate(clear: str) -> str:
        return base64.b64encode(clear.encode()).decode()


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


# ── Persistence ───────────────────────────────────────────────────────────────


class ConfigStore:
    """Read/write ``~/.earl/config.json``."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self._path = path
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
        cfg.profiles.pop(name, None)
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
        self.load().doctor_configs.pop(name, None)
        self.save()

    # -- preferences --

    @property
    def preferences(self) -> Preferences:
        return self.load().preferences

    def save_preferences(self, prefs: Preferences) -> None:
        self.load().preferences = prefs
        self.save()

    # -- (de)serialization --

    @staticmethod
    def _serialize(cfg: AppConfig) -> dict[str, Any]:
        return {
            "active_profile": cfg.active_profile,
            "profiles": {k: asdict(v) for k, v in cfg.profiles.items()},
            "doctor_configs": {k: asdict(v) for k, v in cfg.doctor_configs.items()},
            "preferences": asdict(cfg.preferences),
        }

    @staticmethod
    def _deserialize(raw: dict[str, Any]) -> AppConfig:
        profiles = {
            k: AuthProfile(**v) for k, v in raw.get("profiles", {}).items()
        }
        doctors = {
            k: DoctorConfig(**v) for k, v in raw.get("doctor_configs", {}).items()
        }
        prefs_raw = raw.get("preferences", {})
        prefs = Preferences(**{k: v for k, v in prefs_raw.items() if k in Preferences.__dataclass_fields__})
        return AppConfig(
            profiles=profiles,
            active_profile=raw.get("active_profile", ""),
            doctor_configs=doctors,
            preferences=prefs,
        )
