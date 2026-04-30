"""Secret storage for Earl SDK credentials.

Prefers the OS keyring (macOS Keychain, Windows Credential Vault, Linux Secret
Service via ``python-keyring``).  Falls back to base64-obfuscated JSON in
``~/.earl/config.json`` when keyring is unavailable or explicitly disabled via
``EARL_SECRET_BACKEND=file``.

Key conventions used by the rest of the SDK:

- ``profile:<profile_name>:client_secret`` — Auth0 M2M client secret
- ``profile:<profile_name>:refresh_token`` — OAuth refresh token (M2+)
- ``doctor:<config_name>:api_key`` — External doctor endpoint API key

Public surface:

- :data:`SecretBackend` — current backend (``"keyring"`` or ``"file"``)
- :func:`backend_name` — resolved backend name for the running process
- :func:`set_secret` / :func:`get_secret` / :func:`delete_secret`
- Token cache helpers: :func:`save_token`, :func:`load_token`,
  :func:`clear_token`, :func:`token_cache_path`. Refresh tokens are stored in
  the OS keyring when available; the JSON cache keeps them only when the
  explicit file fallback is in use.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .interactive.storage._atomic import atomic_write_text

logger = logging.getLogger(__name__)

SERVICE_NAME = "earl-sdk"
EARL_DIR = Path.home() / ".earl"
TOKEN_CACHE_DIR = EARL_DIR / "tokens"

SecretBackend = Literal["keyring", "file"]


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


# ── Keyring loading (optional dependency) ─────────────────────────────────────

_keyring_module = None
_keyring_checked = False


def _load_keyring():
    """Import keyring lazily; cache result (including failure) for the process."""
    global _keyring_module, _keyring_checked
    if _keyring_checked:
        return _keyring_module
    _keyring_checked = True
    if os.getenv("EARL_SECRET_BACKEND", "").lower() == "file":
        _keyring_module = None
        return None
    try:
        import keyring  # type: ignore
        from keyring.errors import KeyringError  # type: ignore  # noqa: F401

        backend = keyring.get_keyring()
        backend_name = type(backend).__module__ + "." + type(backend).__name__
        if "fail" in backend_name.lower() or "null" in backend_name.lower():
            logger.debug("keyring backend is a null/fail backend: %s", backend_name)
            _keyring_module = None
            return None
        _keyring_module = keyring
        logger.debug("keyring backend: %s", backend_name)
        return keyring
    except Exception as exc:  # ImportError, initialisation errors, etc.
        logger.debug("keyring unavailable: %s", exc)
        _keyring_module = None
        return None


def backend_name() -> SecretBackend:
    """Return the backend that will be used for new writes."""
    return "keyring" if _load_keyring() is not None else "file"


# ── Secret set/get/delete ─────────────────────────────────────────────────────


def set_secret(key: str, value: str) -> SecretBackend:
    """Persist *value* under *key*.  Returns the backend actually used.

    Callers that get back ``"file"`` are responsible for serialising an
    obfuscated copy into ``config.json`` themselves (this module does not
    touch ``config.json`` directly to avoid a circular dependency).
    """
    kr = _load_keyring()
    if kr is None:
        return "file"
    try:
        kr.set_password(SERVICE_NAME, key, value)
        return "keyring"
    except Exception as exc:
        logger.warning("keyring.set_password failed for %s (%s); falling back to file", key, exc)
        return "file"


def get_secret(key: str, legacy_base64: str = "") -> str:
    """Return the clear-text secret for *key*.

    Resolution order:

    1. Keyring (if available).
    2. ``legacy_base64`` — decoded base64 string from config.json (migration path).
    3. Empty string.
    """
    kr = _load_keyring()
    if kr is not None:
        try:
            val = kr.get_password(SERVICE_NAME, key)
            if val:
                return val
        except Exception as exc:
            logger.debug("keyring.get_password failed for %s: %s", key, exc)
    if not legacy_base64:
        return ""
    try:
        return base64.b64decode(legacy_base64.encode()).decode()
    except Exception:
        # Already clear text from a very old config; return as-is.
        return legacy_base64


def delete_secret(key: str) -> None:
    """Remove *key* from the keyring (best-effort; no error if missing)."""
    kr = _load_keyring()
    if kr is None:
        return
    try:
        kr.delete_password(SERVICE_NAME, key)
    except Exception as exc:
        logger.debug("keyring.delete_password failed for %s: %s", key, exc)


def obfuscate(clear: str) -> str:
    """Base64-encode a secret for file-backend storage.

    Kept as a stable helper so callers do not reach for ``base64`` directly.
    This is **not** encryption; it prevents only over-the-shoulder reads.
    """
    return base64.b64encode(clear.encode()).decode()


# ── Auth0 token cache (on-disk, per-credential) ───────────────────────────────


@dataclass
class CachedToken:
    access_token: str
    token_type: str
    expires_at: float
    scope: str | None = None
    # Present only for device-flow tokens. Refresh tokens for the M2M grant
    # don't exist; the SDK just fetches a new one with the client secret.
    refresh_token: str | None = None

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expires_at - 60)

    def to_dict(self, *, include_refresh_token: bool = True) -> dict:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "refresh_token": self.refresh_token if include_refresh_token else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> CachedToken:
        return cls(
            access_token=str(raw["access_token"]),
            token_type=str(raw.get("token_type", "Bearer")),
            expires_at=float(raw["expires_at"]),
            scope=raw.get("scope"),
            refresh_token=raw.get("refresh_token"),
        )


def token_cache_key(*parts: str) -> str:
    """Deterministic cache key for a set of identifying parts."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()


def token_cache_path(key: str) -> Path:
    return TOKEN_CACHE_DIR / f"{key}.json"


def _refresh_token_secret_key(key: str) -> str:
    return f"token:{key}:refresh_token"


def save_token(key: str, token: CachedToken) -> None:
    """Persist *token* to disk (best-effort; cache failures never raise)."""
    try:
        include_refresh_token = True
        if token.refresh_token:
            # Prefer the OS keyring for long-lived refresh tokens. If keyring
            # is unavailable, set_secret returns "file" and the JSON fallback
            # retains the refresh token so existing headless/dev setups keep
            # working.
            include_refresh_token = (
                set_secret(_refresh_token_secret_key(key), token.refresh_token)
                != "keyring"
            )
        _secure_dir(TOKEN_CACHE_DIR)
        path = token_cache_path(key)
        atomic_write_text(
            path,
            json.dumps(
                token.to_dict(include_refresh_token=include_refresh_token)
            ) + "\n",
        )
        _secure_file(path)
    except Exception as exc:
        logger.debug("token cache write failed for %s: %s", key, exc)


def load_token(key: str) -> CachedToken | None:
    """Return a still-valid cached token or ``None``."""
    path = token_cache_path(key)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        token = CachedToken.from_dict(raw)
    except Exception as exc:
        logger.debug("token cache read failed for %s: %s", key, exc)
        return None
    if not token.refresh_token:
        token.refresh_token = get_secret(_refresh_token_secret_key(key))
    if token.is_expired:
        return None
    return token


def clear_token(key: str) -> None:
    """Delete the cached token file for *key* (no error if missing)."""
    path = token_cache_path(key)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("token cache delete failed for %s: %s", key, exc)
    delete_secret(_refresh_token_secret_key(key))
