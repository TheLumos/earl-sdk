"""Auth0 authentication for Earl SDK.

Supports two grants:

- ``m2m`` — classic client_credentials flow (``client_id`` + ``client_secret``).
  Used by server-side integrations and CI.
- ``device`` — OAuth 2.0 Device Authorization Grant (RFC 8628). The CLI's
  ``earl auth login`` acquires an access token + refresh token via a browser
  on the user's laptop, and ``Auth0Client`` refreshes the access token as
  needed (re-prompting the user to log in only when the refresh token is
  revoked or rotated out).

Access tokens are cached on disk (0600) per
``(client_id, audience, organization, domain, auth_kind)`` so long-running
agent workflows do not re-hit Auth0 on every ``earl`` command.  See
:mod:`earl_sdk.auth_storage` for the cache backend and
:mod:`earl_sdk.device_flow` for the device-flow primitives.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Literal

from .auth_storage import (
    CachedToken,
    clear_token,
    load_token,
    save_token,
    token_cache_key,
)
from .exceptions import AuthenticationError

AuthKind = Literal["m2m", "device"]


@dataclass
class TokenInfo:
    """In-memory access-token record."""

    access_token: str
    token_type: str
    expires_at: float  # unix seconds
    scope: str | None = None
    refresh_token: str | None = None  # device-flow only

    @property
    def is_expired(self) -> bool:
        """True when the token has <60s of life remaining."""
        return time.time() >= (self.expires_at - 60)

    def to_cached(self) -> CachedToken:
        return CachedToken(
            access_token=self.access_token,
            token_type=self.token_type,
            expires_at=self.expires_at,
            scope=self.scope,
            refresh_token=self.refresh_token,
        )

    @classmethod
    def from_cached(cls, cached: CachedToken) -> TokenInfo:
        return cls(
            access_token=cached.access_token,
            token_type=cached.token_type,
            expires_at=cached.expires_at,
            scope=cached.scope,
            refresh_token=cached.refresh_token,
        )


class Auth0Client:
    """Auth0 client-credentials helper.

    Thread-safe: concurrent :meth:`get_token` calls coalesce onto a single
    refresh.  The access token is additionally cached on disk (0600 perms
    under ``~/.earl/tokens/``) unless ``use_disk_cache=False`` or the env
    var ``EARL_NO_TOKEN_CACHE=1`` is set.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        organization: str,
        domain: str,
        audience: str,
        *,
        auth_kind: AuthKind = "m2m",
        refresh_token: str | None = None,
        use_disk_cache: bool = True,
    ):
        """
        Args:
            client_id: Auth0 application client ID. For ``auth_kind="m2m"``
                this is the M2M app's ID; for ``auth_kind="device"`` it is
                the public Native app's ID.
            client_secret: Auth0 M2M application client secret. Ignored when
                ``auth_kind="device"``; pass an empty string.
            organization: Auth0 organization ID (``org_xxx``) or empty string.
            domain: Auth0 tenant domain (required; no default – callers must
                resolve the env-specific value via ``EnvironmentConfig``).
            audience: API audience (required; same note as ``domain``).
            auth_kind: ``"m2m"`` (default) or ``"device"``. See module docstring.
            refresh_token: Initial refresh token for device-flow clients. The
                refresh token is also persisted in the on-disk cache so that
                subsequent SDK processes can reuse it without re-prompting.
            use_disk_cache: Disable to bypass the on-disk token cache.
        """
        if not domain:
            raise ValueError("Auth0Client requires an explicit domain")
        if not audience:
            raise ValueError("Auth0Client requires an explicit audience")
        if auth_kind == "m2m" and not client_secret:
            raise ValueError("Auth0Client(auth_kind='m2m') requires a client_secret")
        self.client_id = client_id
        self.client_secret = client_secret
        self.organization = organization
        self.domain = domain
        self.audience = audience
        self.auth_kind: AuthKind = auth_kind

        import os

        self._use_disk_cache = use_disk_cache and not os.getenv("EARL_NO_TOKEN_CACHE")
        # Include auth_kind so an M2M cache entry cannot collide with a device
        # entry for the same (client_id, audience, org, domain).
        self._cache_key = token_cache_key(client_id, audience, organization, domain, auth_kind)

        self._token_info: TokenInfo | None = None
        self._token_lock = threading.Lock()

        if self._use_disk_cache:
            cached = load_token(self._cache_key)
            if cached is not None:
                self._token_info = TokenInfo.from_cached(cached)
                # Prefer the freshest refresh token we know about.
                if refresh_token is None:
                    refresh_token = cached.refresh_token

        # Seed refresh token from constructor if provided (e.g. straight after
        # `earl auth login`). May be None for M2M — that's fine.
        if refresh_token and (self._token_info is None or not self._token_info.refresh_token):
            if self._token_info is None:
                self._token_info = TokenInfo(
                    access_token="",
                    token_type="Bearer",
                    expires_at=0.0,
                    refresh_token=refresh_token,
                )
            else:
                self._token_info.refresh_token = refresh_token

    @property
    def token_url(self) -> str:
        return f"https://{self.domain}/oauth/token"

    def get_token(self) -> str:
        """Return a valid access token, refreshing (and caching) as needed."""
        info = self._token_info
        if info and info.access_token and not info.is_expired:
            return info.access_token

        with self._token_lock:
            cur = self._token_info
            if cur and cur.access_token and not cur.is_expired:
                return cur.access_token

            if self.auth_kind == "device":
                self._token_info = self._refresh_device_token(cur)
            else:
                self._token_info = self._fetch_m2m_token()

            if self._use_disk_cache:
                save_token(self._cache_key, self._token_info.to_cached())
            return self._token_info.access_token

    def _refresh_device_token(self, cur: TokenInfo | None) -> TokenInfo:
        """Refresh a device-flow access token via the stored refresh token."""
        from .device_flow import refresh_access_token

        if cur is None or not cur.refresh_token:
            raise AuthenticationError(
                "No cached device-flow credentials. Run `earl auth login` first.",
                code="no_refresh_token",
                hint="Run `earl auth login --env <env>` to sign in via the browser.",
                details={"domain": self.domain},
            )
        resp = refresh_access_token(
            domain=self.domain,
            client_id=self.client_id,
            refresh_token=cur.refresh_token,
        )
        return TokenInfo(
            access_token=resp.access_token,
            token_type=resp.token_type,
            expires_at=resp.expires_at,
            scope=resp.scope or None,
            refresh_token=resp.refresh_token,
        )

    def _fetch_m2m_token(self) -> TokenInfo:
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
        }
        if self.organization:
            payload["organization"] = self.organization

        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            self.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return TokenInfo(
                    access_token=result["access_token"],
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=time.time() + result.get("expires_in", 86400),
                    scope=result.get("scope"),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            err_code: str | None = None
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error_description", error_data.get("error", str(e)))
                raw_err = error_data.get("error")
                if isinstance(raw_err, str):
                    err_code = raw_err
            except json.JSONDecodeError:
                error_data = {}
                error_msg = error_body or str(e)
            hint = None
            if e.code in (401, 403):
                hint = (
                    "Check that EARL_CLIENT_ID / EARL_CLIENT_SECRET match the "
                    "target environment, or run `earl auth login --env <env>` "
                    "for device-flow."
                )
            raise AuthenticationError(
                f"Failed to authenticate with Auth0: {error_msg}",
                status_code=e.code,
                code=err_code or ("unauthorized" if e.code == 401 else None),
                url=self.token_url,
                method="POST",
                hint=hint,
                details={**error_data, "domain": self.domain},
            ) from e
        except Exception as e:
            raise AuthenticationError(
                f"Failed to connect to Auth0: {e}",
                code="auth0_unreachable",
                url=self.token_url,
                method="POST",
                hint="Verify network connectivity and that the Auth0 domain is reachable.",
                details={"domain": self.domain},
            ) from e

    def invalidate_token(self) -> None:
        """Drop the in-memory and on-disk token; next call refreshes."""
        self._token_info = None
        if self._use_disk_cache:
            clear_token(self._cache_key)

    def get_headers(self) -> dict[str, str]:
        token = self.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        if self.organization:
            headers["X-Organization-Id"] = self.organization
        return headers
