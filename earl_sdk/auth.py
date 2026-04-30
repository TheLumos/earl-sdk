"""Auth0 authentication for Earl SDK.

Supports three grants:

- ``m2m`` — classic client_credentials flow (``client_id`` + ``client_secret``).
  Used by CI and any automation with a provisioned service account.
- ``pkce`` — Authorization Code + PKCE over a loopback redirect. The
  preferred interactive flow: ``earl login`` opens the browser, Auth0
  Universal Login handles the org picker for multi-tenant users, and the
  CLI receives an access token + refresh token.
- ``device`` — OAuth 2.0 Device Authorization Grant (RFC 8628). Headless
  fallback for environments without a browser.

Both ``pkce`` and ``device`` issue refresh tokens and drive the same refresh
code path — :meth:`Auth0Client.get_token` treats them uniformly.

Access tokens are cached on disk (0600) per
``(client_id, audience, organization, domain, auth_kind)`` so long-running
agent workflows do not re-hit Auth0 on every ``earl`` command.  See
:mod:`earl_sdk.auth_storage` for the cache backend and
:mod:`earl_sdk.device_flow` / :mod:`earl_sdk.pkce_flow` for the primitives.
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

AuthKind = Literal["m2m", "pkce", "device"]
# Flows that obtain access tokens via a refresh_token rather than
# client_credentials. Used to centralise the "is this a browser-issued token?"
# check so adding a new interactive flow is a one-line change.
BROWSER_KINDS: frozenset[AuthKind] = frozenset({"pkce", "device"})


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
                this is the M2M app's ID; for ``pkce``/``device`` it is the
                public Native app's ID.
            client_secret: Auth0 M2M application client secret. Ignored when
                ``auth_kind`` is ``pkce`` or ``device``; pass an empty string.
            organization: Auth0 organization ID (``org_xxx``) or empty
                string. Only used as a body parameter on the ``m2m`` token
                request; the resulting access token carries ``org_id`` in its
                claims, which is the sole source of tenancy for the backend.
            domain: Auth0 tenant domain (required; no default – callers must
                resolve the env-specific value via ``EnvironmentConfig``).
            audience: API audience (required; same note as ``domain``).
            auth_kind: ``"m2m"`` (default), ``"pkce"``, or ``"device"``.
            refresh_token: Initial refresh token for browser-issued tokens
                (``pkce``/``device``). Also persisted in the on-disk cache.
            use_disk_cache: Disable to bypass the on-disk token cache.
        """
        if not domain:
            raise ValueError("Auth0Client requires an explicit domain")
        if not audience:
            raise ValueError("Auth0Client requires an explicit audience")
        if auth_kind not in {"m2m", "pkce", "device"}:
            raise ValueError("auth_kind must be one of: m2m, pkce, device")
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
        # The "requested" key is pinned to whatever org the caller asked for;
        # it's what we look up on first boot. After we actually have a token
        # we switch to the "resolved" key (keyed on the ``org_id`` claim the
        # token actually carries) so a user that resolved multiple orgs
        # under the same profile doesn't clobber their own caches. Include
        # auth_kind so an M2M cache entry cannot collide with a device
        # entry for the same (client_id, audience, org, domain).
        self._requested_cache_key = token_cache_key(
            client_id, audience, organization, domain, auth_kind
        )
        self._cache_key = self._requested_cache_key

        self._token_info: TokenInfo | None = None
        self._token_lock = threading.Lock()

        if self._use_disk_cache:
            cached = load_token(self._requested_cache_key)
            if cached is not None:
                self._token_info = TokenInfo.from_cached(cached)
                # Switch to the resolved key so subsequent writes go to the
                # right file, and migrate the existing cache file if needed.
                self._promote_resolved_key(cached.access_token, cached)
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

    def _promote_resolved_key(
        self, access_token: str, cached: "CachedToken | None"
    ) -> None:
        """Switch ``_cache_key`` to one keyed on the token's resolved ``org_id``.

        Called after every successful token exchange and on first load. If
        the resolved ``org_id`` differs from what the caller requested
        (e.g. the caller passed ``organization=""`` and the IdP attached
        a default org), we migrate the cache file onto the resolved key
        and delete the old one so the next run hits the right cache.
        Failures here are intentionally swallowed: token-cache keying is
        a correctness/perf optimisation, not a security boundary.
        """
        if not self._use_disk_cache or not access_token:
            return
        try:
            from .claims import project_access_token

            claims = project_access_token(access_token)
        except Exception:
            return
        resolved_org = claims.org_id or self.organization
        resolved_key = token_cache_key(
            self.client_id, self.audience, resolved_org, self.domain, self.auth_kind
        )
        if resolved_key == self._cache_key:
            return
        old_key = self._cache_key
        self._cache_key = resolved_key
        if cached is not None:
            try:
                save_token(resolved_key, cached)
            except Exception:
                return
        if old_key != resolved_key:
            try:
                clear_token(old_key)
            except Exception:
                pass

    def get_token(self) -> str:
        """Return a valid access token, refreshing (and caching) as needed."""
        info = self._token_info
        if info and info.access_token and not info.is_expired:
            return info.access_token

        with self._token_lock:
            cur = self._token_info
            if cur and cur.access_token and not cur.is_expired:
                return cur.access_token

            if self.auth_kind in BROWSER_KINDS:
                self._token_info = self._refresh_browser_token(cur)
            else:
                self._token_info = self._fetch_m2m_token()

            cached = self._token_info.to_cached()
            self._promote_resolved_key(self._token_info.access_token, cached)
            if self._use_disk_cache:
                save_token(self._cache_key, cached)
            return self._token_info.access_token

    def _refresh_browser_token(self, cur: TokenInfo | None) -> TokenInfo:
        """Refresh a PKCE- or device-flow access token via the stored refresh token.

        Both flows use ``grant_type=refresh_token`` against the same
        ``/oauth/token`` endpoint with the public Native ``client_id``; we
        just pick the helper module whose error taxonomy matches the flow
        that produced the refresh token. Either works for either, so we
        prefer the PKCE helper by default.
        """
        if self.auth_kind == "device":
            from .device_flow import refresh_access_token
        else:
            from .pkce_flow import refresh_access_token

        if cur is None or not cur.refresh_token:
            raise AuthenticationError(
                "No cached credentials. Run `earl login` first.",
                code="no_refresh_token",
                hint=(
                    "Run `earl login --env <env>` to sign in via the browser, "
                    "or set EARL_CLIENT_ID/EARL_CLIENT_SECRET/EARL_ORG_ID for "
                    "automation."
                ),
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

    # Backwards-compat shim: existing tests / external callers still reach
    # ``_refresh_device_token`` directly. Route it through the unified path.
    def _refresh_device_token(self, cur: TokenInfo | None) -> TokenInfo:
        return self._refresh_browser_token(cur)

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
        """Return request headers carrying the access token.

        The orchestrator derives tenancy **exclusively** from the validated
        JWT's ``org_id`` claim — never from request headers. We therefore do
        not (and must not) send an ``X-Organization-Id`` header; doing so
        would create a second, trust-dubious source of org context and
        invite confusion during security review.
        """
        return {"Authorization": f"Bearer {self.get_token()}"}
