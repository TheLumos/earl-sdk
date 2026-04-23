"""OAuth 2.0 Device Authorization Grant (Device Code flow) helpers.

Implements just the three endpoints we need against an Auth0 tenant:

1. ``POST /oauth/device/code``            — start the authorization
2. ``POST /oauth/token`` (device_code)    — poll until the user approves
3. ``POST /oauth/token`` (refresh_token)  — refresh an expired access token

Kept independent of :mod:`earl_sdk.auth` / :mod:`earl_sdk.api` so it's easy to
unit test without touching the rest of the SDK.

References:
- RFC 8628 (OAuth 2.0 Device Authorization Grant)
- Auth0 docs: https://auth0.com/docs/get-started/authentication-and-authorization-flows/device-authorization-flow
- Auth0 Organizations + Device Flow:
  https://auth0.com/docs/manage-users/organizations/using-organizations-flows/device-authorization-flow
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import AuthenticationError, EarlError

logger = logging.getLogger("earl_sdk.device_flow")

DEFAULT_SCOPES: tuple[str, ...] = ("openid", "profile", "email", "offline_access")

# Auth0-specific error codes that are expected / recoverable during polling.
_PENDING = "authorization_pending"
_SLOW_DOWN = "slow_down"
_EXPIRED = "expired_token"
_DENIED = "access_denied"


@dataclass(frozen=True)
class DeviceAuthorization:
    """Response from ``POST /oauth/device/code``."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int  # seconds until device_code expires
    interval: int  # recommended seconds between polls

    @property
    def expires_at(self) -> float:
        return time.time() + self.expires_in


@dataclass
class DeviceTokenResponse:
    """Token response from the final ``/oauth/token`` call (or a refresh)."""

    access_token: str
    expires_in: int
    token_type: str = "Bearer"
    refresh_token: str | None = None
    scope: str = ""
    id_token: str | None = None
    # Absolute clock time when ``access_token`` expires (filled in by caller).
    expires_at: float = field(default_factory=lambda: 0.0)

    def with_expiry(self) -> DeviceTokenResponse:
        """Return a copy with ``expires_at`` set from ``expires_in``."""
        return DeviceTokenResponse(
            access_token=self.access_token,
            expires_in=self.expires_in,
            token_type=self.token_type,
            refresh_token=self.refresh_token,
            scope=self.scope,
            id_token=self.id_token,
            expires_at=time.time() + max(0, self.expires_in),
        )


class DeviceFlowError(EarlError):
    """Raised when the device flow fails for non-recoverable reasons."""


# ── Low-level HTTP helpers ────────────────────────────────────────────────────


def _post_form(url: str, form: dict[str, str], *, timeout: float = 15.0) -> tuple[int, dict]:
    """POST ``application/x-www-form-urlencoded`` and return ``(status, json)``.

    Returns the JSON body for both 2xx and 4xx responses; callers inspect the
    ``error`` field to distinguish retryable states from fatal ones. Only raw
    network / JSON-parse errors propagate as :class:`EarlError`.
    """
    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "EarlSDK-DeviceFlow/1.0",
        },
    )
    logger.debug("device-flow POST %s (fields=%s)", url, sorted(form.keys()))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else {}
            return resp.status, data
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp is not None else ""
        try:
            data = json.loads(raw) if raw else {"error": "http_error", "error_description": str(e)}
        except json.JSONDecodeError:
            data = {"error": "http_error", "error_description": raw or str(e)}
        return e.code, data
    except urllib.error.URLError as e:
        raise EarlError(f"Cannot reach Auth0 at {url}: {e}") from e
    except (TimeoutError, ConnectionError, OSError) as e:
        raise EarlError(f"Network error talking to Auth0: {e}") from e


# ── Public API ────────────────────────────────────────────────────────────────


def start_device_authorization(
    *,
    domain: str,
    client_id: str,
    audience: str,
    scopes: list[str] | tuple[str, ...] | None = None,
    organization: str | None = None,
    timeout: float = 15.0,
) -> DeviceAuthorization:
    """Begin the device flow by requesting a user_code and device_code."""
    if not domain:
        raise ValueError("Auth0 domain is required")
    if not client_id:
        raise ValueError("public Auth0 client_id is required for the device flow")
    if not audience:
        raise ValueError("audience is required so the issued token targets the EARL API")

    url = f"https://{domain}/oauth/device/code"
    form: dict[str, str] = {
        "client_id": client_id,
        "scope": " ".join(scopes or DEFAULT_SCOPES),
        "audience": audience,
    }
    if organization:
        # Auth0 passes organization through as a query/body param on the
        # authorize side; it ends up embedded in user_code -> authorize redirect.
        form["organization"] = organization

    status, data = _post_form(url, form, timeout=timeout)
    if status != 200:
        err = data.get("error", "unknown_error")
        desc = data.get("error_description") or data.get("message") or ""
        raise DeviceFlowError(
            f"Auth0 rejected device authorization request ({err}): {desc}".rstrip(": ")
        )

    try:
        return DeviceAuthorization(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            verification_uri_complete=data.get(
                "verification_uri_complete", data["verification_uri"]
            ),
            expires_in=int(data.get("expires_in", 900)),
            interval=int(data.get("interval", 5)),
        )
    except KeyError as e:
        raise DeviceFlowError(f"Unexpected Auth0 device response: missing {e.args[0]}") from e


def poll_for_token(
    *,
    domain: str,
    client_id: str,
    device_code: str,
    interval: int,
    expires_in: int,
    on_pending: Callable[[float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> DeviceTokenResponse:
    """Poll Auth0 until the user approves the device, the code expires, or fails.

    Parameters
    ----------
    on_pending
        Optional callback invoked after every ``authorization_pending`` /
        ``slow_down`` response with the seconds-remaining-until-expiry.
        Useful for progress UX; must not block for long.
    sleep, now
        Injectable so unit tests can fast-forward time deterministically.
    """
    url = f"https://{domain}/oauth/token"
    deadline = now() + max(0, expires_in)
    poll_interval = max(1, int(interval))

    while True:
        if now() >= deadline:
            raise DeviceFlowError(
                "Device code expired before the user completed sign-in. "
                "Re-run `earl auth login` to generate a fresh code."
            )

        status, data = _post_form(
            url,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": client_id,
            },
        )

        if status == 200 and "access_token" in data:
            tok = DeviceTokenResponse(
                access_token=data["access_token"],
                expires_in=int(data.get("expires_in", 3600)),
                token_type=data.get("token_type", "Bearer"),
                refresh_token=data.get("refresh_token"),
                scope=data.get("scope", ""),
                id_token=data.get("id_token"),
            )
            return tok.with_expiry()

        err = (data.get("error") or "").lower()
        if err == _PENDING:
            if on_pending:
                on_pending(deadline - now())
            sleep(poll_interval)
            continue
        if err == _SLOW_DOWN:
            # RFC 8628: increase the poll interval by at least 5s.
            poll_interval += 5
            if on_pending:
                on_pending(deadline - now())
            sleep(poll_interval)
            continue
        if err == _EXPIRED:
            raise DeviceFlowError(
                "Auth0 reports the device code has expired. Re-run `earl auth login`."
            )
        if err == _DENIED:
            raise AuthenticationError(
                "Sign-in was denied in the browser. Re-run `earl auth login` to try again."
            )
        # Fall-through: some other error (invalid_grant, invalid_client, ...).
        desc = data.get("error_description") or data.get("message") or ""
        raise DeviceFlowError(f"Auth0 returned {err or 'unexpected error'}: {desc}".rstrip(": "))


def refresh_access_token(
    *,
    domain: str,
    client_id: str,
    refresh_token: str,
    timeout: float = 15.0,
) -> DeviceTokenResponse:
    """Exchange a refresh token for a fresh access token (device-flow profiles).

    Auth0 may rotate the refresh token on every call; when it does the new
    one is in the response and must replace the stored value.
    """
    if not refresh_token:
        raise ValueError("refresh_token is required")

    url = f"https://{domain}/oauth/token"
    status, data = _post_form(
        url,
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
        timeout=timeout,
    )

    if status != 200 or "access_token" not in data:
        err = data.get("error") or "refresh_failed"
        desc = data.get("error_description") or data.get("message") or ""
        if err in {"invalid_grant", "invalid_request"}:
            # Most commonly: refresh token was rotated/revoked, or the user
            # was removed from the org. Tell the caller to re-login.
            raise AuthenticationError(
                f"Refresh token no longer valid ({err}): {desc}. "
                "Run `earl auth login` to sign in again.".rstrip(": ")
            )
        raise DeviceFlowError(f"Auth0 refresh failed ({err}): {desc}".rstrip(": "))

    tok = DeviceTokenResponse(
        access_token=data["access_token"],
        expires_in=int(data.get("expires_in", 3600)),
        token_type=data.get("token_type", "Bearer"),
        # Auth0 only returns a new refresh_token when rotation is configured.
        refresh_token=data.get("refresh_token") or refresh_token,
        scope=data.get("scope", ""),
        id_token=data.get("id_token"),
    )
    return tok.with_expiry()


def refresh_access_token_with_organization(
    *,
    domain: str,
    client_id: str,
    refresh_token: str,
    organization: str,
    audience: str | None = None,
    scopes: list[str] | tuple[str, ...] | None = None,
    timeout: float = 15.0,
) -> DeviceTokenResponse:
    """Exchange a refresh token for an access token scoped to an organization.

    Called during interactive login after the user picks which org to sign in
    as. The refresh token minted by the initial (no-org) device flow is valid
    for the user's whole identity on the tenant; Auth0 will re-issue an access
    token that includes ``org_id`` / ``org_name`` claims when we pass the
    ``organization`` parameter here — no second browser round-trip required.

    Reference: https://auth0.com/docs/secure/tokens/refresh-tokens/refresh-tokens-with-organizations
    """
    if not refresh_token:
        raise ValueError("refresh_token is required")
    if not organization:
        raise ValueError("organization is required — call refresh_access_token() for no-org refreshes")

    url = f"https://{domain}/oauth/token"
    form: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
        "organization": organization,
    }
    if audience:
        # Not strictly required — Auth0 remembers the original audience — but
        # passing it explicitly avoids ambiguity when an app targets multiple
        # audiences.
        form["audience"] = audience
    if scopes:
        form["scope"] = " ".join(scopes)

    status, data = _post_form(url, form, timeout=timeout)
    if status != 200 or "access_token" not in data:
        err = (data.get("error") or "refresh_failed").lower()
        desc = data.get("error_description") or data.get("message") or ""
        # Auth0 surfaces a very specific error when the user isn't a member of
        # the requested org (or it isn't enabled on this application). Map it
        # to AuthenticationError so the CLI can show a helpful hint.
        if err in {"invalid_grant", "invalid_request", "access_denied", "unauthorized_client"}:
            raise AuthenticationError(
                f"Could not mint an access token for org={organization}: {err}"
                + (f" — {desc}" if desc else "")
                + ". Check that the user is a member of the org, the org has the "
                "right Connection enabled, and the Auth0 Native app has "
                "organization_usage set to 'allow'."
            )
        raise DeviceFlowError(
            f"Auth0 org-scoped refresh failed ({err}): {desc}".rstrip(": ")
        )

    tok = DeviceTokenResponse(
        access_token=data["access_token"],
        expires_in=int(data.get("expires_in", 3600)),
        token_type=data.get("token_type", "Bearer"),
        refresh_token=data.get("refresh_token") or refresh_token,
        scope=data.get("scope", ""),
        id_token=data.get("id_token"),
    )
    return tok.with_expiry()


# ── Token introspection (no signature verification) ─────────────────────────


def decode_jwt_payload(jwt: str) -> dict[str, Any]:
    """Decode the JSON payload of a JWT without verifying its signature.

    We use this *only* on tokens we just received directly from Auth0 over
    HTTPS, so the trust boundary has already been crossed via TLS. Never use
    this to validate tokens forwarded from untrusted sources — that's the
    orchestrator's job.

    Returns an empty dict on any parse failure so callers can treat the claims
    as "best-effort" metadata for UX (e.g. naming profiles) without having to
    handle exceptions.
    """
    if not jwt or jwt.count(".") < 2:
        return {}
    payload_b64 = jwt.split(".", 2)[1]
    # JWTs use URL-safe base64 without padding; re-pad for b64decode.
    padding = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + padding)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("could not decode JWT payload: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class TokenOrgInfo:
    """Organization metadata extracted from an Auth0 access token."""

    org_id: str
    org_name: str
    subject: str  # Auth0 `sub` claim (e.g. "auth0|abc123")
    email: str = ""

    @property
    def has_org(self) -> bool:
        return bool(self.org_id)


def extract_org_info(access_token: str) -> TokenOrgInfo:
    """Pull organization + user fields out of an Auth0 access token.

    Works for both Auth0's standard ``org_id`` / ``org_name`` claims and the
    namespaced custom claims frequently added via a Post-Login Action.
    """
    claims = decode_jwt_payload(access_token)
    org_id = str(claims.get("org_id") or claims.get("https://earl/org_id") or "")
    org_name = str(claims.get("org_name") or claims.get("https://earl/org_name") or "")
    email = str(claims.get("email") or claims.get("https://earl/email") or "")
    sub = str(claims.get("sub") or "")
    return TokenOrgInfo(org_id=org_id, org_name=org_name, subject=sub, email=email)


def slugify_org(org_name: str, org_id: str) -> str:
    """Derive a short, filesystem-safe profile suffix from org metadata.

    Preference order:
    1. ``org_name`` lowercased, with non-alphanum → ``-`` and collapsed runs.
    2. Last 8 chars of ``org_id`` when the name is missing/unusable.
    3. Empty string when neither is present (caller should fall back).
    """
    if org_name:
        cleaned = []
        prev_dash = False
        for ch in org_name.lower():
            if ch.isalnum():
                cleaned.append(ch)
                prev_dash = False
            elif not prev_dash:
                cleaned.append("-")
                prev_dash = True
        slug = "".join(cleaned).strip("-")
        if slug:
            return slug[:32]
    if org_id:
        return org_id[-8:]
    return ""
