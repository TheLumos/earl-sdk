"""Auth0 claims projection helper (SDK-side, unverified).

Mirrors :mod:`src.orchestrator.auth_claims` so the CLI/TUI extract ``org_id``
/ ``roles`` / ``permissions`` / ``email`` from a JWT payload the same way
the backend does. We deliberately keep the two copies structurally
identical and pin them via a contract test
(``sdk/tests/test_claims_contract.py``).

Important: this module decodes payloads **without verifying signatures**.
Call :func:`project_access_token` only on tokens we just received directly
from Auth0 over HTTPS, never on tokens forwarded from an untrusted
source — that's the orchestrator's job.
"""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

# Keep in lockstep with ``src/orchestrator/auth_claims.py::AUTH0_NAMESPACE``.
# The SDK does not read environment for this because a misconfigured CI
# shouldn't change how the CLI decodes tokens.
AUTH0_NAMESPACE = "https://earl.thelumos.xyz"

_LEGACY_NAMESPACES: tuple[str, ...] = (
    "https://earl.thelumos.ai",
    "https://earl-api.thelumos.xyz",
    "https://earl",
)


def _candidate_namespaces() -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for ns in (AUTH0_NAMESPACE, *_LEGACY_NAMESPACES):
        if ns and ns not in seen:
            seen.add(ns)
            out.append(ns)
    return tuple(out)


def _ns_keys(suffix: str) -> tuple[str, ...]:
    return tuple(f"{ns}/{suffix}" for ns in _candidate_namespaces())


ORG_ID_CLAIM_KEYS: tuple[str, ...] = (
    "org_id",
    *_ns_keys("org_id"),
    *_ns_keys("organization_id"),
)

ORG_NAME_CLAIM_KEYS: tuple[str, ...] = (
    "org_name",
    *_ns_keys("org_name"),
    *_ns_keys("organization_name"),
)

ROLE_CLAIM_KEYS: tuple[str, ...] = (
    "roles",
    *_ns_keys("roles"),
    *_ns_keys("user_roles"),
)

PERMISSION_CLAIM_KEYS: tuple[str, ...] = (
    "permissions",
    *_ns_keys("permissions"),
)

EMAIL_CLAIM_KEYS: tuple[str, ...] = (
    "email",
    *_ns_keys("email"),
)

NAME_CLAIM_KEYS: tuple[str, ...] = (
    "name",
    "nickname",
    *_ns_keys("name"),
)


@dataclass(frozen=True)
class TokenClaims:
    """Typed projection of an Auth0 JWT payload (SDK-side mirror)."""

    subject: str
    org_id: str
    org_name: str
    email: str
    name: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    is_m2m: bool = False
    grant_type: str = ""
    raw: dict = field(default_factory=dict)


def _first_string(payload: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _collect_strings(payload: dict, keys: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item and item not in seen:
                    seen.add(item)
                    out.append(item)
        elif isinstance(value, str) and value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _detect_m2m(payload: dict) -> tuple[bool, str]:
    grant_type = str(payload.get("gty") or "")
    sub = str(payload.get("sub") or "")
    is_m2m = grant_type == "client-credentials" or sub.endswith("@clients")
    return is_m2m, grant_type


def decode_payload(jwt: str) -> dict[str, Any]:
    """Decode the JSON payload of a JWT *without* verifying its signature.

    Returns an empty dict on any parse failure so callers can treat the
    claims as best-effort metadata (e.g. naming profiles, detecting roles
    for UI gating) without having to handle exceptions.
    """
    if not jwt or jwt.count(".") < 2:
        return {}
    payload_b64 = jwt.split(".", 2)[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + padding)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def project_claims(payload: Optional[dict]) -> TokenClaims:
    """Project an already-decoded JWT payload into a :class:`TokenClaims`."""
    if not isinstance(payload, dict):
        return TokenClaims(subject="", org_id="", org_name="", email="", name="")
    return TokenClaims(
        subject=_first_string(payload, ("sub",)),
        org_id=_first_string(payload, ORG_ID_CLAIM_KEYS),
        org_name=_first_string(payload, ORG_NAME_CLAIM_KEYS),
        email=_first_string(payload, EMAIL_CLAIM_KEYS),
        name=_first_string(payload, NAME_CLAIM_KEYS),
        roles=_collect_strings(payload, ROLE_CLAIM_KEYS),
        permissions=_collect_strings(payload, PERMISSION_CLAIM_KEYS),
        is_m2m=_detect_m2m(payload)[0],
        grant_type=_detect_m2m(payload)[1],
        raw=payload,
    )


def project_access_token(jwt: str) -> TokenClaims:
    """Decode and project an Auth0 access token in one call."""
    return project_claims(decode_payload(jwt))


__all__ = [
    "AUTH0_NAMESPACE",
    "EMAIL_CLAIM_KEYS",
    "NAME_CLAIM_KEYS",
    "ORG_ID_CLAIM_KEYS",
    "ORG_NAME_CLAIM_KEYS",
    "PERMISSION_CLAIM_KEYS",
    "ROLE_CLAIM_KEYS",
    "TokenClaims",
    "decode_payload",
    "project_access_token",
    "project_claims",
]
