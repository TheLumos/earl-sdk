"""Shared helpers for building :class:`EarlClient` from an ``AuthProfile`` and
running a credential-validation test against the orchestrator.

Both the command CLI (``earl auth profile …``) and the interactive CLI need to
do the same two operations: turn a saved profile into client construction
kwargs (with the right shape for M2M vs. PKCE/device profiles), and probe the
API to record whether the profile currently works. Keeping that logic in one
place avoids the kind of drift that previously broke browser profiles when the
interactive flow hard-coded ``auth_kind='m2m'``.

The helpers here are intentionally side-effect-light: they do not import from
the CLI/interactive layers, do not print anything, and return plain data so
both surfaces can render the result however they like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .interactive.storage.config_store import AuthProfile, ConfigStore


@dataclass(frozen=True)
class ProfileTestResult:
    """Outcome of a single ``test_connection`` against a profile.

    ``ok`` mirrors the boolean returned by :meth:`EarlClient.test_connection`.
    ``error`` captures the short human-readable failure reason (truncated to
    keep config.json compact) when ``ok`` is ``False`` or initialization
    raised; it is ``""`` on success.
    """

    ok: bool
    error: str = ""


def build_client_kwargs(profile: AuthProfile) -> dict[str, Any]:
    """Return :class:`EarlClient` ``__init__`` kwargs for *profile*.

    For browser-issued profiles (``pkce``/``device``) the client must be told
    not to expect a client secret and must be handed the refresh token instead.
    Doing this in one place prevents the ``Auth0Client(auth_kind='m2m')
    requires a client_secret`` regression that hit the interactive CLI when
    it tried to rebuild a PKCE profile.
    """
    kwargs: dict[str, Any] = {
        "client_id": profile.client_id,
        "organization": profile.organization or "",
        "environment": profile.environment,
        "auth_kind": profile.auth_kind,
    }
    if profile.auth_kind in ("pkce", "device"):
        kwargs["client_secret"] = ""
        kwargs["refresh_token"] = profile.refresh_token_clear()
    else:
        kwargs["client_secret"] = profile.secret_clear()
    return kwargs


def build_client(profile: AuthProfile):
    """Construct an :class:`EarlClient` from *profile*.

    Lazy-imports :class:`EarlClient` so this helper stays cheap to import
    from CLI argparse setup paths that may not need it.
    """
    from . import EarlClient

    return EarlClient(**build_client_kwargs(profile))


def test_profile(profile: AuthProfile) -> ProfileTestResult:
    """Build a client and call ``test_connection``; never raises.

    Any construction error or transport failure is captured as a short string
    in :attr:`ProfileTestResult.error` so the caller can both surface it to
    the user and persist it on the profile via
    :meth:`AuthProfile.mark_test_result`.
    """
    try:
        client = build_client(profile)
    except Exception as exc:
        return ProfileTestResult(ok=False, error=f"init failed: {exc}")

    try:
        ok = bool(client.test_connection())
    except Exception as exc:
        return ProfileTestResult(ok=False, error=str(exc) or exc.__class__.__name__)

    if ok:
        return ProfileTestResult(ok=True)
    return ProfileTestResult(ok=False, error="test_connection returned False")


def test_and_record(
    store: ConfigStore,
    profile: AuthProfile,
) -> ProfileTestResult:
    """Run :func:`test_profile` and persist the result on the named profile.

    Convenient one-shot used by both the command CLI's ``auth profile list``
    refresh path and the interactive ``_view_profile`` test action.
    """
    result = test_profile(profile)
    store.record_profile_test(profile.name, ok=result.ok, error=result.error)
    return result
