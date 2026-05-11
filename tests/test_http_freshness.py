"""Tests for SDK freshness handling in :mod:`earl_sdk._http`.

These cover Phase 8 of the runaway-recovery hardening work: the SDK
emits a one-shot ``UserWarning`` when the orchestrator advertises a
newer release in the ``Earl-Latest-Sdk-Version`` response header, and
maps a 426 ``Upgrade Required`` to a structured ``EarlError`` with an
explicit upgrade hint.

We use ``httpx.MockTransport`` so the tests are hermetic — no real
HTTP, no real orchestrator, no real keychain.
"""

from __future__ import annotations

import warnings

import httpx
import pytest

import earl_sdk
import earl_sdk._http as http_mod
from earl_sdk._http import (
    close_shared_client,
    request_json,
    set_shared_client,
)
from earl_sdk.exceptions import EarlError


@pytest.fixture(autouse=True)
def reset_freshness_flag(monkeypatch):
    """The freshness warning is one-shot per process; reset between tests."""
    monkeypatch.setattr(http_mod, "_FRESHNESS_WARNING_EMITTED", False)
    yield
    close_shared_client()


def _client_returning(status: int, headers: dict[str, str], body: dict | None = None):
    body = body if body is not None else {"ok": True}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, json=body)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, base_url="http://test")


class TestUserAgent:
    def test_user_agent_includes_version_and_runtime(self):
        ua = http_mod._build_user_agent()
        # We don't pin the exact version — we pin the SHAPE so a future
        # version bump doesn't break the test, but a malformed UA does.
        assert ua.startswith("earl-sdk-python/")
        assert " httpx/" in ua
        assert " python/" in ua


class TestFreshnessWarning:
    def test_warning_emitted_when_sdk_below_latest(self, monkeypatch):
        # Pretend the SDK is much older than what the server advertises.
        monkeypatch.setattr(earl_sdk, "__version__", "0.1.0")

        client = _client_returning(
            200, {"Earl-Latest-Sdk-Version": "999.0.0"}
        )
        set_shared_client(client)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request_json("GET", "http://test/v1/anything")
        upgrade_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "earl-sdk" in str(w.message)
        ]
        assert len(upgrade_warnings) == 1
        assert "999.0.0" in str(upgrade_warnings[0].message)

    def test_warning_only_emitted_once_per_process(self, monkeypatch):
        monkeypatch.setattr(earl_sdk, "__version__", "0.1.0")

        client = _client_returning(
            200, {"Earl-Latest-Sdk-Version": "999.0.0"}
        )
        set_shared_client(client)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request_json("GET", "http://test/v1/a")
            request_json("GET", "http://test/v1/b")
            request_json("GET", "http://test/v1/c")
        upgrade_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "earl-sdk" in str(w.message)
        ]
        assert len(upgrade_warnings) == 1

    def test_no_warning_when_sdk_at_or_above_latest(self, monkeypatch):
        monkeypatch.setattr(earl_sdk, "__version__", "5.0.0")

        client = _client_returning(
            200, {"Earl-Latest-Sdk-Version": "1.0.0"}
        )
        set_shared_client(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request_json("GET", "http://test/v1/anything")
        upgrade_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "earl-sdk" in str(w.message)
        ]
        assert upgrade_warnings == []

    def test_no_warning_when_disabled_by_env(self, monkeypatch):
        monkeypatch.setattr(earl_sdk, "__version__", "0.1.0")
        monkeypatch.setenv("EARL_DISABLE_VERSION_NUDGE", "1")

        client = _client_returning(
            200, {"Earl-Latest-Sdk-Version": "999.0.0"}
        )
        set_shared_client(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request_json("GET", "http://test/v1/anything")
        upgrade_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "earl-sdk" in str(w.message)
        ]
        assert upgrade_warnings == []

    def test_no_warning_when_header_absent(self):
        client = _client_returning(200, {})
        set_shared_client(client)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request_json("GET", "http://test/v1/anything")
        upgrade_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "earl-sdk" in str(w.message)
        ]
        assert upgrade_warnings == []


class TestUpgradeRequired:
    def test_426_maps_to_earl_error_with_upgrade_hint(self):
        client = _client_returning(
            426,
            {
                "Earl-Min-Supported-Sdk-Version": "10.0.0",
                "Earl-Latest-Sdk-Version": "10.5.0",
            },
            body={
                "error": "sdk_upgrade_required",
                "message": "earl-sdk too old",
                "min_supported_sdk_version": "10.0.0",
            },
        )
        set_shared_client(client)
        with pytest.raises(EarlError) as exc_info:
            request_json("GET", "http://test/v1/anything")
        err = exc_info.value
        assert err.status_code == 426
        assert "pip install -U earl-sdk" in (err.hint or "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
