"""Configuration flow — auth profiles, doctor configs, preferences.

Manage EARL credentials, external doctor API endpoints, and UI preferences.
All settings are stored locally in ``~/.earl/config.json``.
"""

from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("earl.ui")

from ..ui import (
    ask_confirm,
    ask_int,
    ask_text,
    console,
    error,
    info_panel,
    kvtable,
    muted,
    select_one,
    success,
    warn,
)
from ..storage.config_store import AuthProfile, ConfigStore, DoctorConfig, Preferences


def flow_config(store: ConfigStore, client_ref: list) -> None:
    """Top-level configuration menu."""
    while True:
        action = select_one("Configuration", [
            ("profiles", "Auth Profiles          — manage EARL API credentials for dev/test/prod environments"),
            ("doctors",  "Doctor Configs         — save and validate your doctor API endpoints for reuse"),
            ("prefs",    "Preferences            — set defaults for pipelines, parallelism, and auto-save"),
            ("test",     "Test Connection        — verify the active profile can reach the EARL API"),
            ("export",   "Show Config Path       — reveal where local settings are stored on disk"),
        ])
        if action is None:
            return
        if action == "profiles":
            _flow_profiles(store, client_ref)
        elif action == "doctors":
            _flow_doctors(store, client_ref)
        elif action == "prefs":
            _flow_preferences(store)
        elif action == "test":
            _test_connection(store, client_ref)
        elif action == "export":
            from ..storage.config_store import CONFIG_PATH
            muted(f"Config file: {CONFIG_PATH}")
            muted(f"Runs folder: {CONFIG_PATH.parent / 'runs'}")


# ── Auth Profiles ─────────────────────────────────────────────────────────────


def _flow_profiles(store: ConfigStore, client_ref: list) -> None:
    while True:
        cfg = store.config
        profiles = list(cfg.profiles.values())

        choices: list[tuple[str, str]] = []
        if profiles:
            choices.append(("__hdr__", "── Saved Profiles ──"))
            for p in profiles:
                active = " ★" if p.name == cfg.active_profile else ""
                choices.append((
                    f"view:{p.name}",
                    f"{p.name}  [{p.environment}]  org={p.organization or '(none)'}{active}",
                ))
        choices.extend([
            ("__actions__", "── Actions ──"),
            ("add",    "Add New Profile        — enter Auth0 M2M credentials for a new environment"),
            ("switch", "Switch Active Profile  — change which profile is used for API calls"),
        ])

        action = select_one("Auth Profiles", choices)
        if action is None:
            return

        if action.startswith("view:"):
            name = action.split(":", 1)[1]
            _view_profile(store, name, client_ref)
        elif action == "add":
            _add_profile(store, client_ref)
        elif action == "switch":
            _switch_profile(store, client_ref)


def _view_profile(store: ConfigStore, name: str, client_ref: list) -> None:
    p = store.config.profiles.get(name)
    if not p:
        error(f"Profile '{name}' not found")
        return

    is_active = name == store.config.active_profile
    info_panel(f"Profile: {name}" + (" ★ active" if is_active else ""), [
        f"[bold]Client ID:[/]     {p.client_id}",
        f"[bold]Secret:[/]        {'*' * 8}...{p.secret_clear()[-4:]}",
        f"[bold]Organization:[/]  {p.organization or '(none)'}",
        f"[bold]Environment:[/]   {p.environment}",
    ])

    action = select_one("Profile Actions", [
        ("activate", "Set as Active          — use this profile for all API calls"),
        ("test",     "Test Connection        — verify credentials reach the EARL API"),
        ("delete",   "Delete Profile         — remove this profile from local storage"),
    ])
    if action == "activate":
        store.set_active_profile(name)
        _rebuild_client(store, client_ref)
        success(f"'{name}' is now the active profile")
    elif action == "test":
        _test_connection(store, client_ref)
    elif action == "delete":
        if ask_confirm(f"Delete profile '{name}'?", default=False):
            store.delete_profile(name)
            if is_active:
                _rebuild_client(store, client_ref)
            success(f"Profile '{name}' deleted")


def _add_profile(store: ConfigStore, client_ref: list) -> None:
    console.print("\n[bold]Add Auth Profile[/]")
    muted("You'll need Auth0 M2M credentials from the EARL dashboard.")
    muted("These are used to authenticate SDK API calls.\n")

    name = ask_text("Profile name (e.g. 'prod', 'staging')", default="default")
    if not name:
        return
    client_id = ask_text("Auth0 Client ID")
    if not client_id:
        return
    client_secret = ask_text("Auth0 Client Secret", secret=True)
    if not client_secret:
        return
    organization = ask_text("Organization ID (leave empty if none)", default="") or ""

    env = select_one("Environment", [
        ("dev",  "dev   — development (dev-api.onlyevals.com)"),
        ("test", "test  — staging / QA (test-api.thelumos.xyz)"),
        ("prod", "prod  — production (api.earl.thelumos.ai)"),
    ], allow_back=False)
    if not env:
        return

    profile = AuthProfile(
        name=name,
        client_id=client_id,
        client_secret=AuthProfile.obfuscate(client_secret),
        organization=organization,
        environment=env,
    )
    store.upsert_profile(profile)
    _rebuild_client(store, client_ref)
    success(f"Profile '{name}' saved ({env})")


def _switch_profile(store: ConfigStore, client_ref: list) -> None:
    cfg = store.config
    if len(cfg.profiles) < 2:
        warn("Only one profile exists — nothing to switch to.")
        return

    choices = [
        (name, f"{name}  [{p.environment}]  org={p.organization or '(none)'}"
         + (" ★ active" if name == cfg.active_profile else ""))
        for name, p in cfg.profiles.items()
    ]
    pick = select_one("Switch to", choices)
    if pick:
        store.set_active_profile(pick)
        _rebuild_client(store, client_ref)
        success(f"Switched to '{pick}'")


# ── Doctor Configs ────────────────────────────────────────────────────────────


def _flow_doctors(store: ConfigStore, client_ref: list) -> None:
    while True:
        configs = store.list_doctor_configs()

        choices: list[tuple[str, str]] = []
        if configs:
            choices.append(("__hdr__", "── Saved Doctor Configs ──"))
            for dc in configs:
                type_label = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}.get(dc.type, dc.type)
                url_hint = f"  {dc.api_url}" if dc.api_url else ""
                choices.append((f"view:{dc.name}", f"{dc.name}  [{type_label}]{url_hint}"))

        choices.extend([
            ("__actions__", "── Actions ──"),
            ("add",      "Add Doctor Config      — save a new doctor API endpoint for reuse in pipelines"),
            ("validate", "Validate Doctor API    — send a test request to verify your doctor endpoint works"),
        ])

        action = select_one("Doctor Configs", choices)
        if action is None:
            return

        if action.startswith("view:"):
            name = action.split(":", 1)[1]
            _view_doctor(store, name, client_ref)
        elif action == "add":
            _add_doctor(store, client_ref)
        elif action == "validate":
            _validate_doctor(store, client_ref)


def _view_doctor(store: ConfigStore, name: str, client_ref: list) -> None:
    dc = store.config.doctor_configs.get(name)
    if not dc:
        error(f"Doctor config '{name}' not found")
        return

    type_label = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}.get(dc.type, dc.type)
    lines = [f"[bold]Type:[/]       {type_label}"]
    if dc.api_url:
        lines.append(f"[bold]URL:[/]        {dc.api_url}")
    if dc.api_key:
        lines.append(f"[bold]API Key:[/]    {'*' * 8}...{dc.key_clear()[-4:]}")
        lines.append(f"[bold]Auth Type:[/]  {dc.auth_type}")

    info_panel(f"Doctor: {name}", lines)

    action_choices: list[tuple[str, str]] = []
    if dc.type == "external" and dc.api_url:
        action_choices.append(("test",   "Test Doctor API        — send a probe request to verify the endpoint responds"))
    action_choices.append(("delete", "Delete Config          — remove this saved doctor config"))

    action = select_one("Doctor Actions", action_choices)
    if action == "test":
        _run_doctor_test(dc, client_ref)
    elif action == "delete":
        if ask_confirm(f"Delete doctor config '{name}'?", default=False):
            store.delete_doctor_config(name)
            success(f"Doctor config '{name}' deleted")


def _add_doctor(store: ConfigStore, client_ref: list | None = None) -> None:
    console.print("\n[bold]Add Doctor Config[/]")
    muted("Save a doctor API endpoint so you can reuse it across pipelines.\n")

    name = ask_text("Config name (e.g. 'my-gpt4-doctor', 'claude-api')")
    if not name:
        return

    doc_type = select_one("Doctor type", [
        ("external",       "Client's Doctor       — your own doctor API that EARL will call during evaluation"),
        ("internal",       "Lumos's Doctor        — EARL's built-in AI doctor (no setup needed)"),
        ("client_driven",  "Client-Driven         — you drive the conversation loop from your side (VPN/firewall)"),
    ], allow_back=False)
    if not doc_type:
        return

    api_url = ""
    api_key = ""
    auth_type = "bearer"

    if doc_type == "external":
        api_url = ask_text("Doctor API URL (e.g. https://my-doctor.example.com/chat)") or ""
        if not api_url:
            return
        api_key = ask_text("API Key (leave empty if none)", secret=True) or ""
        if api_key:
            auth_type_pick = select_one("Auth type", [
                ("bearer",  "Bearer token          — sent as 'Authorization: Bearer <key>' header"),
                ("api_key", "API Key header         — sent as 'X-API-Key: <key>' header"),
            ], allow_back=False)
            auth_type = auth_type_pick or "bearer"

    dc = DoctorConfig(
        name=name,
        type=doc_type,
        api_url=api_url,
        api_key=DoctorConfig.obfuscate(api_key) if api_key else "",
        auth_type=auth_type,
    )
    store.upsert_doctor_config(dc)
    success(f"Doctor config '{name}' saved")

    # Offer immediate test for Client's doctor
    if doc_type == "external" and api_url:
        if ask_confirm("Test this doctor API now?"):
            _run_doctor_test(dc, client_ref or [])


def _validate_doctor(store: ConfigStore, client_ref: list) -> None:
    """Pick a saved Client's doctor config and test it."""
    configs = [dc for dc in store.list_doctor_configs() if dc.type == "external"]
    if not configs:
        warn("No Client's doctor configs saved yet — add one first.")
        return

    choices = [(dc.name, f"{dc.name}  →  {dc.api_url}") for dc in configs]
    pick = select_one("Select doctor to validate", choices)
    if not pick:
        return

    dc = store.config.doctor_configs.get(pick)
    if not dc:
        return

    _run_doctor_test(dc, client_ref)


def _run_doctor_test(dc: DoctorConfig, client_ref: list) -> None:
    """Test a doctor API endpoint — tries SDK validation first, falls back to direct HTTP probe."""
    import urllib.request
    import urllib.error
    import time as _time

    api_url = dc.api_url
    api_key = dc.key_clear() if dc.api_key else ""

    console.print(f"\n  Testing [bold]{api_url}[/] ...")

    # 1) Try SDK-level validation if we have a connected client
    client = client_ref[0] if client_ref else None
    if client:
        try:
            result = client.pipelines.validate_doctor_api(
                api_url=api_url,
                api_key=api_key or None,
                timeout=15.0,
            )
            if result.get("valid"):
                success("Doctor API validated via EARL platform — endpoint is reachable and responds correctly")
                return
        except Exception as sdk_err:
            warn(f"SDK validation failed: {sdk_err}")
            muted("Falling back to direct HTTP probe...")

    # 2) Direct HTTP probe — works even without EARL credentials
    console.print("  [dim]Sending HTTP probe...[/]")
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "EARL-SDK-UI/1.0 (doctor-api-test)",
    }
    if api_key:
        if dc.auth_type == "api_key":
            headers["X-API-Key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    # Send a minimal test payload (matches EARL's doctor request format)
    import json
    test_payload = json.dumps({
        "conversation_history": [
            {"role": "patient", "content": "Hello, I have a headache."}
        ],
        "metadata": {"test": True, "source": "earl-sdk-ui"},
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=test_payload, headers=headers, method="POST")
    start = _time.monotonic()

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed_ms = int((_time.monotonic() - start) * 1000)
            status = resp.status
            body = resp.read().decode("utf-8")

            success(f"Doctor API responded — HTTP {status}")
            muted(f"Response time: {elapsed_ms}ms")

            # Try to parse and show a snippet of the response
            try:
                data = json.loads(body)
                # Look for common response fields
                if isinstance(data, dict):
                    response_text = (
                        data.get("response")
                        or data.get("message")
                        or data.get("content")
                        or data.get("reply")
                        or data.get("text")
                    )
                    if response_text and isinstance(response_text, str):
                        muted(f"Response preview: \"{response_text[:120]}{'...' if len(response_text) > 120 else ''}\"")
                    else:
                        muted(f"Response keys: {', '.join(list(data.keys())[:6])}")
            except json.JSONDecodeError:
                if body.strip():
                    muted(f"Response (non-JSON): {body[:100]}")

    except urllib.error.HTTPError as e:
        elapsed_ms = int((_time.monotonic() - start) * 1000)
        try:
            err_body = e.read().decode("utf-8")[:200]
        except Exception as read_err:
            _log.debug("Failed to read HTTP error body: %s", read_err, exc_info=True)
            err_body = ""

        if e.code in (401, 403):
            error(f"HTTP {e.code} — authentication failed ({elapsed_ms}ms)")
            warn("Check that your API key is correct and the auth type (Bearer vs X-API-Key) matches")
            if err_body:
                muted(f"  Server said: {err_body}")
        elif e.code == 404:
            error(f"HTTP 404 — endpoint not found ({elapsed_ms}ms)")
            warn("Verify the URL path is correct (e.g. /chat, /v1/chat, /api/respond)")
        elif e.code >= 500:
            warn(f"HTTP {e.code} — server error, but the endpoint IS reachable ({elapsed_ms}ms)")
            muted("The doctor API is running but returned an internal error on the test payload")
            if err_body:
                muted(f"  Server said: {err_body}")
        else:
            error(f"HTTP {e.code} ({elapsed_ms}ms)")
            if err_body:
                muted(f"  Server said: {err_body}")

    except urllib.error.URLError as e:
        error(f"Cannot reach {api_url}")
        reason = str(e.reason)
        if "Name or service not known" in reason or "nodename nor servname" in reason:
            warn("DNS resolution failed — check the hostname")
        elif "Connection refused" in reason:
            warn("Connection refused — the server may be down or the port is wrong")
        elif "timed out" in reason:
            warn("Connection timed out — the server didn't respond within 15 seconds")
        else:
            warn(f"Network error: {reason}")
    except Exception as e:
        error(f"Unexpected error: {e}")


# ── Preferences ───────────────────────────────────────────────────────────────


def _flow_preferences(store: ConfigStore) -> None:
    prefs = store.preferences

    info_panel("Current Preferences", [
        f"[bold]Default Pipeline:[/]   {prefs.default_pipeline or '(none)'}",
        f"[bold]Default Parallel:[/]   {prefs.default_parallel}",
        f"[bold]Auto-save Runs:[/]     {'yes' if prefs.auto_save_runs else 'no'}",
        f"[bold]Max Local Runs:[/]     {prefs.max_local_runs}",
    ])

    action = select_one("Edit Preferences", [
        ("pipeline", "Default Pipeline       — pipeline used when none is specified during 'Run Simulation'"),
        ("parallel", "Default Parallel Count — how many episodes to run simultaneously (1-10)"),
        ("autosave", "Auto-save Runs         — automatically save completed simulation results locally"),
        ("maxruns",  "Max Local Runs         — limit how many run results are kept on disk"),
    ])
    if action is None:
        return

    if action == "pipeline":
        val = ask_text("Default pipeline name", default=prefs.default_pipeline)
        if val is not None:
            prefs.default_pipeline = val
            store.save_preferences(prefs)
            success(f"Default pipeline → '{val}'")
    elif action == "parallel":
        val = ask_int("Default parallel count", default=prefs.default_parallel, min_val=1, max_val=10)
        if val:
            prefs.default_parallel = val
            store.save_preferences(prefs)
            success(f"Default parallel → {val}")
    elif action == "autosave":
        prefs.auto_save_runs = not prefs.auto_save_runs
        store.save_preferences(prefs)
        success(f"Auto-save runs → {'ON' if prefs.auto_save_runs else 'OFF'}")
    elif action == "maxruns":
        val = ask_int("Max local runs", default=prefs.max_local_runs, min_val=5, max_val=500)
        if val:
            prefs.max_local_runs = val
            store.save_preferences(prefs)
            success(f"Max local runs → {val}")


# ── Connection test ───────────────────────────────────────────────────────────


def _test_connection(store: ConfigStore, client_ref: list) -> None:
    client = client_ref[0] if client_ref else None
    if not client:
        error("No active profile — add one first under Auth Profiles.")
        return

    profile = store.get_active_profile()
    env = profile.environment if profile else "?"
    console.print(f"\n  Testing connection to [bold]{env}[/] ...")

    # Show connection details for debugging
    if client:
        from earl_sdk.client import EnvironmentConfig
        muted(f"API URL:   {EnvironmentConfig.get_api_url(env)}")
        muted(f"Auth0:     {EnvironmentConfig.get_auth0_domain(env)}")
        muted(f"Audience:  {EnvironmentConfig.get_auth0_audience(env)}")
        if profile:
            muted(f"Client ID: {profile.client_id[:8]}...{profile.client_id[-4:]}")
            muted(f"Org:       {profile.organization or '(none)'}")

    try:
        ok = client.test_connection()
        if ok:
            success("Connected to EARL API — credentials are valid")
        else:
            error("Connection test returned False — check credentials")
    except Exception as e:
        error(f"Connection failed: {e}")
        console.print()
        warn("Troubleshooting tips:")
        muted("  1. Verify Client ID and Client Secret are correct (from Auth0 dashboard)")
        muted("  2. Ensure the M2M app is authorized for the correct API audience")
        muted("  3. Check that the environment matches your credentials (dev/test/prod)")
        if profile and not profile.organization:
            muted("  4. Try adding an Organization ID if your Auth0 tenant requires it")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rebuild_client(store: ConfigStore, client_ref: list, quiet: bool = False) -> None:
    """Rebuild the EarlClient from the active profile.

    Note: EarlClient construction is lazy — it doesn't call Auth0 until the
    first API request.  So construction succeeding does NOT mean the
    credentials are valid; ``test_connection()`` must be called for that.
    """
    profile = store.get_active_profile()
    if not profile:
        client_ref.clear()
        return
    try:
        from earl_sdk import EarlClient
        client = EarlClient(
            client_id=profile.client_id,
            client_secret=profile.secret_clear(),
            organization=profile.organization or "",
            environment=profile.environment,
        )
        client_ref.clear()
        client_ref.append(client)
    except Exception as e:
        client_ref.clear()
        if not quiet:
            error(f"Failed to initialize client: {e}")


def build_client_from_store(store: ConfigStore) -> list:
    """Create initial client list from stored profile. Returns mutable [client] or []."""
    ref: list = []
    _rebuild_client(store, ref, quiet=True)
    return ref
