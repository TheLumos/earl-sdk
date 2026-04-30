"""Command-line interface for Earl SDK.

The ``earl`` command provides a scriptable CLI surface on top of existing SDK
APIs and local interactive storage primitives.
"""

# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional

from earl_sdk import DoctorApiConfig, EarlClient, SimulationStatus, __version__, auth_storage
from earl_sdk.client import _normalize_env
from earl_sdk.interactive.storage.config_store import AuthProfile, ConfigStore, DoctorConfig
from earl_sdk.interactive.storage.run_store import LocalRun, RunStore

from . import schema as _schema_mod

logger = logging.getLogger("earl_sdk.cli")

# Valid values for ``--env``. ``test`` is kept as a deprecated alias for
# ``staging`` so older profiles, scripts, and docs keep working; it is
# normalised to ``staging`` right after argparse parses the arguments.
_ENV_CHOICES = ("local", "dev", "staging", "test", "prod")


CLI_DESCRIPTION = """\
EARL command-line interface.

Designed to be equally usable by humans and LLM-driven agents:

- Every subcommand has rich ``--help`` examples.
- ``earl schema [--format json|markdown]`` emits the entire command tree in a
  machine-readable form so agents can discover every flag at once.
- ``--json`` (alias of ``--output json``) produces stable JSON on stdout.
- ``--dry-run`` on mutating commands prints the payload and exits without
  contacting the API.
- ``--debug`` enables structured request logging on stderr (secrets redacted).

Authentication is resolved from (in order):
  1. Service-account env vars — ``EARL_CLIENT_ID`` + ``EARL_CLIENT_SECRET``
     (+ ``EARL_ORG_ID``). This is the CI / automation path.
  2. Active / named profile in ``~/.earl/config.json`` (populated by
     ``earl login``). This is the interactive human path.
  3. Per-command ``--client-id`` / ``--client-secret`` / ``--organization``
     flags, for scripts that pre-date the env-var model.

``EARL_ORGANIZATION`` is accepted as a deprecated alias for ``EARL_ORG_ID``.
``EARL_ENVIRONMENT`` still selects the target deployment.
"""

CLI_EPILOG = """\
Common workflows:

  # One-off discovery (agent-friendly):
  earl schema --format json | jq '.commands | keys'

  # Run a pipeline end-to-end:
  earl auth profile add --name prod --client-id ... --env prod   # prompts for secret
  earl pipelines list --json
  earl simulations start --pipeline my-eval --num-episodes 5 --json
  earl simulations wait <id>

  # CI / scripted use:
  echo "$CLIENT_SECRET" | earl auth profile add --name ci --client-id ... --env staging --client-secret -

Exit codes:
  0 = success, 1 = unhandled error, 2 = argparse/usage error.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earl",
        description=CLI_DESCRIPTION,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", help="Use saved auth profile name from ~/.earl/config.json")
    parser.add_argument(
        "--env",
        choices=_ENV_CHOICES,
        help="Environment override (``test`` is a deprecated alias for ``staging``)",
    )
    parser.add_argument("--client-id", help="Auth0 client ID")
    parser.add_argument(
        "--client-secret",
        help=(
            "Auth0 client secret. Avoid on the command line (shell history / `ps` leaks); "
            "prefer env var EARL_CLIENT_SECRET or a saved profile."
        ),
    )
    parser.add_argument("--organization", help="Auth0 organization ID")
    parser.add_argument("--api-url", help="Global API base URL override")
    parser.add_argument("--cases-api-url")
    parser.add_argument("--dimensions-api-url")
    parser.add_argument("--patients-api-url")
    parser.add_argument("--pipelines-api-url")
    parser.add_argument("--simulations-api-url")
    parser.add_argument("--verifiers-api-url")
    parser.add_argument("--auth0-domain")
    parser.add_argument("--auth0-audience")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format for list/detail commands (default: table)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Shortcut for --output json. Recommended for scripts and LLM agents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "For mutating commands (create/update/delete/start/stop), print the "
            "resolved payload on stdout and exit without calling the API."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Emit structured request/response logs on stderr (secrets redacted).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential success messages (errors still printed to stderr).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks on error.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # schema: self-describing CLI for LLMs / docs generators
    schema_p = sub.add_parser(
        "schema",
        help="Emit a machine-readable description of the entire CLI",
        description=(
            "Dump the full command tree (every subcommand, every flag) as JSON or "
            "Markdown. Agents can run this once to learn the surface without "
            "scraping --help output."
        ),
        epilog=(
            "Examples:\n"
            "  earl schema                       # pretty-printed JSON on stdout\n"
            "  earl schema --format markdown     # docs-site friendly\n"
            "  earl schema | jq '.commands.pipelines.subcommands.create.arguments'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    schema_p.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Output format"
    )
    schema_p.add_argument(
        "--out", help="Write to file instead of stdout (useful for docs generation)"
    )

    # auth
    auth = sub.add_parser("auth", help="Manage auth profiles and connectivity")
    auth_sub = auth.add_subparsers(dest="auth_cmd", required=True)
    auth_prof = auth_sub.add_parser("profile", help="Manage profiles")
    auth_prof_sub = auth_prof.add_subparsers(dest="profile_cmd", required=True)

    prof_add = auth_prof_sub.add_parser(
        "add",
        help="Add a profile",
        description=(
            "Save a named Auth0 M2M profile. The client secret is stored in the "
            "OS keyring when available (macOS Keychain / Windows Credential Vault "
            "/ Linux Secret Service) and falls back to a base64-obfuscated entry "
            "in ~/.earl/config.json otherwise."
        ),
        epilog=(
            "Examples:\n"
            "  earl auth profile add --name prod --client-id abc --organization org_x --env prod\n"
            "      (prompts for secret via getpass)\n"
            "  earl auth profile add --name ci --client-id abc --env staging --client-secret -\n"
            "      (reads secret from stdin; safe for CI pipelines)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    prof_add.add_argument("--name", required=True)
    prof_add.add_argument("--client-id", required=True)
    prof_add.add_argument(
        "--client-secret",
        help=(
            "Auth0 client secret. Omit to be prompted securely (getpass). "
            "Pass '-' to read from stdin. Avoid passing on the command line "
            "(leaks to shell history and `ps`)."
        ),
    )
    prof_add.add_argument("--organization", default="")
    prof_add.add_argument("--env", required=True, choices=_ENV_CHOICES)

    auth_prof_sub.add_parser("list", help="List profiles")

    prof_use = auth_prof_sub.add_parser("use", help="Set active profile")
    prof_use.add_argument("name")

    prof_show = auth_prof_sub.add_parser("show", help="Show profile details")
    prof_show.add_argument("name")

    prof_del = auth_prof_sub.add_parser("delete", help="Delete profile")
    prof_del.add_argument("name")

    auth_sub.add_parser("test", help="Test connectivity with current credentials")

    # ── auth login / logout (OAuth Device Authorization Grant) ───────────────
    auth_login = auth_sub.add_parser(
        "login",
        help="Sign in via the browser using the OAuth Device Authorization Grant",
        description=(
            "Starts the OAuth 2.0 Device Authorization Grant (RFC 8628) against "
            "the EARL environment's Auth0 tenant. The CLI prints a user code, "
            "opens the Auth0 activation URL in your default browser, and waits "
            "for you to approve. On success, a long-lived refresh token is "
            "stored in the OS keyring (or a base64-obfuscated entry in "
            "~/.earl/config.json) and a new auth profile is created.\n\n"
            "Multi-org support: if you omit --organization, the CLI runs the "
            "device flow without an org, asks the orchestrator which orgs "
            "you belong to, prompts you to pick one (or auto-selects if there "
            "is only one), and then exchanges your refresh token for an "
            "org-scoped access token WITHOUT a second browser visit. Pass "
            "--organization to skip the picker and pre-select an org.\n\n"
            "Requires a public 'Native' Auth0 application per env with the "
            "Device Code + Refresh Token grants enabled. Register its client "
            "ID with `earl auth device-clients set --env <env> <client_id>` "
            "or pass --device-client-id each run."
        ),
        epilog=(
            "Examples:\n"
            "  earl auth login --env staging                     # interactive picker\n"
            "  earl auth login --env prod --organization org_abc123\n"
            "  earl auth login --env dev --device-client-id abc123 --no-browser\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    auth_login.add_argument("--env", required=True, choices=_ENV_CHOICES)
    auth_login.add_argument(
        "--name",
        help="Profile name to save. Default: 'device-<env>' (overwritten if it exists).",
    )
    auth_login.add_argument(
        "--device-client-id",
        help=(
            "Public Auth0 Native-app client ID to use for this env. Overrides "
            "the value stored in config.json. Native clients are public by "
            "design, so this flag does not leak secrets."
        ),
    )
    auth_login.add_argument(
        "--organization",
        default=None,
        help="Auth0 organization ID to embed in the login (optional).",
    )
    auth_login.add_argument(
        "--scopes",
        help=(
            "Space-separated OAuth scopes. Default: 'openid profile email "
            "offline_access'. offline_access is required for refresh tokens."
        ),
    )
    auth_login.add_argument(
        "--audience",
        help="Override the API audience (defaults to the env's EARL API audience).",
    )
    auth_login.add_argument(
        "--domain",
        help="Override the Auth0 tenant domain (defaults to the env's Auth0 tenant).",
    )
    auth_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open a browser; just print the verification URL.",
    )
    auth_login.add_argument(
        "--activate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set the newly created profile as the active profile (default: yes).",
    )

    auth_logout = auth_sub.add_parser(
        "logout",
        help="Remove a device-flow profile and clear its cached tokens",
        description=(
            "Deletes the saved refresh token from the OS keyring, clears the "
            "on-disk access-token cache, and removes the profile from "
            "config.json. Does not revoke the Auth0 session in the browser."
        ),
    )
    auth_logout.add_argument(
        "--name",
        help="Profile name to log out of. Default: active profile.",
    )

    auth_devclients = auth_sub.add_parser(
        "device-clients",
        help="View or set per-env public Device-Flow client IDs",
        description=(
            "Public Auth0 Native-app client IDs used by `earl auth login`. "
            "These are not secrets; they identify a CLI application in each "
            "Auth0 tenant. One ID per env: local/dev/staging/prod."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    devclients_sub = auth_devclients.add_subparsers(dest="devclients_cmd", required=True)
    devclients_sub.add_parser("list", help="List configured device-flow client IDs")
    dc_set = devclients_sub.add_parser("set", help="Register a client ID for an env")
    dc_set.add_argument("--env", required=True, choices=_ENV_CHOICES)
    dc_set.add_argument("client_id")
    dc_clear = devclients_sub.add_parser("clear", help="Remove the client ID for an env")
    dc_clear.add_argument("--env", required=True, choices=_ENV_CHOICES)

    auth_myorgs = auth_sub.add_parser(
        "my-orgs",
        help="List the Auth0 organizations the current user is a member of",
        description=(
            "Calls the orchestrator's read-only /api/v1/auth/my-orgs endpoint "
            "using the active profile's access token. Useful to discover an "
            "org_id before running `earl auth login --organization ...`.\n\n"
            "Works with any valid user token — including a no-org token from "
            "an incomplete `earl auth login` — because the endpoint is "
            "specifically scoped to allow tokens without an org_id claim."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    auth_myorgs.add_argument(
        "--env",
        choices=_ENV_CHOICES,
        help=(
            "Environment to query. Defaults to the active profile's env. "
            "Required when there is no active profile."
        ),
    )
    auth_myorgs.add_argument(
        "--profile",
        help="Named profile to use. Defaults to the active profile.",
    )

    auth_sub.add_parser(
        "migrate-secrets",
        help="Move base64-obfuscated secrets from ~/.earl/config.json into the OS keyring",
    )

    auth_sub.add_parser(
        "backend",
        help="Show which secret-storage backend will be used for new writes",
        description=(
            "Reports whether new profile/doctor secrets will go into the OS "
            "keyring or a base64-obfuscated entry in ~/.earl/config.json. "
            "Use the global --json flag for machine-readable output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # doctor
    doctor = sub.add_parser(
        "doctor",
        help="Manage saved doctor endpoint configs",
        description=(
            "Named configurations for doctor endpoints that pipelines can reuse. "
            "Three types are supported: 'internal' (EARL's built-in AI doctor), "
            "'external' (HTTP API you host), and 'client_driven' (you poll and "
            "submit responses yourself, useful behind VPN/firewall)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_sub = doctor.add_subparsers(dest="doctor_cmd", required=True)
    doctor_add = doctor_sub.add_parser(
        "add",
        help="Add doctor config",
        description=(
            "Save a named doctor endpoint config. API keys follow the same "
            "keyring-first storage rules as auth profiles."
        ),
        epilog=(
            "Examples:\n"
            "  earl doctor add --name gpt4 --type external --api-url https://api.example.com/chat\n"
            "      (prompts for --api-key)\n"
            "  earl doctor add --name internal --type internal\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_add.add_argument("--name", required=True)
    doctor_add.add_argument(
        "--type", required=True, choices=["internal", "external", "client_driven"]
    )
    doctor_add.add_argument("--api-url")
    doctor_add.add_argument(
        "--api-key",
        help=(
            "Bearer token or API key for external doctor. Omit to be prompted "
            "securely; pass '-' to read from stdin. Ignored for non-external types."
        ),
    )
    doctor_add.add_argument("--auth-type", choices=["bearer", "api_key"], default="bearer")
    doctor_sub.add_parser("list", help="List doctor configs")
    doctor_show = doctor_sub.add_parser("show", help="Show doctor config")
    doctor_show.add_argument("name")
    doctor_del = doctor_sub.add_parser("delete", help="Delete doctor config")
    doctor_del.add_argument("name")
    doctor_validate = doctor_sub.add_parser("validate", help="Validate external doctor config")
    doctor_validate.add_argument("name")
    doctor_validate.add_argument("--timeout", type=float, default=10.0)

    # simple catalog commands
    cases = sub.add_parser("cases", help="Case catalog")
    cases_sub = cases.add_subparsers(dest="cases_cmd", required=True)
    cases_sub.add_parser("list", help="List cases")
    case_get = cases_sub.add_parser("get", help="Get case details")
    case_get.add_argument("case_id")

    patients = sub.add_parser("patients", help="Patients catalog")
    patients_sub = patients.add_subparsers(dest="patients_cmd", required=True)
    p_list = patients_sub.add_parser("list", help="List patients")
    p_list.add_argument("--difficulty")
    p_list.add_argument("--tags", help="Comma-separated tags")
    p_list.add_argument("--limit", type=int, default=100)
    p_list.add_argument("--offset", type=int, default=0)
    p_get = patients_sub.add_parser("get", help="Get patient details")
    p_get.add_argument("patient_id")

    dims = sub.add_parser("dimensions", help="Dimensions API")
    dims_sub = dims.add_subparsers(dest="dimensions_cmd", required=True)
    d_list = dims_sub.add_parser("list", help="List dimensions")
    d_list.add_argument("--include-custom", action=argparse.BooleanOptionalAction, default=True)
    d_get = dims_sub.add_parser("get", help="Get dimension")
    d_get.add_argument("dimension_id")
    d_create = dims_sub.add_parser("create", help="Create custom dimension")
    d_create.add_argument("--name", required=True)
    d_create.add_argument("--description", required=True)
    d_create.add_argument("--category", default="custom")
    d_create.add_argument("--weight", type=float, default=1.0)

    verifiers = sub.add_parser("verifiers", help="List generic verifier catalog")
    verifiers_sub = verifiers.add_subparsers(dest="verifiers_cmd", required=True)
    v_list = verifiers_sub.add_parser("list", help="List verifiers")
    v_list.add_argument(
        "--type", choices=["all", "hard-gates", "scoring-dimensions"], default="all"
    )

    # pipelines
    pipelines = sub.add_parser(
        "pipelines",
        help="Manage pipelines (evaluation configurations)",
        description=(
            "A pipeline couples a case (or custom patient+verifier list) with a "
            "doctor endpoint configuration. Pipelines are reusable across "
            "simulation runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipelines_sub = pipelines.add_subparsers(dest="pipelines_cmd", required=True)
    pl_list = pipelines_sub.add_parser("list", help="List pipelines")
    pl_list.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True)
    pl_get = pipelines_sub.add_parser("get", help="Get pipeline")
    pl_get.add_argument("pipeline_name")
    pl_create = pipelines_sub.add_parser(
        "create",
        help="Create pipeline",
        description="Create a new pipeline. Use --dry-run to preview the payload.",
        epilog=(
            "Examples:\n"
            "  earl pipelines create --name my-eval --case-id carla-hypertension-yasmin \\\n"
            "      --doctor internal --json\n"
            "  earl pipelines create --name vpn --case-id ... --doctor client_driven --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_pipeline_common_args(pl_create, creating=True)
    pl_update = pipelines_sub.add_parser("update", help="Update pipeline")
    pl_update.add_argument("pipeline_name")
    _add_pipeline_common_args(pl_update, creating=False)
    pl_delete = pipelines_sub.add_parser("delete", help="Delete pipeline")
    pl_delete.add_argument("pipeline_name")
    pl_validate = pipelines_sub.add_parser(
        "validate-doctor", help="Validate external doctor endpoint"
    )
    pl_validate.add_argument("--api-url", required=True)
    pl_validate.add_argument("--api-key")
    pl_validate.add_argument("--timeout", type=float, default=10.0)

    # simulations
    sims = sub.add_parser(
        "simulations",
        help="Manage simulation runs against pipelines",
        description=(
            "A simulation runs a pipeline against N episodes. Typical workflow:\n"
            "  1. `earl simulations start --pipeline <name> --num-episodes 5`\n"
            "  2. `earl simulations wait <id>` (or poll with `get`)\n"
            "  3. `earl simulations report <id>` for the final scorecard"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sims_sub = sims.add_subparsers(dest="simulations_cmd", required=True)
    s_list = sims_sub.add_parser("list", help="List simulations")
    s_list.add_argument(
        "--status", choices=[s.value for s in SimulationStatus], help="Status filter"
    )
    s_list.add_argument("--limit", type=int, default=50)
    s_list.add_argument("--skip", type=int, default=0)
    s_get = sims_sub.add_parser("get", help="Get simulation")
    s_get.add_argument("simulation_id")
    s_start = sims_sub.add_parser("start", help="Start simulation")
    s_start.add_argument("--pipeline", required=True)
    s_start.add_argument("--num-episodes", type=int)
    s_start.add_argument("--parallel-count", type=int, default=1)
    s_wait = sims_sub.add_parser("wait", help="Wait for simulation completion")
    s_wait.add_argument("simulation_id")
    s_wait.add_argument("--poll-interval", type=float, default=5.0)
    s_wait.add_argument("--timeout", type=float)
    sims_sub.add_parser(
        "pending", help="List pending episodes (requires --simulation-id)"
    ).add_argument("--simulation-id", required=True)
    s_stop = sims_sub.add_parser("stop", help="Stop simulation")
    s_stop.add_argument("simulation_id")
    s_episodes = sims_sub.add_parser("episodes", help="List episodes")
    s_episodes.add_argument("simulation_id")
    s_episodes.add_argument(
        "--include-dialogue", action=argparse.BooleanOptionalAction, default=False
    )
    s_episode = sims_sub.add_parser("episode", help="Get one episode")
    s_episode.add_argument("simulation_id")
    s_episode.add_argument("episode_id")
    s_report = sims_sub.add_parser("report", help="Get report")
    s_report.add_argument("simulation_id")
    s_report.add_argument("--save", help="Write JSON report to file")
    s_respond = sims_sub.add_parser(
        "respond", help="Submit doctor response for client-driven episode"
    )
    s_respond.add_argument("simulation_id")
    s_respond.add_argument("episode_id")
    s_respond.add_argument("--message")
    s_respond.add_argument("--message-file")

    # runs
    runs = sub.add_parser("runs", help="Local run storage commands")
    runs_sub = runs.add_subparsers(dest="runs_cmd", required=True)
    r_list = runs_sub.add_parser("list", help="List local runs")
    r_list.add_argument("--limit", type=int, default=50)
    r_show = runs_sub.add_parser("show", help="Show local run report")
    r_show.add_argument("simulation_id")
    r_save = runs_sub.add_parser("save", help="Download and save remote simulation report locally")
    r_save.add_argument("simulation_id")
    r_delete = runs_sub.add_parser("delete", help="Delete local run")
    r_delete.add_argument("simulation_id")
    r_compare = runs_sub.add_parser("compare", help="Compare local runs")
    r_compare.add_argument("simulation_ids", nargs="+")
    r_prune = runs_sub.add_parser("prune", help="Prune old local runs")
    r_prune.add_argument("--max-runs", type=int, default=50)

    # preferences/config
    config = sub.add_parser("config", help="Configuration helpers")
    config_sub = config.add_subparsers(dest="config_cmd", required=True)
    config_sub.add_parser("show", help="Show active config")
    config_sub.add_parser("path", help="Show config file path")

    prefs = sub.add_parser("prefs", help="Preferences")
    prefs_sub = prefs.add_subparsers(dest="prefs_cmd", required=True)
    prefs_sub.add_parser("show", help="Show preferences")
    prefs_set = prefs_sub.add_parser("set", help="Set preferences")
    prefs_set.add_argument("--default-pipeline")
    prefs_set.add_argument("--default-parallel", type=int)
    prefs_set.add_argument("--auto-save-runs", choices=["true", "false"])
    prefs_set.add_argument("--max-local-runs", type=int)

    # utility
    sub.add_parser("whoami", help="Show resolved connection metadata")
    rl = sub.add_parser("rate-limits", help="Show rate limits")
    rl.add_argument("--category", default="default")

    # chat command: delegates to existing interactive flow implementation
    chat = sub.add_parser("chat", help="Chat flows")
    chat_sub = chat.add_subparsers(dest="chat_cmd", required=True)
    chat_sub.add_parser("start", help="Start interactive chat (existing UI flow)")

    # ── top-level login / logout (PKCE default; device fallback) ─────────────
    top_login = sub.add_parser(
        "login",
        help="Sign in interactively via the browser (PKCE + loopback)",
        description=(
            "Opens the default browser against Auth0 Universal Login and "
            "completes OAuth 2.0 Authorization Code + PKCE over a loopback "
            "redirect (RFC 7636 + RFC 8252). On success, a long-lived "
            "refresh token is saved in the OS keyring (or a base64-obfuscated "
            "entry in ~/.earl/config.json).\n\n"
            "Use --headless to fall back to the OAuth Device Authorization "
            "Grant — useful on SSH sessions or machines without a GUI browser."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top_login.add_argument("--env", required=True, choices=_ENV_CHOICES)
    top_login.add_argument("--name", help="Profile name to save. Default: derived from env+org.")
    top_login.add_argument(
        "--headless",
        action="store_true",
        help="Use the OAuth Device Authorization Grant instead of PKCE.",
    )
    top_login.add_argument(
        "--client-id",
        dest="device_client_id",
        help="Override the public Auth0 Native-app client ID for this env.",
    )
    top_login.add_argument("--organization", default=None, help="Preselect an Auth0 organization.")
    top_login.add_argument("--scopes", help="Space-separated OAuth scopes.")
    top_login.add_argument("--audience", help="Override the Earl API audience.")
    top_login.add_argument("--domain", help="Override the Auth0 tenant domain.")
    top_login.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically; just print the URL.",
    )
    top_login.add_argument(
        "--activate", action=argparse.BooleanOptionalAction, default=True,
        help="Set the newly created profile as active (default: yes).",
    )

    top_logout = sub.add_parser(
        "logout",
        help="Sign out of a browser-issued profile (PKCE or device)",
        description=(
            "Deletes the saved refresh token from the OS keyring, clears the "
            "on-disk access-token cache, and removes the profile from "
            "config.json. Does not revoke the Auth0 session in the browser."
        ),
    )
    top_logout.add_argument("--name", help="Profile name to log out of. Default: active profile.")

    # ── top-level service-account management ─────────────────────────────────
    sa = sub.add_parser(
        "service-account",
        help="Provision, list, and revoke M2M service-account credentials",
        description=(
            "Service accounts are Auth0 M2M applications provisioned on demand "
            "by the orchestrator. They issue ``grant_type=client_credentials`` "
            "access tokens scoped to a single organization (same shape as a "
            "human ``earl login`` token — both carry ``org_id``). Create one "
            "per CI system; rotate with ``revoke`` + ``create``. Credentials "
            "are shown exactly once."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sa_sub = sa.add_subparsers(dest="sa_cmd", required=True)

    sa_create = sa_sub.add_parser(
        "create",
        help="Provision a new service account and print its credentials (once).",
    )
    sa_create.add_argument("--name", required=True)
    sa_create.add_argument(
        "--scopes",
        default="earl:read",
        help=(
            "Space- or comma-separated list of Earl API scopes. "
            "Default: earl:read."
        ),
    )
    sa_create.add_argument("--description", default="")
    sa_create.add_argument(
        "--org-id",
        help="Target organization (EARL_Admin only; defaults to caller's own org).",
    )

    sa_list = sa_sub.add_parser("list", help="List this org's service accounts.")
    sa_list.add_argument(
        "--org-id",
        help="Target organization (EARL_Admin only; defaults to caller's own org).",
    )

    sa_revoke = sa_sub.add_parser(
        "revoke", help="Delete a service account by its client_id."
    )
    # Use a distinct dest to avoid colliding with the top-level ``--client-id``
    # auth override: argparse would otherwise overwrite ``args.client_id`` and
    # trick ``_build_client`` into using the revocation target as an M2M creds
    # override, triggering a refresh loop against wrong credentials.
    sa_revoke.add_argument(
        "sa_client_id",
        metavar="client_id",
        help="Auth0 client_id of the service account to revoke.",
    )
    sa_revoke.add_argument(
        "--yes",
        action="store_true",
        help="Skip the 'are you sure?' prompt. Required in non-TTY contexts.",
    )

    # ── top-level organization administration ────────────────────────────────
    org = sub.add_parser(
        "org",
        help="Manage Auth0 organizations, members, and invitations",
        description=(
            "Organization administration. All subcommands resolve to endpoints "
            "under ``/api/v1/organizations`` on the orchestrator.\n"
            "\n"
            "Role matrix:\n"
            "  * ``EARL_Admin`` (global) can create/list/delete orgs, and "
            "    grant/revoke the ``EARL_Org_Admin`` role on members.\n"
            "  * ``EARL_Org_Admin`` can read/update/invite/remove members\n"
            "    inside their own organization.\n"
            "\n"
            "Typical flows:\n"
            "  earl org list                        # global admin\n"
            "  earl org create --name acme --display-name 'Acme Corp'\n"
            "  earl org invite ORG --email a@b.com --role EARL_Org_Admin\n"
            "  earl org members list ORG\n"
            "  earl org roles grant ORG auth0|abc EARL_Org_Admin\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    org_sub = org.add_subparsers(dest="org_cmd", required=True)

    # list
    org_list = org_sub.add_parser(
        "list",
        help="List every Auth0 organization on the tenant (EARL_Admin only).",
    )
    org_list.add_argument(
        "--format", choices=("table", "json"), default="table", dest="org_format",
    )

    # create
    org_create = org_sub.add_parser(
        "create",
        help="Create a new Auth0 organization (EARL_Admin only).",
    )
    org_create.add_argument(
        "--name",
        required=True,
        help=(
            "Slug (lowercase letters, digits, hyphens, underscores; 3-50 chars)."
        ),
    )
    org_create.add_argument(
        "--display-name", default="", help="Human label, e.g. 'Acme Corp'."
    )
    org_create.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Custom metadata key/value pair. Repeatable (max 10, 255 chars).",
    )

    # show
    org_show = org_sub.add_parser(
        "show",
        help=(
            "Show one organization. Defaults to the caller's own org. "
            "EARL_Admin may pass any ORG_ID."
        ),
    )
    org_show.add_argument("org_id", nargs="?", help="Target organization id.")

    # update
    org_update = org_sub.add_parser(
        "update",
        help="Update an organization's display name and/or metadata.",
    )
    org_update.add_argument("org_id")
    org_update.add_argument("--display-name")
    org_update.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Replace metadata; repeat for multiple pairs. Omit to leave as-is.",
    )

    # delete
    org_delete = org_sub.add_parser(
        "delete",
        help="Delete an organization (EARL_Admin only).",
    )
    org_delete.add_argument("org_id")
    org_delete.add_argument("--yes", action="store_true")

    # members
    org_members = org_sub.add_parser(
        "members",
        help="List/remove members of an organization.",
    )
    org_members_sub = org_members.add_subparsers(dest="org_members_cmd", required=True)
    om_list = org_members_sub.add_parser(
        "list",
        help="List members of an organization (+their roles).",
    )
    om_list.add_argument("org_id", nargs="?")
    om_list.add_argument(
        "--format", choices=("table", "json"), default="table", dest="org_format",
    )
    om_remove = org_members_sub.add_parser(
        "remove",
        help="Remove a member from an organization.",
    )
    om_remove.add_argument("org_id")
    om_remove.add_argument("user_id")
    om_remove.add_argument("--yes", action="store_true")

    # invite
    org_invite = org_sub.add_parser(
        "invite",
        help="Invite a user to an organization by email.",
    )
    org_invite.add_argument("org_id")
    org_invite.add_argument("--email", required=True)
    org_invite.add_argument(
        "--role",
        action="append",
        default=[],
        help=(
            "Role to grant on invite accept. Repeatable. "
            "E.g. ``--role EARL_Org_Admin``."
        ),
    )
    org_invite.add_argument(
        "--ttl-seconds",
        type=int,
        help="Override invitation TTL (Auth0 default is 7 days).",
    )

    # invitations list/revoke
    org_invs = org_sub.add_parser(
        "invitations",
        help="List/revoke pending invitations for an organization.",
    )
    org_invs_sub = org_invs.add_subparsers(dest="org_invs_cmd", required=True)
    oi_list = org_invs_sub.add_parser("list")
    oi_list.add_argument("org_id", nargs="?")
    oi_list.add_argument(
        "--format", choices=("table", "json"), default="table", dest="org_format",
    )
    oi_revoke = org_invs_sub.add_parser("revoke")
    oi_revoke.add_argument("org_id")
    oi_revoke.add_argument("invitation_id")
    oi_revoke.add_argument("--yes", action="store_true")

    # roles grant/revoke
    org_roles = org_sub.add_parser(
        "roles",
        help="Grant/revoke organization roles on a member (EARL_Admin only).",
    )
    org_roles_sub = org_roles.add_subparsers(dest="org_roles_cmd", required=True)
    orl_grant = org_roles_sub.add_parser(
        "grant",
        help="Grant roles to a member, e.g. ``EARL_Org_Admin``.",
    )
    orl_grant.add_argument("org_id")
    orl_grant.add_argument("user_id")
    orl_grant.add_argument("roles", nargs="+")
    orl_revoke = org_roles_sub.add_parser(
        "revoke",
        help="Revoke roles from a member.",
    )
    orl_revoke.add_argument("org_id")
    orl_revoke.add_argument("user_id")
    orl_revoke.add_argument("roles", nargs="+")
    orl_list = org_roles_sub.add_parser(
        "catalog",
        help="List assignable tenant-level roles (EARL_Admin only).",
    )
    orl_list.add_argument(
        "--format", choices=("table", "json"), default="table", dest="org_format",
    )

    return parser


def _add_pipeline_common_args(parser: argparse.ArgumentParser, *, creating: bool) -> None:
    if creating:
        parser.add_argument("--name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--case-id")
    parser.add_argument("--verifier-id", action="append", default=[])
    parser.add_argument("--verifier-ids", help="Comma-separated verifier IDs")
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument("--patient-ids", help="Comma-separated patient IDs")
    parser.add_argument("--doctor", choices=["internal", "external", "client_driven"])
    parser.add_argument("--doctor-api-url")
    parser.add_argument("--doctor-api-key")
    parser.add_argument("--doctor-auth-type", choices=["bearer", "api_key"], default="bearer")
    parser.add_argument("--validate-doctor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conversation-initiator", choices=["patient", "doctor"])
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--verifiers", choices=["lumos", "legacy"])


def _configure_logging(debug: bool) -> None:
    """Configure the ``earl_sdk`` logger for ``--debug`` output.

    Logs go to stderr so that stdout stays a clean JSON/table stream for
    scripts and LLM agents.
    """
    level = logging.DEBUG if debug else logging.WARNING
    root = logging.getLogger("earl_sdk")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[earl] %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


def main(argv: list[str] | None = None) -> None:
    parser = _parser()

    # Shell completion (only active when `argcomplete` is installed and the
    # shell has sourced `register-python-argcomplete earl`). No-op otherwise.
    try:
        import argcomplete  # type: ignore

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args(argv)

    if getattr(args, "env", None) == "test":
        print(
            "warning: --env test is deprecated; use --env staging instead "
            "(test is kept as a silent alias for backwards compatibility).",
            file=sys.stderr,
        )
        args.env = "staging"

    if getattr(args, "json_output", False):
        args.output = "json"

    _configure_logging(bool(getattr(args, "debug", False)))

    if args.command == "schema":
        _handle_schema(args, parser)
        return

    store = ConfigStore()
    runs = RunStore()

    if not args.command:
        parser.print_help()
        return

    try:
        _dispatch(args, store, runs)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        from ._errors import render_error

        code = render_error(exc, args)
        raise SystemExit(code) from exc


def _handle_schema(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Emit the CLI schema to stdout (or ``--out``)."""
    if args.format == "json":
        body = _schema_mod.schema_json(parser, version=__version__)
    else:
        body = _schema_mod.schema_markdown(parser, version=__version__)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(body if body.endswith("\n") else body + "\n")
        if not getattr(args, "quiet", False):
            print(f"wrote schema ({args.format}) to {args.out}", file=sys.stderr)
        return
    print(body)


def _dry_run_emit(args: argparse.Namespace, action: str, payload: dict) -> bool:
    """If ``--dry-run`` is set, emit the payload and signal the caller to skip.

    Returns ``True`` when the caller should return immediately.
    """
    if not getattr(args, "dry_run", False):
        return False
    envelope = {"dry_run": True, "action": action, "payload": payload}
    print(json.dumps(envelope, indent=2, default=str))
    return True


def _say(args: argparse.Namespace, message: str) -> None:
    """Print a success/info message unless ``--quiet`` is set."""
    if getattr(args, "quiet", False):
        return
    print(message)


def _dry_run_no_client(args: argparse.Namespace) -> bool:
    """Handle ``--dry-run`` for API-bound mutating commands without a client.

    Returns ``True`` when the dry-run payload was emitted and the dispatcher
    should return immediately. Returns ``False`` for read-only commands (they
    are never dry-run) and for unrecognised shapes (fall through to normal
    handling).
    """
    cmd = args.command
    if cmd == "dimensions" and getattr(args, "dimensions_cmd", None) == "create":
        return _dry_run_emit(
            args,
            "dimensions.create",
            {
                "name": args.name,
                "description": args.description,
                "category": args.category,
                "weight": args.weight,
            },
        )
    if cmd == "pipelines":
        sub_cmd = getattr(args, "pipelines_cmd", None)
        if sub_cmd == "delete":
            return _dry_run_emit(args, "pipelines.delete", {"pipeline_name": args.pipeline_name})
        if sub_cmd in {"create", "update"}:
            kwargs = _pipeline_kwargs_from_args(args)
            payload: dict[str, Any] = (
                {"name": args.name}
                if sub_cmd == "create"
                else {"pipeline_name": args.pipeline_name}
            )
            for k, v in kwargs.items():
                payload[k] = v.to_dict() if hasattr(v, "to_dict") else v
            if isinstance(payload.get("doctor_config"), dict) and payload["doctor_config"].get(
                "api_key"
            ):
                payload["doctor_config"]["api_key"] = "<redacted>"
            return _dry_run_emit(args, f"pipelines.{sub_cmd}", payload)
    if cmd == "simulations":
        sub_cmd = getattr(args, "simulations_cmd", None)
        if sub_cmd == "start":
            return _dry_run_emit(
                args,
                "simulations.start",
                {
                    "pipeline_name": args.pipeline,
                    "num_episodes": args.num_episodes,
                    "parallel_count": args.parallel_count,
                },
            )
        if sub_cmd == "stop":
            return _dry_run_emit(args, "simulations.stop", {"simulation_id": args.simulation_id})
        if sub_cmd == "respond":
            message = args.message
            if args.message_file:
                try:
                    with open(args.message_file, encoding="utf-8") as f:
                        message = f.read()
                except OSError as e:
                    raise ValueError(f"Cannot read --message-file: {e}") from e
            if not message:
                raise ValueError("Provide --message or --message-file")
            return _dry_run_emit(
                args,
                "simulations.respond",
                {
                    "simulation_id": args.simulation_id,
                    "episode_id": args.episode_id,
                    "message_length": len(message),
                },
            )
    return False


def _dispatch(args: argparse.Namespace, store: ConfigStore, runs: RunStore) -> None:
    if args.command == "auth":
        _handle_auth(args, store)
        return
    if args.command == "login":
        # Top-level `earl login` is the canonical interactive entry point.
        # It reuses `_handle_auth_login` but flips the default flow to PKCE
        # (with device flow as the `--headless` fallback).
        _handle_top_login(args, store)
        return
    if args.command == "logout":
        _handle_top_logout(args, store)
        return
    if args.command == "doctor":
        _handle_doctor(args, store)
        return
    if args.command == "config":
        _handle_config(args, store)
        return
    if args.command == "prefs":
        _handle_prefs(args, store)
        return
    if args.command == "runs":
        _handle_runs(args, runs, store)
        return
    if args.command == "service-account":
        _handle_service_account(args, store)
        return
    if args.command == "org":
        _handle_org(args, store)
        return

    # --dry-run for API-bound mutating commands: emit the resolved payload
    # without building an authenticated client. Keeps CI pipelines and LLM
    # agents from needing valid credentials just to preview a call.
    if getattr(args, "dry_run", False) and _dry_run_no_client(args):
        return

    client = _build_client(args, store)

    if args.command == "cases":
        _handle_cases(args, client)
    elif args.command == "patients":
        _handle_patients(args, client)
    elif args.command == "dimensions":
        _handle_dimensions(args, client)
    elif args.command == "verifiers":
        _handle_verifiers(args, client)
    elif args.command == "pipelines":
        _handle_pipelines(args, client)
    elif args.command == "simulations":
        _handle_simulations(args, client)
    elif args.command == "whoami":
        _emit(args, _build_whoami_payload(client))
    elif args.command == "rate-limits":
        data = client.rate_limits.get()
        if args.category:
            data["selected_category"] = args.category
            data["selected_category_limit"] = client.rate_limits.get_effective_limit(args.category)
        _emit(args, data)
    elif args.command == "chat":
        _handle_chat(args, client, store, runs)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def _build_whoami_payload(client: EarlClient) -> dict[str, Any]:
    """Resolve the richest identity picture we can produce for ``earl whoami``.

    Always includes environment + org + URLs. Also decodes the current access
    token (refreshing silently if the cached one is expired) to surface the
    user's email, Auth0 ``sub``, the org's display name and the roles the
    backend will see. Falls back to just the connection metadata if anything
    goes wrong — ``whoami`` should never error.
    """
    payload: dict[str, Any] = {
        "environment": client.environment,
        "organization_id": client.organization,
        "organization": client.organization,  # kept for backward-compat
        "api_url": client.api_url,
        "service_api_urls": client.service_api_urls,
    }

    try:
        token = client._auth.get_token()  # noqa: SLF001 — CLI is in-tree
    except Exception:  # noqa: BLE001 — whoami stays local-friendly
        return payload

    try:
        from earl_sdk.device_flow import decode_jwt_payload

        claims = decode_jwt_payload(token) or {}
    except Exception:  # noqa: BLE001
        return payload

    def _first_str(*keys: str) -> str:
        for k in keys:
            v = claims.get(k)
            if isinstance(v, str) and v:
                return v
        return ""

    roles: list[str] = []
    for key in (
        "https://earl/roles",
        "https://earl.thelumos.ai/roles",
        "https://earl.thelumos.xyz/roles",
        "https://earl-api.thelumos.xyz/roles",
        "https://api.earl.thelumos.ai/roles",
    ):
        v = claims.get(key)
        if isinstance(v, list):
            roles = [str(r) for r in v]
            break

    identity = {
        "email": _first_str("email", "https://earl/email"),
        "subject": _first_str("sub"),
        "organization_name": _first_str(
            "org_name",
            "https://earl/org_name",
            "https://api.earl.thelumos.ai/organization_name",
            "https://earl.thelumos.ai/organization_name",
            "https://earl.thelumos.xyz/organization_name",
        ),
        "roles": roles,
        "scope": _first_str("scope"),
        "token_expires_at": claims.get("exp"),
        "auth_kind": "m2m"
        if _first_str("sub").endswith("@clients")
        else "user",
    }
    payload.update({k: v for k, v in identity.items() if v not in (None, "", [])})
    return payload


def _build_client(args: argparse.Namespace, store: ConfigStore) -> EarlClient:
    """Resolve credentials into an :class:`EarlClient`.

    Resolution order (first match wins):

    1. ``EARL_CLIENT_ID`` + ``EARL_CLIENT_SECRET`` environment variables
       → client_credentials (M2M). ``EARL_ORG_ID`` is required so the
       resulting token carries an ``org_id`` claim; the backend derives
       tenancy from that claim alone.
    2. Active / selected profile:
       - PKCE or Device profile → reuse its cached access/refresh token.
       - M2M profile → reuse its client_id/client_secret.
    3. Per-call ``--client-id`` / ``--client-secret`` overrides (still
       supported for scripts that aren't yet on env vars).

    If none of the above produces credentials, the caller gets a clear
    error pointing at ``earl login`` for humans and ``EARL_CLIENT_*`` for
    automation.
    """
    # ── Step 1: env-var M2M takes precedence over everything else. ──────────
    env_cid = os.getenv("EARL_CLIENT_ID", "").strip()
    env_csec = os.getenv("EARL_CLIENT_SECRET", "").strip()
    # ``EARL_ORG_ID`` is canonical; ``EARL_ORGANIZATION`` is a deprecated
    # alias retained so existing CI pipelines don't break on upgrade. Emit a
    # one-time deprecation warning to stderr if only the old name is present.
    raw_org_id = os.getenv("EARL_ORG_ID", "").strip()
    raw_org_legacy = os.getenv("EARL_ORGANIZATION", "").strip()
    if raw_org_legacy and not raw_org_id:
        if not getattr(_build_client, "_org_deprecation_warned", False):
            print(
                "[earl] EARL_ORGANIZATION is deprecated — please migrate to "
                "EARL_ORG_ID (same value, clearer name). The old name still "
                "works but will be removed in a future release.",
                file=sys.stderr,
            )
            setattr(_build_client, "_org_deprecation_warned", True)
    env_org = (raw_org_id or raw_org_legacy).strip()
    explicit_cid = (getattr(args, "client_id", "") or "").strip()
    explicit_csec = (getattr(args, "client_secret", "") or "").strip()

    # If the user passed BOTH env vars (or their --flag equivalents), skip
    # the profile lookup entirely — this is the "CI picked me up" path.
    env_m2m_present = (env_cid and env_csec) or (explicit_cid and explicit_csec)

    profile = None
    if not env_m2m_present:
        if getattr(args, "profile", None):
            profile = store.config.profiles.get(args.profile)
            if not profile:
                raise ValueError(f"Profile not found: {args.profile}")
        elif store.config.active_profile:
            profile = store.get_active_profile()

    auth_kind = "m2m"
    refresh_token: str | None = None

    if env_m2m_present:
        client_id = explicit_cid or env_cid
        client_secret = explicit_csec or env_csec
    elif profile and profile.auth_kind in ("pkce", "device"):
        auth_kind = profile.auth_kind
        client_id = explicit_cid or profile.client_id
        client_secret = ""  # public clients — no secret
        refresh_token = profile.refresh_token_clear()
        if not refresh_token:
            raise ValueError(
                f"Profile '{profile.name}' is a {profile.auth_kind}-flow profile but its "
                "refresh token is missing from storage. Run "
                f"`earl login --env {profile.environment}` to re-authenticate."
            )
    else:
        client_id = explicit_cid or (
            profile.client_id if profile else env_cid
        )
        client_secret = explicit_csec or (
            profile.secret_clear() if profile else env_csec
        )

    organization = getattr(args, "organization", None)
    if organization is None:
        if env_m2m_present:
            organization = env_org
        elif profile:
            organization = profile.organization or ""
        else:
            organization = env_org

    if env_m2m_present and not organization:
        raise ValueError(
            "EARL_CLIENT_ID/EARL_CLIENT_SECRET are set but EARL_ORG_ID is "
            "missing. Every token must carry an org_id claim — set "
            "EARL_ORG_ID=<org_...> alongside the client credentials, or pass "
            "--organization on the command line."
        )

    environment = args.env or (
        profile.environment if profile else os.getenv("EARL_ENVIRONMENT", "prod")
    )
    if auth_kind == "m2m" and (not client_id or not client_secret):
        raise ValueError(
            "Missing credentials. Either:\n"
            "  - Run `earl login --env <env>` for interactive login, OR\n"
            "  - Set EARL_CLIENT_ID / EARL_CLIENT_SECRET / EARL_ORG_ID for "
            "CI/automation (see `earl service-account create`)."
        )
    if auth_kind in ("pkce", "device") and not client_id:
        raise ValueError(
            "Browser-flow profile is missing a client_id; re-run `earl login`."
        )

    service_urls = {}
    if args.cases_api_url:
        service_urls["cases"] = args.cases_api_url
    if args.dimensions_api_url:
        service_urls["dimensions"] = args.dimensions_api_url
    if args.patients_api_url:
        service_urls["patients"] = args.patients_api_url
    if args.pipelines_api_url:
        service_urls["pipelines"] = args.pipelines_api_url
    if args.simulations_api_url:
        service_urls["simulations"] = args.simulations_api_url
    if args.verifiers_api_url:
        service_urls["verifiers"] = args.verifiers_api_url

    return EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization or "",
        environment=environment,
        api_url=args.api_url,
        service_api_urls=service_urls or None,
        auth0_domain=args.auth0_domain,
        auth0_audience=args.auth0_audience,
        request_timeout=args.request_timeout,
        auth_kind=auth_kind,
        refresh_token=refresh_token,
    )


def _resolve_secret(value: str | None, *, prompt: str) -> str:
    """Resolve a CLI secret value without leaking it to shell history.

    Rules:
    - ``None`` → interactive ``getpass`` prompt (fails if stdin is non-TTY).
    - ``"-"`` → read one line from stdin (suitable for CI pipelines).
    - Any other value → used as-is, with a stderr warning.
    """
    if value is None:
        if not sys.stdin.isatty():
            raise ValueError(
                "Secret required but stdin is not a TTY; either pass --client-secret/--api-key '-' "
                "to read from stdin, or run this command interactively."
            )
        return getpass.getpass(prompt)
    if value == "-":
        data = sys.stdin.readline().rstrip("\n")
        if not data:
            raise ValueError("No secret received on stdin")
        return data
    print(
        "warning: passing secrets on the command line leaks them to shell history and `ps`; "
        "prefer omitting the flag or passing '-' to read from stdin.",
        file=sys.stderr,
    )
    return value


def _handle_auth(args: argparse.Namespace, store: ConfigStore) -> None:
    if args.auth_cmd == "profile":
        if args.profile_cmd == "add":
            if _dry_run_emit(
                args,
                "auth.profile.add",
                {
                    "name": args.name,
                    "client_id": args.client_id,
                    "organization": args.organization,
                    "environment": args.env,
                    "client_secret": "<redacted>",
                },
            ):
                return
            secret = _resolve_secret(args.client_secret, prompt="Auth0 client secret: ")
            prof = AuthProfile.create(
                name=args.name,
                client_id=args.client_id,
                clear_secret=secret,
                organization=args.organization,
                environment=args.env,
            )
            store.upsert_profile(prof)
            backend_note = (
                "OS keyring" if prof.secret_backend == "keyring" else "config.json (base64)"
            )
            _say(args, f"Saved profile '{args.name}' (secret stored in {backend_note})")
            return
        if args.profile_cmd == "list":
            cfg = store.config
            rows = []
            for name, prof in cfg.profiles.items():
                rows.append(
                    {
                        "name": name,
                        "environment": prof.environment,
                        "organization": prof.organization,
                        "active": name == cfg.active_profile,
                    }
                )
            _print_rows(rows, headers=["name", "environment", "organization", "active"])
            return
        if args.profile_cmd == "use":
            if _dry_run_emit(args, "auth.profile.use", {"name": args.name}):
                return
            store.set_active_profile(args.name)
            _say(args, f"Active profile: {args.name}")
            return
        if args.profile_cmd == "show":
            p = store.config.profiles.get(args.name)
            if not p:
                raise ValueError(f"Profile not found: {args.name}")
            print(
                json.dumps(
                    {
                        "name": p.name,
                        "client_id": p.client_id,
                        "client_secret_tail": f"...{p.secret_clear()[-4:]}",
                        "organization": p.organization,
                        "environment": p.environment,
                    },
                    indent=2,
                )
            )
            return
        if args.profile_cmd == "delete":
            if _dry_run_emit(args, "auth.profile.delete", {"name": args.name}):
                return
            store.delete_profile(args.name)
            _say(args, f"Deleted profile: {args.name}")
            return
    if args.auth_cmd == "test":
        client = _build_client(args, store)
        ok = client.test_connection()
        print("Connection OK" if ok else "Connection failed")
        return
    if args.auth_cmd == "login":
        _handle_auth_login(args, store)
        return
    if args.auth_cmd == "logout":
        _handle_auth_logout(args, store)
        return
    if args.auth_cmd == "my-orgs":
        _handle_auth_my_orgs(args, store)
        return
    if args.auth_cmd == "device-clients":
        _handle_device_clients(args, store)
        return
    if args.auth_cmd == "migrate-secrets":
        if _dry_run_emit(args, "auth.migrate-secrets", {}):
            return
        summary = store.migrate_secrets_to_keyring()
        if args.output == "json":
            print(json.dumps(summary))
            return
        if summary.get("skipped") == -1:
            _say(
                args,
                "Keyring backend unavailable — nothing to migrate. "
                "Install the 'secure' extra: pip install 'earl-sdk[secure]'",
            )
            return
        _say(
            args,
            f"Migrated {summary['profiles_migrated']} profile secret(s) and "
            f"{summary['doctors_migrated']} doctor key(s) to the OS keyring.",
        )
        return
    if args.auth_cmd == "backend":
        backend = auth_storage.backend_name()
        if args.output == "json":
            print(json.dumps({"backend": backend}))
            return
        if backend == "keyring":
            _say(args, "secret backend: OS keyring (secure)")
        else:
            _say(
                args,
                "secret backend: config.json (base64 only — not encrypted). "
                "Install 'earl-sdk[secure]' for keyring support.",
            )
        return
    raise ValueError("Unsupported auth command")


# ── earl auth login / logout / device-clients ───────────────────────────────


def _handle_device_clients(args: argparse.Namespace, store: ConfigStore) -> None:
    if args.devclients_cmd == "list":
        ids = dict(store.config.device_client_ids)
        if args.output == "json":
            print(json.dumps(ids, indent=2))
            return
        if not ids:
            _say(
                args,
                "No device-flow client IDs configured. Register one with\n"
                "  earl auth device-clients set --env <env> <client_id>",
            )
            return
        rows = [{"environment": env, "client_id": cid} for env, cid in sorted(ids.items())]
        _print_rows(rows, headers=["environment", "client_id"])
        return
    if args.devclients_cmd == "set":
        if _dry_run_emit(
            args,
            "auth.device-clients.set",
            {"env": args.env, "client_id": args.client_id},
        ):
            return
        store.set_device_client_id(args.env, args.client_id)
        _say(args, f"Saved device client_id for env={args.env}")
        return
    if args.devclients_cmd == "clear":
        if _dry_run_emit(args, "auth.device-clients.clear", {"env": args.env}):
            return
        store.clear_device_client_id(args.env)
        _say(args, f"Cleared device client_id for env={args.env}")
        return
    raise ValueError("Unsupported device-clients command")


def _resolve_device_client_id(args: argparse.Namespace, store: ConfigStore) -> str:
    """Pick the Device-Flow client ID from (flag → config.json → env var)."""
    if args.device_client_id:
        return args.device_client_id
    stored = store.get_device_client_id(args.env)
    if stored:
        return stored
    env_var = os.getenv(f"EARL_DEVICE_CLIENT_ID_{args.env.upper()}", "")
    if env_var:
        return env_var
    raise ValueError(
        f"No Device-Flow client_id configured for env={args.env}.\n"
        f"Register one with:\n"
        f"  earl auth device-clients set --env {args.env} <client_id>\n"
        f"or pass --device-client-id on this command. See AGENTS.md / sdk/README.md "
        f"for one-time Auth0 Native-app setup instructions."
    )


def _pick_organization_interactive(
    orgs: list[dict],
    *,
    current_org_id: str = "",
    stream=None,
) -> dict:
    """Prompt an interactive user to choose one of ``orgs``.

    Orgs are expected to be pre-sorted (the orchestrator already does this).
    Accepts the choice number, the org id, or the org short name (case
    insensitive). Default — pressing Enter — picks the first entry (or the
    one whose ``id`` matches ``current_org_id`` if any).

    Raises :class:`SystemExit` if the user cancels (Ctrl-C / Ctrl-D) or
    enters an invalid response three times in a row.
    """
    if stream is None:
        stream = sys.stderr
    if not orgs:
        raise SystemExit(
            "No organizations are available for this user on this tenant. "
            "Ask an admin to add you to an Auth0 organization, or re-run "
            "`earl auth login --organization <org_id>` with an explicit ID."
        )

    # Figure out the default choice (1-indexed).
    default_idx = 1
    for i, o in enumerate(orgs, start=1):
        if current_org_id and o.get("id") == current_org_id:
            default_idx = i
            break
    default_org = orgs[default_idx - 1]

    print("", file=stream)
    print("Select an organization to sign in as:", file=stream)
    for i, o in enumerate(orgs, start=1):
        label = o.get("display_name") or o.get("name") or o.get("id")
        marker = " (current)" if current_org_id and o.get("id") == current_org_id else ""
        print(f"  [{i}] {label}  —  {o.get('id')}{marker}", file=stream)
    print("", file=stream)

    for attempt in range(3):
        try:
            prompt = (
                f"Choose a number, id, or name "
                f"[default: {default_idx} = {default_org.get('id')}]: "
            )
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=stream)
            raise SystemExit("Login cancelled.")
        if not raw:
            return default_org

        # Number?
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(orgs):
                return orgs[n - 1]
        # Exact id / name match (case-insensitive)?
        lowered = raw.lower()
        for o in orgs:
            if (
                o.get("id", "").lower() == lowered
                or o.get("name", "").lower() == lowered
                or o.get("display_name", "").lower() == lowered
            ):
                return o
        print(f"  '{raw}' is not a valid choice. Try again.", file=stream)

    raise SystemExit("No valid organization chosen after 3 attempts. Aborting.")


def _discover_user_orgs(
    *,
    env: str,
    api_url: str,
    access_token: str,
    no_browser: bool,
) -> list[dict]:
    """Call the orchestrator's ``/auth/my-orgs`` using a raw bearer token.

    We cannot use :class:`EarlClient` here because the bearer belongs to a
    no-org token, and the client's ``Auth0Client`` expects a profile with
    an organization. We go direct with the shared httpx transport instead,
    which also means we get the full structured-error treatment for free.

    Returns a list of ``{id, name, display_name}`` dicts. Returns an empty
    list on a 503 "discovery disabled" response from the orchestrator so
    the caller can surface a helpful fallback message.
    """
    from earl_sdk import _http
    from earl_sdk.exceptions import EarlError

    url = api_url.rstrip("/") + "/auth/my-orgs"
    try:
        body, _resp = _http.request_json(
            "GET",
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20.0,
        )
    except EarlError:
        raise

    if not isinstance(body, dict):
        return []
    items = body.get("organizations") or body.get("items") or []
    if not isinstance(items, list):
        return []
    return [o for o in items if isinstance(o, dict) and o.get("id")]


def _handle_auth_login(args: argparse.Namespace, store: ConfigStore) -> None:
    """Run the OAuth Device Authorization Grant and persist the result.

    Multi-org login flow:

    1. Device authorization without ``organization`` → access token with no
       ``org_id`` claim, plus a long-lived refresh token.
    2. ``GET /api/v1/auth/my-orgs`` on the orchestrator → list of orgs the
       user belongs to.
    3. Pick one (auto if exactly one; interactive prompt otherwise; error
       out if 0 or if non-TTY with multiple choices).
    4. Exchange the refresh token for an org-scoped access token using
       ``grant_type=refresh_token&organization=<picked>`` — no second
       browser trip required.

    ``--organization`` bypasses steps 2-3 and passes the org straight to
    Auth0 at step 1 (useful for scripted/admin setups).
    """
    from earl_sdk.client import EnvironmentConfig
    from earl_sdk.device_flow import (
        DEFAULT_SCOPES,
        extract_org_info,
        poll_for_token,
        refresh_access_token_with_organization,
        slugify_org,
        start_device_authorization,
    )
    from earl_sdk.exceptions import EarlError

    client_id = _resolve_device_client_id(args, store)
    domain = (
        args.domain
        or os.getenv("EARL_AUTH0_DOMAIN")
        or EnvironmentConfig.get_auth0_domain(args.env)
    )
    audience = (
        args.audience
        or os.getenv("EARL_AUTH0_AUDIENCE")
        or EnvironmentConfig.get_auth0_audience(args.env)
    )
    api_url = EnvironmentConfig.get_api_url(args.env)
    scopes = args.scopes.split() if args.scopes else list(DEFAULT_SCOPES)
    if "offline_access" not in scopes:
        # Without offline_access Auth0 will not issue a refresh token, which
        # defeats the whole point of persisting the login.
        scopes.append("offline_access")

    pre_selected_org = (args.organization or "").strip()

    if _dry_run_emit(
        args,
        "auth.login",
        {
            "env": args.env,
            "profile_name_hint": args.name,
            "client_id": client_id,
            "domain": domain,
            "audience": audience,
            "pre_selected_organization": pre_selected_org,
            "scopes": scopes,
            "will_use_picker": not bool(pre_selected_org),
        },
    ):
        return

    # 1. Request a user/device code — with org if the user pre-selected one,
    # without org otherwise (enables the org-picker flow).
    device = start_device_authorization(
        domain=domain,
        client_id=client_id,
        audience=audience,
        scopes=scopes,
        organization=pre_selected_org or None,
    )

    # 2. Prompt the user.
    print("", file=sys.stderr)
    print(f"  Visit: {device.verification_uri_complete}", file=sys.stderr)
    print(f"  Code:  {device.user_code}", file=sys.stderr)
    print(
        f"  (expires in ~{device.expires_in // 60} minutes)",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    if not args.no_browser:
        try:
            import webbrowser

            webbrowser.open(device.verification_uri_complete)
        except Exception:  # noqa: BLE001 - best-effort; fall through to manual copy/paste
            pass

    # 3. Poll.
    last_remaining = [device.expires_in]

    def _pending(remaining: float) -> None:
        if int(remaining) // 30 < int(last_remaining[0]) // 30:
            print(
                f"  waiting for browser approval (~{int(remaining)}s remaining)…",
                file=sys.stderr,
            )
        last_remaining[0] = remaining

    token = poll_for_token(
        domain=domain,
        client_id=client_id,
        device_code=device.device_code,
        interval=device.interval,
        expires_in=device.expires_in,
        on_pending=_pending,
    )

    if not token.refresh_token:
        raise SystemExit(
            "Auth0 did not issue a refresh token. Check that the API's "
            "'Allow Offline Access' is ON and that the Native app has the "
            "Refresh Token grant enabled."
        )

    # 4. Decide which org this session is for.
    org_info = extract_org_info(token.access_token)
    chosen_org_id = pre_selected_org or org_info.org_id
    chosen_org_name = ""
    if org_info.org_id == chosen_org_id:
        chosen_org_name = org_info.org_name

    # Auth0 device flow silently ignores the ``organization`` parameter on
    # ``/oauth/device/code`` when the Native app has ``organization_usage: allow``
    # (it only enforces org binding when set to ``require``). That means even
    # if the user passed --organization, the access token we just got back may
    # still be missing the ``org_id`` claim. Detect that and re-mint an
    # org-scoped token from the refresh token below.
    needs_org_rebind = bool(chosen_org_id) and org_info.org_id != chosen_org_id

    if not chosen_org_id:
        # No org baked into the token → consult the orchestrator for the
        # user's memberships. The no-org token is used EXCLUSIVELY for this
        # single read-only call — after we pick an org we mint a fresh
        # org-scoped token via refresh and never touch the no-org one again.
        print("Looking up your organization memberships…", file=sys.stderr)
        try:
            orgs = _discover_user_orgs(
                env=args.env,
                api_url=api_url,
                access_token=token.access_token,
                no_browser=args.no_browser,
            )
        except EarlError as exc:
            # Typical reasons: 503 (org discovery disabled on this deployment),
            # 502 (Auth0 Management unreachable), 401 (orchestrator rejected
            # our no-org token). In all cases the right next move is to fall
            # back to --organization so the user isn't stuck.
            raise SystemExit(
                f"Could not discover your organizations ({exc}). "
                f"Re-run `earl auth login --env {args.env} --organization <org_id>`."
            )

        if not orgs:
            raise SystemExit(
                "You are not a member of any Auth0 organization on the "
                f"{args.env} tenant. Ask an admin to add you to an org, "
                "then re-run `earl auth login`."
            )

        # Auto-select when there's exactly one — zero friction for the
        # common case (customers typically have one org per env).
        if len(orgs) == 1:
            chosen = orgs[0]
            print(
                f"Found one organization: {chosen.get('display_name') or chosen.get('name')} "
                f"({chosen['id']}). Using it.",
                file=sys.stderr,
            )
        else:
            if not sys.stdin.isatty():
                # Can't prompt in a pipe / CI. Tell the user *which* orgs they
                # have so they can re-run non-interactively.
                orgs_list = "\n".join(
                    f"  - {o['id']}  ({o.get('display_name') or o.get('name') or ''})"
                    for o in orgs
                )
                raise SystemExit(
                    "Multiple organizations found but stdin is not a TTY so "
                    "the picker cannot run. Available orgs:\n"
                    f"{orgs_list}\n"
                    f"Re-run `earl auth login --env {args.env} --organization <org_id>`."
                )
            chosen = _pick_organization_interactive(orgs)

        chosen_org_id = chosen["id"]
        chosen_org_name = chosen.get("display_name") or chosen.get("name") or ""

        # 5. Swap the no-org refresh token for an org-scoped access token.
        print(
            f"Signing in to {chosen_org_name or chosen_org_id}…",
            file=sys.stderr,
        )
        try:
            token = refresh_access_token_with_organization(
                domain=domain,
                client_id=client_id,
                refresh_token=token.refresh_token,
                organization=chosen_org_id,
                audience=audience,
                scopes=scopes,
            )
        except EarlError as exc:
            raise SystemExit(
                f"Auth0 refused to issue an org-scoped token for "
                f"{chosen_org_id}: {exc}"
            )

        # Re-extract in case the org-scoped token carries richer metadata
        # (e.g. email + org_name claims not present before).
        org_info = extract_org_info(token.access_token)
        if org_info.org_name:
            chosen_org_name = org_info.org_name
    elif needs_org_rebind:
        # User pre-selected an org via --organization, but Auth0's device-flow
        # /oauth/device/code did NOT bind it into the access token. Re-mint
        # from the refresh token so the stored access token has an ``org_id``
        # claim and the backend accepts it.
        print(
            f"Binding session to {chosen_org_id}…",
            file=sys.stderr,
        )
        try:
            token = refresh_access_token_with_organization(
                domain=domain,
                client_id=client_id,
                refresh_token=token.refresh_token,
                organization=chosen_org_id,
                audience=audience,
                scopes=scopes,
            )
        except EarlError as exc:
            raise SystemExit(
                f"Auth0 refused to issue an org-scoped token for "
                f"{chosen_org_id}: {exc}"
            )
        org_info = extract_org_info(token.access_token)
        if org_info.org_name:
            chosen_org_name = org_info.org_name

    # 6. Persist. Profile name includes the org slug so multiple-org users
    # end up with distinct profiles they can switch between.
    if args.name:
        profile_name = args.name
    else:
        slug = slugify_org(chosen_org_name, chosen_org_id)
        profile_name = f"device-{args.env}-{slug}" if slug else f"device-{args.env}"

    prof = AuthProfile.create_device(
        name=profile_name,
        client_id=client_id,
        clear_refresh_token=token.refresh_token,
        organization=chosen_org_id,
        environment=args.env,
    )
    store.upsert_profile(prof)
    if args.activate:
        store.set_active_profile(prof.name)

    # Seed the on-disk access-token cache so the next `earl` call skips Auth0.
    cache_key = auth_storage.token_cache_key(
        client_id, audience, chosen_org_id, domain, "device"
    )
    auth_storage.save_token(
        cache_key,
        auth_storage.CachedToken(
            access_token=token.access_token,
            token_type=token.token_type,
            expires_at=token.expires_at,
            scope=token.scope or None,
            refresh_token=token.refresh_token,
        ),
    )

    backend_note = "OS keyring" if prof.secret_backend == "keyring" else "config.json (base64)"
    who = f" ({org_info.email})" if org_info.email else ""
    org_label = chosen_org_name or chosen_org_id or "no organization"
    _say(
        args,
        f"Signed in to env={args.env}{who} as profile '{prof.name}' "
        f"→ {org_label}. Refresh token stored in {backend_note}.",
    )


def _handle_auth_my_orgs(args: argparse.Namespace, store: ConfigStore) -> None:
    """Implement ``earl auth my-orgs`` — a diagnostic of the picker source."""
    from earl_sdk.client import EnvironmentConfig
    from earl_sdk.exceptions import EarlError

    # Resolve (profile, env) with the same precedence as other auth commands:
    # explicit --profile > explicit --env (no profile, impossible unless a
    # profile for that env exists) > active profile.
    profile_name = (
        getattr(args, "profile", None)
        or store.config.active_profile
    )
    prof: Optional[AuthProfile] = None
    if profile_name:
        prof = store.config.profiles.get(profile_name)
    if prof is None and not args.env:
        raise SystemExit(
            "No active profile. Pass --env <dev|staging|prod> or --profile <name>."
        )

    env = args.env or (prof.environment if prof else None)
    if not env:
        raise SystemExit("Could not determine environment.")

    api_url = EnvironmentConfig.get_api_url(env)

    if _dry_run_emit(args, "auth.my-orgs", {"env": env, "api_url": api_url}):
        return

    # Build an Auth0Client and fetch a valid access token for whichever
    # profile we resolved. For no-org device profiles this will cheerfully
    # hand us back the no-org token — which is exactly the shape /auth/my-orgs
    # accepts. For org-scoped profiles we just use their regular token.
    if prof is not None:
        client = _build_client(args, store)
        access_token = client._auth.get_headers()["Authorization"].split(" ", 1)[1]
    else:
        raise SystemExit(
            "Fetching organizations requires a profile (or a prior "
            "`earl auth login`). None available."
        )

    try:
        orgs = _discover_user_orgs(
            env=env,
            api_url=api_url,
            access_token=access_token,
            no_browser=True,
        )
    except EarlError as exc:
        raise SystemExit(f"Could not list organizations: {exc}")

    if getattr(args, "json_output", False) or getattr(args, "output", "table") == "json":
        print(json.dumps({"environment": env, "organizations": orgs}, indent=2))
        return

    if not orgs:
        _say(args, f"No organizations found for the current user in env={env}.")
        return

    rows = [
        {
            "id": o.get("id", ""),
            "name": o.get("name", ""),
            "display_name": o.get("display_name", ""),
        }
        for o in orgs
    ]
    _print_rows(rows, headers=["id", "name", "display_name"])


def _handle_auth_logout(args: argparse.Namespace, store: ConfigStore) -> None:
    target = args.name or store.config.active_profile
    if not target:
        raise ValueError("No profile to log out of (no --name and no active profile).")
    prof = store.config.profiles.get(target)
    if prof is None:
        raise ValueError(f"Profile not found: {target}")
    if _dry_run_emit(args, "auth.logout", {"name": target, "auth_kind": prof.auth_kind}):
        return

    # Purge token cache.
    try:
        from earl_sdk.client import EnvironmentConfig

        domain = EnvironmentConfig.get_auth0_domain(prof.environment)
        audience = EnvironmentConfig.get_auth0_audience(prof.environment)
        cache_key = auth_storage.token_cache_key(
            prof.client_id, audience, prof.organization or "", domain, prof.auth_kind
        )
        auth_storage.clear_token(cache_key)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.debug("token cache clear failed: %s", exc)

    store.delete_profile(target)
    _say(args, f"Logged out of profile '{target}'.")


# ── Top-level login / logout / service-account ──────────────────────────────


def _handle_top_login(args: argparse.Namespace, store: ConfigStore) -> None:
    """Interactive login entry point (`earl login`).

    Delegates to :func:`_handle_auth_login` after normalising the argument
    shape — and flips the default from Device Flow to Authorization Code +
    PKCE. ``--headless`` restores the Device-Flow behaviour for environments
    without a browser.
    """
    if getattr(args, "headless", False):
        logger.debug("earl login --headless → device authorization grant")
        _handle_auth_login(args, store)
        return
    _handle_auth_login_pkce(args, store)


def _handle_top_logout(args: argparse.Namespace, store: ConfigStore) -> None:
    _handle_auth_logout(args, store)


def _handle_auth_login_pkce(args: argparse.Namespace, store: ConfigStore) -> None:
    """Run the Authorization Code + PKCE loopback login and persist the result."""
    from earl_sdk.client import EnvironmentConfig
    from earl_sdk.exceptions import EarlError
    from earl_sdk.pkce_flow import (
        DEFAULT_SCOPES,
        PKCEFlowError,
        run_loopback_login,
    )
    from earl_sdk.device_flow import (
        extract_org_info,
        refresh_access_token_with_organization,
        slugify_org,
    )

    client_id = _resolve_device_client_id(args, store)
    domain = (
        getattr(args, "domain", None)
        or os.getenv("EARL_AUTH0_DOMAIN")
        or EnvironmentConfig.get_auth0_domain(args.env)
    )
    audience = (
        getattr(args, "audience", None)
        or os.getenv("EARL_AUTH0_AUDIENCE")
        or EnvironmentConfig.get_auth0_audience(args.env)
    )
    api_url = EnvironmentConfig.get_api_url(args.env)
    scopes_str = getattr(args, "scopes", None)
    scopes = scopes_str.split() if scopes_str else list(DEFAULT_SCOPES)
    if "offline_access" not in scopes:
        # offline_access is required to get a refresh token; without it the
        # CLI would re-prompt on every run, which defeats the whole purpose.
        scopes.append("offline_access")

    pre_selected_org = (getattr(args, "organization", None) or "").strip()

    if _dry_run_emit(
        args,
        "auth.login.pkce",
        {
            "env": args.env,
            "profile_name_hint": getattr(args, "name", None),
            "client_id": client_id,
            "domain": domain,
            "audience": audience,
            "pre_selected_organization": pre_selected_org,
            "scopes": scopes,
        },
    ):
        return

    print("Opening your browser to complete login…", file=sys.stderr)

    def _manual(url: str) -> None:
        print("", file=sys.stderr)
        print("Could not open a browser automatically.", file=sys.stderr)
        print(f"Paste this URL into any browser to continue:\n  {url}", file=sys.stderr)

    try:
        token = run_loopback_login(
            domain=domain,
            client_id=client_id,
            audience=audience,
            scopes=scopes,
            organization=pre_selected_org or None,
            print_instructions=_manual,
        )
    except (PKCEFlowError, EarlError) as exc:
        raise SystemExit(
            f"PKCE login failed ({exc}). "
            f"Retry, or run with --headless to use the Device Authorization Grant."
        )

    if not token.refresh_token:
        raise SystemExit(
            "Auth0 did not issue a refresh token. Confirm the API's 'Allow "
            "Offline Access' is ON and the Native app has the Refresh Token "
            "grant enabled."
        )

    org_info = extract_org_info(token.access_token)
    chosen_org_id = pre_selected_org or org_info.org_id
    chosen_org_name = org_info.org_name or ""
    needs_org_rebind = bool(chosen_org_id) and org_info.org_id != chosen_org_id

    if not chosen_org_id:
        print("Looking up your organization memberships…", file=sys.stderr)
        try:
            orgs = _discover_user_orgs(
                env=args.env,
                api_url=api_url,
                access_token=token.access_token,
                no_browser=getattr(args, "no_browser", False),
            )
        except EarlError as exc:
            raise SystemExit(
                f"Could not discover your organizations ({exc}). "
                f"Re-run `earl login --env {args.env} --organization <org_id>`."
            )

        if not orgs:
            raise SystemExit(
                "You are not a member of any Auth0 organization on the "
                f"{args.env} tenant. Ask an admin to add you to an org, "
                "then re-run `earl login`."
            )

        if len(orgs) == 1:
            chosen = orgs[0]
            print(
                f"Found one organization: {chosen.get('display_name') or chosen.get('name')} "
                f"({chosen['id']}). Using it.",
                file=sys.stderr,
            )
        else:
            if not sys.stdin.isatty():
                orgs_list = "\n".join(
                    f"  - {o['id']}  ({o.get('display_name') or o.get('name') or ''})"
                    for o in orgs
                )
                raise SystemExit(
                    "Multiple organizations found but stdin is not a TTY so "
                    "the picker cannot run. Available orgs:\n"
                    f"{orgs_list}\n"
                    f"Re-run `earl login --env {args.env} --organization <org_id>`."
                )
            chosen = _pick_organization_interactive(orgs)

        chosen_org_id = chosen["id"]
        chosen_org_name = chosen.get("display_name") or chosen.get("name") or ""
        needs_org_rebind = True

    if needs_org_rebind:
        print(
            f"Signing in to {chosen_org_name or chosen_org_id}…",
            file=sys.stderr,
        )
        try:
            token = refresh_access_token_with_organization(
                domain=domain,
                client_id=client_id,
                refresh_token=token.refresh_token,
                organization=chosen_org_id,
                audience=audience,
                scopes=scopes,
            )
        except EarlError as exc:
            raise SystemExit(
                f"Auth0 refused to issue an org-scoped token for "
                f"{chosen_org_id}: {exc}"
            )
        org_info = extract_org_info(token.access_token)
        if org_info.org_name:
            chosen_org_name = org_info.org_name

    # Persist.
    if getattr(args, "name", None):
        profile_name = args.name
    else:
        slug = slugify_org(chosen_org_name, chosen_org_id)
        profile_name = f"pkce-{args.env}-{slug}" if slug else f"pkce-{args.env}"

    prof = AuthProfile.create_pkce(
        name=profile_name,
        client_id=client_id,
        clear_refresh_token=token.refresh_token,
        organization=chosen_org_id,
        environment=args.env,
    )
    store.upsert_profile(prof)
    if getattr(args, "activate", True):
        store.set_active_profile(prof.name)

    cache_key = auth_storage.token_cache_key(
        client_id, audience, chosen_org_id, domain, "pkce"
    )
    auth_storage.save_token(
        cache_key,
        auth_storage.CachedToken(
            access_token=token.access_token,
            token_type=token.token_type,
            expires_at=token.expires_at,
            scope=token.scope or None,
            refresh_token=token.refresh_token,
        ),
    )

    backend_note = "OS keyring" if prof.secret_backend == "keyring" else "config.json (base64)"
    who = f" ({org_info.email})" if org_info.email else ""
    org_label = chosen_org_name or chosen_org_id or "no organization"
    _say(
        args,
        f"Signed in to env={args.env}{who} as profile '{prof.name}' "
        f"→ {org_label}. Refresh token stored in {backend_note}.",
    )


def _handle_service_account(args: argparse.Namespace, store: ConfigStore) -> None:
    """Dispatch ``earl service-account <cmd>``."""
    if args.sa_cmd == "create":
        _handle_service_account_create(args, store)
    elif args.sa_cmd == "list":
        _handle_service_account_list(args, store)
    elif args.sa_cmd == "revoke":
        _handle_service_account_revoke(args, store)
    else:
        raise ValueError(f"Unknown service-account command: {args.sa_cmd}")


def _parse_scopes(raw: str) -> list[str]:
    """Accept either space- or comma-separated scope lists."""
    out: list[str] = []
    for chunk in raw.replace(",", " ").split():
        chunk = chunk.strip()
        if chunk and chunk not in out:
            out.append(chunk)
    return out


def _svc_request(
    client: EarlClient, method: str, path: str, *, json_body: dict | None = None
) -> dict:
    """Issue an authenticated request against the orchestrator, returning JSON.

    The Earl Python SDK intentionally does not expose a generic "raw request"
    verb on :class:`EarlClient` — route helpers are per-resource. Service
    accounts are a tiny surface that only CLI calls use, so we reach directly
    into the shared httpx transport here rather than wiring a new
    ``ServiceAccountsAPI`` module.

    ``path`` may start with ``/api/v1/...`` (absolute) or a bare ``/...``
    (relative to ``client.api_url``). We normalise so callers don't need to
    remember whether the base URL already has the ``/api/v1`` prefix.
    """
    from earl_sdk import _http
    from urllib.parse import urlsplit

    base = client.api_url.rstrip("/")
    base_path = urlsplit(base).path.rstrip("/")
    p = "/" + path.lstrip("/")
    if base_path and p.startswith(base_path + "/"):
        p = p[len(base_path):]
    url = base + p
    headers = client._auth.get_headers()
    body, _resp = _http.request_json(
        method,
        url,
        headers=headers,
        json_body=json_body,
        timeout=30.0,
    )
    return body


def _handle_service_account_create(
    args: argparse.Namespace, store: ConfigStore
) -> None:
    from earl_sdk.exceptions import EarlError

    scopes = _parse_scopes(args.scopes)
    body: dict = {"name": args.name, "scopes": scopes}
    if args.description:
        body["description"] = args.description
    if args.org_id:
        body["org_id"] = args.org_id

    if _dry_run_emit(args, "service-account.create", body):
        return

    client = _build_client(args, store)
    try:
        resp = _svc_request(client, "POST", "/api/v1/service-accounts", json_body=body)
    except EarlError as exc:
        raise SystemExit(f"Service-account create failed: {exc}")

    # Machine-readable output — always available via the global --json flag.
    if getattr(args, "json_output", False) or getattr(args, "output", "table") == "json":
        print(json.dumps(resp, indent=2))
        return

    warn = resp.get("warning") or (
        "Store the client_secret now — it cannot be shown again."
    )
    print("", file=sys.stderr)
    print(f"Created service account '{resp.get('name')}'", file=sys.stderr)
    print(f"  id:           {resp.get('client_id')}", file=sys.stderr)
    print(f"  org:          {resp.get('org_id')}", file=sys.stderr)
    print(f"  scopes:       {', '.join(resp.get('scopes') or [])}", file=sys.stderr)
    print("", file=sys.stderr)
    # Secret goes to stdout so shell redirection captures it cleanly; the
    # human-readable context goes to stderr.
    print(f"EARL_CLIENT_ID={resp.get('client_id')}")
    print(f"EARL_CLIENT_SECRET={resp.get('client_secret')}")
    print(f"EARL_ORG_ID={resp.get('org_id')}")
    print("", file=sys.stderr)
    print(f"⚠  {warn}", file=sys.stderr)


def _handle_service_account_list(
    args: argparse.Namespace, store: ConfigStore
) -> None:
    from earl_sdk.exceptions import EarlError

    if _dry_run_emit(args, "service-account.list", {"org_id": args.org_id or None}):
        return

    client = _build_client(args, store)
    path = "/api/v1/service-accounts"
    if args.org_id:
        path += "?org_id=" + args.org_id
    try:
        resp = _svc_request(client, "GET", path)
    except EarlError as exc:
        raise SystemExit(f"Service-account list failed: {exc}")

    accts = resp.get("service_accounts") or []
    if getattr(args, "json_output", False) or getattr(args, "output", "table") == "json":
        print(json.dumps({"org_id": resp.get("org_id"), "service_accounts": accts}, indent=2))
        return

    if not accts:
        _say(args, f"No service accounts for org {resp.get('org_id')}.")
        return
    rows = [
        {
            "id": a.get("id") or a.get("client_id"),
            "name": a.get("name", ""),
            "scopes": " ".join(a.get("scopes") or []),
            "created_by": a.get("created_by", ""),
            "created_at": a.get("created_at", ""),
        }
        for a in accts
    ]
    _print_rows(rows, headers=["id", "name", "scopes", "created_by", "created_at"])


def _handle_service_account_revoke(
    args: argparse.Namespace, store: ConfigStore
) -> None:
    from earl_sdk.exceptions import EarlError

    target = args.sa_client_id

    if _dry_run_emit(args, "service-account.revoke", {"client_id": target}):
        return

    # Confirm destructive action. ``--yes`` opts out (required in CI).
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to revoke non-interactively; pass --yes to confirm."
            )
        resp = input(
            f"Revoke service account {target}? This is irreversible. [y/N]: "
        ).strip().lower()
        if resp != "y":
            _say(args, "Aborted.")
            return

    client = _build_client(args, store)
    try:
        _svc_request(
            client, "DELETE", f"/api/v1/service-accounts/{target}"
        )
    except EarlError as exc:
        raise SystemExit(f"Service-account revoke failed: {exc}")

    _say(args, f"Revoked service account {target}.")


# ── earl org … ──────────────────────────────────────────────────────────────
#
# Every subcommand below is a thin client of ``/api/v1/organizations`` on the
# orchestrator, which in turn wraps the Auth0 Management API. The backend
# enforces the role matrix (``EARL_Admin`` global, ``EARL_Org_Admin`` per-org).
# The CLI just shapes input/output and flags destructive operations with
# ``--yes`` confirmation.


def _parse_metadata_pairs(raw: list[str]) -> dict[str, str]:
    """Parse ``['k=v', 'k2=v2']`` → ``{"k": "v", "k2": "v2"}``.

    Raises :class:`SystemExit` on malformed pairs so users get a friendly
    error instead of a stack trace.
    """
    out: dict[str, str] = {}
    for pair in raw:
        if "=" not in pair:
            raise SystemExit(
                f"Invalid --metadata value {pair!r}. Expected KEY=VALUE."
            )
        k, v = pair.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Empty metadata key in {pair!r}.")
        out[k] = v
    return out


def _resolve_target_org(
    args: argparse.Namespace, client: EarlClient, *, required: bool = True
) -> str:
    """Resolve the target ``org_id`` for a per-org ``earl org ...`` command.

    Order: explicit positional ``org_id`` > ``--organization`` on the client >
    whatever the active profile/env exposes. Errors out if still empty and
    ``required``.
    """
    org_id = getattr(args, "org_id", None) or client.organization or ""
    if not org_id and required:
        raise SystemExit(
            "No organization specified. Pass an ORG_ID positional or set one via "
            "--organization / EARL_ORG_ID / your active PKCE profile."
        )
    return org_id


def _org_want_json(args: argparse.Namespace) -> bool:
    """True iff we should emit raw JSON (``--json`` global or ``--format json``)."""
    if getattr(args, "json_output", False):
        return True
    if getattr(args, "output", "table") == "json":
        return True
    return getattr(args, "org_format", None) == "json"


def _handle_org(args: argparse.Namespace, store: ConfigStore) -> None:
    """Dispatch ``earl org <cmd>``."""
    cmd = args.org_cmd
    if cmd == "list":
        _handle_org_list(args, store)
    elif cmd == "create":
        _handle_org_create(args, store)
    elif cmd == "show":
        _handle_org_show(args, store)
    elif cmd == "update":
        _handle_org_update(args, store)
    elif cmd == "delete":
        _handle_org_delete(args, store)
    elif cmd == "members":
        _handle_org_members(args, store)
    elif cmd == "invite":
        _handle_org_invite(args, store)
    elif cmd == "invitations":
        _handle_org_invitations(args, store)
    elif cmd == "roles":
        _handle_org_roles(args, store)
    else:
        raise ValueError(f"Unknown org command: {cmd}")


def _handle_org_list(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    if _dry_run_emit(args, "org.list", {}):
        return
    client = _build_client(args, store)
    try:
        resp = _svc_request(client, "GET", "/api/v1/organizations")
    except EarlError as exc:
        raise SystemExit(f"org list failed: {exc}")
    orgs = resp.get("organizations") or []
    if _org_want_json(args):
        print(json.dumps({"organizations": orgs}, indent=2))
        return
    if not orgs:
        _say(args, "No organizations on this tenant.")
        return
    _print_rows(
        [
            {"id": o.get("id", ""), "name": o.get("name", ""), "display_name": o.get("display_name", "")}
            for o in orgs
        ],
        headers=["id", "name", "display_name"],
    )


def _handle_org_create(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    body: dict[str, Any] = {"name": args.name}
    if args.display_name:
        body["display_name"] = args.display_name
    if args.metadata:
        body["metadata"] = _parse_metadata_pairs(args.metadata)
    if _dry_run_emit(args, "org.create", body):
        return
    client = _build_client(args, store)
    try:
        resp = _svc_request(client, "POST", "/api/v1/organizations", json_body=body)
    except EarlError as exc:
        raise SystemExit(f"org create failed: {exc}")
    if _org_want_json(args):
        print(json.dumps(resp, indent=2))
        return
    print(f"Created organization: {resp.get('id')} ({resp.get('name')})", file=sys.stderr)
    _print_rows([resp], headers=["id", "name", "display_name"])


def _handle_org_show(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    client = _build_client(args, store)
    org_id = _resolve_target_org(args, client, required=True)
    if _dry_run_emit(args, "org.show", {"org_id": org_id}):
        return
    try:
        resp = _svc_request(client, "GET", f"/api/v1/organizations/{org_id}")
    except EarlError as exc:
        raise SystemExit(f"org show failed: {exc}")
    if _org_want_json(args):
        print(json.dumps(resp, indent=2))
        return
    _print_rows([resp], headers=["id", "name", "display_name"])
    if resp.get("metadata"):
        print("metadata:", file=sys.stderr)
        for k, v in resp["metadata"].items():
            print(f"  {k}: {v}", file=sys.stderr)


def _handle_org_update(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    body: dict[str, Any] = {}
    if args.display_name is not None:
        body["display_name"] = args.display_name
    if args.metadata:
        body["metadata"] = _parse_metadata_pairs(args.metadata)
    if not body:
        raise SystemExit("Nothing to update; pass --display-name and/or --metadata.")
    if _dry_run_emit(args, "org.update", {"org_id": args.org_id, **body}):
        return
    client = _build_client(args, store)
    try:
        resp = _svc_request(
            client, "PATCH", f"/api/v1/organizations/{args.org_id}", json_body=body
        )
    except EarlError as exc:
        raise SystemExit(f"org update failed: {exc}")
    if _org_want_json(args):
        print(json.dumps(resp, indent=2))
        return
    _print_rows([resp], headers=["id", "name", "display_name"])


def _handle_org_delete(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    target = args.org_id
    if _dry_run_emit(args, "org.delete", {"org_id": target}):
        return
    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit(
                "Refusing to delete non-interactively; pass --yes to confirm."
            )
        resp = input(
            f"Delete organization {target}? This is irreversible. [y/N]: "
        ).strip().lower()
        if resp != "y":
            _say(args, "Aborted.")
            return
    client = _build_client(args, store)
    try:
        _svc_request(client, "DELETE", f"/api/v1/organizations/{target}")
    except EarlError as exc:
        raise SystemExit(f"org delete failed: {exc}")
    _say(args, f"Deleted organization {target}.")


def _handle_org_members(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    sub = args.org_members_cmd
    if sub == "list":
        client = _build_client(args, store)
        org_id = _resolve_target_org(args, client, required=True)
        if _dry_run_emit(args, "org.members.list", {"org_id": org_id}):
            return
        try:
            resp = _svc_request(
                client, "GET", f"/api/v1/organizations/{org_id}/members"
            )
        except EarlError as exc:
            raise SystemExit(f"org members list failed: {exc}")
        members = resp.get("members") or []
        if _org_want_json(args):
            print(json.dumps(resp, indent=2))
            return
        if not members:
            _say(args, f"No members in org {org_id}.")
            return
        _print_rows(
            [
                {
                    "user_id": m.get("user_id", ""),
                    "email": m.get("email", ""),
                    "name": m.get("name", ""),
                    "roles": " ".join(m.get("roles") or []),
                }
                for m in members
            ],
            headers=["user_id", "email", "name", "roles"],
        )
    elif sub == "remove":
        target = args.user_id
        if _dry_run_emit(
            args, "org.members.remove", {"org_id": args.org_id, "user_id": target}
        ):
            return
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit(
                    "Refusing to remove non-interactively; pass --yes to confirm."
                )
            resp = input(
                f"Remove {target} from org {args.org_id}? [y/N]: "
            ).strip().lower()
            if resp != "y":
                _say(args, "Aborted.")
                return
        client = _build_client(args, store)
        try:
            _svc_request(
                client,
                "DELETE",
                f"/api/v1/organizations/{args.org_id}/members/{target}",
            )
        except EarlError as exc:
            raise SystemExit(f"org members remove failed: {exc}")
        _say(args, f"Removed {target} from {args.org_id}.")
    else:
        raise ValueError(f"Unknown org members subcommand: {sub}")


def _handle_org_invite(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    body: dict[str, Any] = {"email": args.email}
    if args.role:
        body["roles"] = list(args.role)
    if args.ttl_seconds:
        body["ttl_seconds"] = args.ttl_seconds
    if _dry_run_emit(args, "org.invite", {"org_id": args.org_id, **body}):
        return
    client = _build_client(args, store)
    try:
        resp = _svc_request(
            client,
            "POST",
            f"/api/v1/organizations/{args.org_id}/invitations",
            json_body=body,
        )
    except EarlError as exc:
        raise SystemExit(f"org invite failed: {exc}")
    if _org_want_json(args):
        print(json.dumps(resp, indent=2))
        return
    print(
        f"Invited {resp.get('email')} to {resp.get('organization_id')} "
        f"(invitation {resp.get('id')}, expires {resp.get('expires_at') or 'default'})",
        file=sys.stderr,
    )
    url = resp.get("invitation_url")
    if url:
        # Acceptance link goes to stdout so it can be captured and emailed.
        print(url)


def _handle_org_invitations(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    sub = args.org_invs_cmd
    if sub == "list":
        client = _build_client(args, store)
        org_id = _resolve_target_org(args, client, required=True)
        if _dry_run_emit(args, "org.invitations.list", {"org_id": org_id}):
            return
        try:
            resp = _svc_request(
                client, "GET", f"/api/v1/organizations/{org_id}/invitations"
            )
        except EarlError as exc:
            raise SystemExit(f"org invitations list failed: {exc}")
        invs = resp.get("invitations") or []
        if _org_want_json(args):
            print(json.dumps(resp, indent=2))
            return
        if not invs:
            _say(args, f"No pending invitations in org {org_id}.")
            return
        _print_rows(
            [
                {
                    "id": i.get("id", ""),
                    "email": i.get("email", ""),
                    "expires_at": i.get("expires_at", ""),
                }
                for i in invs
            ],
            headers=["id", "email", "expires_at"],
        )
    elif sub == "revoke":
        target = args.invitation_id
        if _dry_run_emit(
            args, "org.invitations.revoke",
            {"org_id": args.org_id, "invitation_id": target},
        ):
            return
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit(
                    "Refusing to revoke non-interactively; pass --yes to confirm."
                )
            resp = input(f"Revoke invitation {target}? [y/N]: ").strip().lower()
            if resp != "y":
                _say(args, "Aborted.")
                return
        client = _build_client(args, store)
        try:
            _svc_request(
                client,
                "DELETE",
                f"/api/v1/organizations/{args.org_id}/invitations/{target}",
            )
        except EarlError as exc:
            raise SystemExit(f"org invitations revoke failed: {exc}")
        _say(args, f"Revoked invitation {target}.")
    else:
        raise ValueError(f"Unknown invitations subcommand: {sub}")


def _handle_org_roles(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.exceptions import EarlError

    sub = args.org_roles_cmd
    if sub == "catalog":
        if _dry_run_emit(args, "org.roles.catalog", {}):
            return
        client = _build_client(args, store)
        try:
            resp = _svc_request(
                client, "GET", "/api/v1/organizations/_roles/catalog"
            )
        except EarlError as exc:
            raise SystemExit(f"org roles catalog failed: {exc}")
        roles = resp.get("roles") or []
        if _org_want_json(args):
            print(json.dumps({"roles": roles}, indent=2))
            return
        if not roles:
            _say(args, "No assignable tenant-level roles configured.")
            return
        _print_rows(
            [
                {"name": r.get("name", ""), "description": r.get("description", "")}
                for r in roles
            ],
            headers=["name", "description"],
        )
        return

    body: dict[str, Any] = {"roles": list(args.roles)}
    if sub == "grant":
        if _dry_run_emit(
            args, "org.roles.grant",
            {"org_id": args.org_id, "user_id": args.user_id, **body},
        ):
            return
        client = _build_client(args, store)
        try:
            resp = _svc_request(
                client,
                "POST",
                f"/api/v1/organizations/{args.org_id}/members/{args.user_id}/roles",
                json_body=body,
            )
        except EarlError as exc:
            raise SystemExit(f"org roles grant failed: {exc}")
        if _org_want_json(args):
            print(json.dumps(resp, indent=2))
            return
        _say(
            args,
            f"Granted roles={resp.get('roles')} to {args.user_id} in {args.org_id}.",
        )
    elif sub == "revoke":
        if _dry_run_emit(
            args, "org.roles.revoke",
            {"org_id": args.org_id, "user_id": args.user_id, **body},
        ):
            return
        client = _build_client(args, store)
        try:
            resp = _svc_request(
                client,
                "DELETE",
                f"/api/v1/organizations/{args.org_id}/members/{args.user_id}/roles",
                json_body=body,
            )
        except EarlError as exc:
            raise SystemExit(f"org roles revoke failed: {exc}")
        if _org_want_json(args):
            print(json.dumps(resp, indent=2))
            return
        _say(
            args,
            f"Revoked roles={resp.get('roles')} from {args.user_id} in {args.org_id}.",
        )
    else:
        raise ValueError(f"Unknown roles subcommand: {sub}")


def _handle_doctor(args: argparse.Namespace, store: ConfigStore) -> None:
    if args.doctor_cmd == "add":
        if args.type == "external" and not args.api_url:
            raise ValueError("--api-url is required for external doctor")
        if _dry_run_emit(
            args,
            "doctor.add",
            {
                "name": args.name,
                "type": args.type,
                "api_url": args.api_url or "",
                "auth_type": args.auth_type,
                "api_key": "<redacted>" if args.api_key else None,
            },
        ):
            return
        api_key: str = ""
        if args.type == "external":
            if args.api_key is None and sys.stdin.isatty():
                resp = input("Provide an API key for this doctor? [y/N]: ").strip().lower()
                if resp == "y":
                    api_key = _resolve_secret(None, prompt="Doctor API key: ")
            elif args.api_key is not None:
                api_key = _resolve_secret(args.api_key, prompt="Doctor API key: ")
        cfg = DoctorConfig.create(
            name=args.name,
            type=args.type,
            api_url=args.api_url or "",
            clear_api_key=api_key,
            auth_type=args.auth_type,
        )
        store.upsert_doctor_config(cfg)
        backend_note = (
            "OS keyring" if cfg.secret_backend == "keyring" and api_key else "config.json"
        )
        _say(args, f"Saved doctor config: {cfg.name} ({backend_note})")
        return
    if args.doctor_cmd == "list":
        rows = [
            {
                "name": d.name,
                "type": d.type,
                "api_url": d.api_url,
                "auth_type": d.auth_type,
            }
            for d in store.list_doctor_configs()
        ]
        _print_rows(rows, headers=["name", "type", "api_url", "auth_type"])
        return
    if args.doctor_cmd == "show":
        d = store.config.doctor_configs.get(args.name)
        if not d:
            raise ValueError(f"Doctor config not found: {args.name}")
        print(
            json.dumps(
                {
                    "name": d.name,
                    "type": d.type,
                    "api_url": d.api_url,
                    "api_key_tail": f"...{d.key_clear()[-4:]}" if d.api_key else "",
                    "auth_type": d.auth_type,
                },
                indent=2,
            )
        )
        return
    if args.doctor_cmd == "delete":
        if _dry_run_emit(args, "doctor.delete", {"name": args.name}):
            return
        store.delete_doctor_config(args.name)
        _say(args, f"Deleted doctor config: {args.name}")
        return
    if args.doctor_cmd == "validate":
        d = store.config.doctor_configs.get(args.name)
        if not d:
            raise ValueError(f"Doctor config not found: {args.name}")
        if d.type != "external":
            raise ValueError("Only external doctor configs can be validated")
        client = _build_client(args, store)
        result = client.pipelines.validate_doctor_api(
            api_url=d.api_url,
            api_key=d.key_clear() or None,
            timeout=args.timeout,
        )
        _emit(args, result)
        return
    raise ValueError("Unsupported doctor command")


def _handle_cases(args: argparse.Namespace, client: EarlClient) -> None:
    if args.cases_cmd == "list":
        _emit(args, client.cases.list())
        return
    if args.cases_cmd == "get":
        _emit(args, client.cases.get(args.case_id))
        return
    raise ValueError("Unsupported cases command")


def _handle_patients(args: argparse.Namespace, client: EarlClient) -> None:
    if args.patients_cmd == "list":
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        data = client.patients.list(
            difficulty=args.difficulty,
            tags=tags,
            limit=args.limit,
            offset=args.offset,
        )
        _emit(args, [_to_dict(x) for x in data])
        return
    if args.patients_cmd == "get":
        _emit(args, _to_dict(client.patients.get(args.patient_id)))
        return
    raise ValueError("Unsupported patients command")


def _handle_dimensions(args: argparse.Namespace, client: EarlClient) -> None:
    if args.dimensions_cmd == "list":
        _emit(
            args, [_to_dict(d) for d in client.dimensions.list(include_custom=args.include_custom)]
        )
        return
    if args.dimensions_cmd == "get":
        _emit(args, _to_dict(client.dimensions.get(args.dimension_id)))
        return
    if args.dimensions_cmd == "create":
        payload = {
            "name": args.name,
            "description": args.description,
            "category": args.category,
            "weight": args.weight,
        }
        if _dry_run_emit(args, "dimensions.create", payload):
            return
        created = client.dimensions.create(**payload)
        _emit(args, _to_dict(created))
        return
    raise ValueError("Unsupported dimensions command")


def _handle_verifiers(args: argparse.Namespace, client: EarlClient) -> None:
    if args.verifiers_cmd != "list":
        raise ValueError("Unsupported verifiers command")
    payload = client.verifiers.list()
    if args.type == "all":
        _emit(args, payload)
    else:
        key = "hard-gates" if args.type == "hard-gates" else "scoring-dimensions"
        _emit(args, {key: payload.get(key, {})})


def _handle_pipelines(args: argparse.Namespace, client: EarlClient) -> None:
    if args.pipelines_cmd == "list":
        data = client.pipelines.list(active_only=args.active_only)
        _emit(args, [_to_dict(x) for x in data])
        return
    if args.pipelines_cmd == "get":
        _emit(args, _to_dict(client.pipelines.get(args.pipeline_name)))
        return
    if args.pipelines_cmd == "delete":
        if _dry_run_emit(args, "pipelines.delete", {"pipeline_name": args.pipeline_name}):
            return
        client.pipelines.delete(args.pipeline_name)
        _say(args, f"Deleted pipeline: {args.pipeline_name}")
        return
    if args.pipelines_cmd == "validate-doctor":
        out = client.pipelines.validate_doctor_api(
            api_url=args.api_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        _emit(args, out)
        return
    if args.pipelines_cmd in {"create", "update"}:
        kwargs = _pipeline_kwargs_from_args(args)
        payload = {
            **(
                {"name": args.name}
                if args.pipelines_cmd == "create"
                else {"pipeline_name": args.pipeline_name}
            ),
            **{k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in kwargs.items()},
        }
        # Redact secrets in --dry-run output (doctor_config may carry an api_key).
        dc = payload.get("doctor_config")
        if isinstance(dc, dict) and dc.get("api_key"):
            dc["api_key"] = "<redacted>"
        if _dry_run_emit(args, f"pipelines.{args.pipelines_cmd}", payload):
            return
        if args.pipelines_cmd == "create":
            pipe = client.pipelines.create(name=args.name, **kwargs)
        else:
            # ``update`` accepts a subset of ``create``'s kwargs — filter
            # anything else out so we don't trip a TypeError.
            update_allowed = {
                "verifier_ids",
                "doctor_config",
                "patient_ids",
                "description",
                "conversation_initiator",
                "max_turns",
                "verifiers",
                "dimension_ids",
            }
            update_kwargs = {k: v for k, v in kwargs.items() if k in update_allowed}
            pipe = client.pipelines.update(args.pipeline_name, **update_kwargs)
        _emit(args, _to_dict(pipe))
        return
    raise ValueError("Unsupported pipelines command")


def _pipeline_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    verifier_ids = list(args.verifier_id or [])
    if args.verifier_ids:
        verifier_ids.extend([x.strip() for x in args.verifier_ids.split(",") if x.strip()])

    patient_ids = list(args.patient_id or [])
    if args.patient_ids:
        patient_ids.extend([x.strip() for x in args.patient_ids.split(",") if x.strip()])

    kwargs: dict[str, Any] = {}
    if args.description is not None:
        kwargs["description"] = args.description
    if args.case_id is not None:
        kwargs["case_id"] = args.case_id
    if verifier_ids:
        kwargs["verifier_ids"] = verifier_ids
    if patient_ids:
        kwargs["patient_ids"] = patient_ids

    if args.doctor == "internal":
        kwargs["doctor_config"] = DoctorApiConfig.internal()
    elif args.doctor == "external":
        if not args.doctor_api_url:
            raise ValueError("--doctor-api-url is required when --doctor=external")
        kwargs["doctor_config"] = DoctorApiConfig.external(
            api_url=args.doctor_api_url,
            api_key=args.doctor_api_key,
            auth_type=args.doctor_auth_type,
        )
    elif args.doctor == "client_driven":
        kwargs["doctor_config"] = DoctorApiConfig.client_driven()

    if args.validate_doctor is not None:
        kwargs["validate_doctor"] = args.validate_doctor
    if args.conversation_initiator is not None:
        kwargs["conversation_initiator"] = args.conversation_initiator
    if args.max_turns is not None:
        kwargs["max_turns"] = args.max_turns
    if args.verifiers is not None:
        kwargs["verifiers"] = args.verifiers
    return kwargs


def _handle_simulations(args: argparse.Namespace, client: EarlClient) -> None:
    if args.simulations_cmd == "list":
        status = SimulationStatus(args.status) if args.status else None
        sims = client.simulations.list(status=status, limit=args.limit, skip=args.skip)
        _emit(args, [_to_dict(x) for x in sims])
        return
    if args.simulations_cmd == "get":
        _emit(args, _to_dict(client.simulations.get(args.simulation_id)))
        return
    if args.simulations_cmd == "start":
        payload = {
            "pipeline_name": args.pipeline,
            "num_episodes": args.num_episodes,
            "parallel_count": args.parallel_count,
        }
        if _dry_run_emit(args, "simulations.start", payload):
            return
        sim = client.simulations.create(**payload)
        _emit(args, _to_dict(sim))
        return
    if args.simulations_cmd == "wait":

        def _progress(sim):
            print(
                f"{sim.id[:8]} status={sim.status.value} "
                f"episodes={sim.completed_episodes}/{sim.total_episodes}"
            )

        done = client.simulations.wait_for_completion(
            args.simulation_id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            on_progress=_progress,
        )
        _emit(args, _to_dict(done))
        return
    if args.simulations_cmd == "stop":
        if _dry_run_emit(args, "simulations.stop", {"simulation_id": args.simulation_id}):
            return
        _emit(args, _to_dict(client.simulations.stop(args.simulation_id)))
        return
    if args.simulations_cmd == "episodes":
        _emit(
            args,
            client.simulations.get_episodes(
                args.simulation_id,
                include_dialogue=args.include_dialogue,
            ),
        )
        return
    if args.simulations_cmd == "episode":
        _emit(args, client.simulations.get_episode(args.simulation_id, args.episode_id))
        return
    if args.simulations_cmd == "report":
        report = client.simulations.get_report(args.simulation_id)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Saved: {args.save}")
            return
        _emit(args, report)
        return
    if args.simulations_cmd == "pending":
        pending = client.simulations.get_pending_episodes(args.simulation_id)
        _emit(args, pending)
        return
    if args.simulations_cmd == "respond":
        message = args.message
        if args.message_file:
            with open(args.message_file, encoding="utf-8") as f:
                message = f.read()
        if not message:
            raise ValueError("Provide --message or --message-file")
        if _dry_run_emit(
            args,
            "simulations.respond",
            {
                "simulation_id": args.simulation_id,
                "episode_id": args.episode_id,
                "message_length": len(message),
            },
        ):
            return
        out = client.simulations.submit_response(
            args.simulation_id,
            args.episode_id,
            message,
        )
        _emit(args, out)
        return
    raise ValueError("Unsupported simulations command")


def _handle_runs(args: argparse.Namespace, runs: RunStore, store: ConfigStore) -> None:
    if args.runs_cmd == "list":
        entries = [asdict(x) for x in runs.list_runs(limit=args.limit)]
        _print_rows(
            entries,
            headers=["simulation_id", "pipeline_name", "status", "average_score", "saved_at"],
        )
        return
    if args.runs_cmd == "show":
        report = runs.load_report(args.simulation_id)
        if not report:
            raise ValueError(f"No local run found for: {args.simulation_id}")
        print(json.dumps(report, indent=2, default=str))
        return
    if args.runs_cmd == "delete":
        ok = runs.delete_run(args.simulation_id)
        if not ok:
            raise ValueError(f"No local run found for: {args.simulation_id}")
        print(f"Deleted local run: {args.simulation_id}")
        return
    if args.runs_cmd == "prune":
        deleted = runs.prune(max_runs=args.max_runs)
        print(f"Pruned {deleted} run(s)")
        return
    if args.runs_cmd in {"save", "compare"}:
        client = _build_client(args, store)
        if args.runs_cmd == "save":
            sim = client.simulations.get(args.simulation_id)
            report = client.simulations.get_report(args.simulation_id)
            meta = LocalRun(
                simulation_id=sim.id,
                pipeline_name=sim.pipeline_name,
                status=sim.status.value,
                total_episodes=sim.total_episodes,
                completed_episodes=sim.completed_episodes,
                average_score=(sim.summary or {}).get("average_score"),
                started_at=str(sim.started_at) if sim.started_at else "",
                finished_at=str(sim.finished_at) if sim.finished_at else "",
                environment=client.environment,
            )
            runs.save_run(meta, report)
            print(f"Saved local run: {meta.short_id}")
            return
        # compare
        rows = []
        for sim_id in args.simulation_ids:
            run = runs.get_run(sim_id)
            if run:
                rows.append(run)
                continue
            sim = client.simulations.get(sim_id)
            rows.append(
                LocalRun(
                    simulation_id=sim.id,
                    pipeline_name=sim.pipeline_name,
                    status=sim.status.value,
                    total_episodes=sim.total_episodes,
                    completed_episodes=sim.completed_episodes,
                    average_score=(sim.summary or {}).get("average_score"),
                    started_at=str(sim.started_at) if sim.started_at else "",
                    finished_at=str(sim.finished_at) if sim.finished_at else "",
                    environment=client.environment,
                )
            )
        _print_rows(
            [asdict(r) for r in rows],
            headers=[
                "simulation_id",
                "pipeline_name",
                "status",
                "completed_episodes",
                "total_episodes",
                "average_score",
                "environment",
            ],
        )
        return
    raise ValueError("Unsupported runs command")


def _handle_config(args: argparse.Namespace, store: ConfigStore) -> None:
    from earl_sdk.interactive.storage.config_store import CONFIG_PATH

    if args.config_cmd == "path":
        print(f"config={CONFIG_PATH}")
        print(f"runs={CONFIG_PATH.parent / 'runs'}")
        return
    if args.config_cmd == "show":
        profile = store.get_active_profile()
        data = {
            "active_profile": profile.name if profile else "",
            "profiles": list(store.config.profiles.keys()),
            "doctor_configs": list(store.config.doctor_configs.keys()),
            "preferences": asdict(store.preferences),
        }
        print(json.dumps(data, indent=2))
        return
    raise ValueError("Unsupported config command")


def _handle_prefs(args: argparse.Namespace, store: ConfigStore) -> None:
    prefs = store.preferences
    if args.prefs_cmd == "show":
        print(json.dumps(asdict(prefs), indent=2))
        return
    if args.prefs_cmd == "set":
        if args.default_pipeline is not None:
            prefs.default_pipeline = args.default_pipeline
        if args.default_parallel is not None:
            prefs.default_parallel = args.default_parallel
        if args.auto_save_runs is not None:
            prefs.auto_save_runs = args.auto_save_runs == "true"
        if args.max_local_runs is not None:
            prefs.max_local_runs = args.max_local_runs
        store.save_preferences(prefs)
        print(json.dumps(asdict(prefs), indent=2))
        return
    raise ValueError("Unsupported prefs command")


def _handle_chat(
    args: argparse.Namespace, client: EarlClient, store: ConfigStore, runs: RunStore
) -> None:
    if args.chat_cmd != "start":
        raise ValueError("Unsupported chat command")
    try:
        from earl_sdk.interactive.flows.chat import flow_chat
    except Exception as exc:
        raise RuntimeError(
            "Interactive chat requires UI extras. Install: pip install 'earl-sdk[ui]'"
        ) from exc

    flow_chat(client, store, runs)


def _emit(args: argparse.Namespace, data: Any) -> None:
    if args.output == "json":
        print(json.dumps(data, indent=2, default=str))
        return
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            headers = sorted({k for row in data for k in row.keys()})
            _print_rows(data, headers=headers)
            return
    print(json.dumps(data, indent=2, default=str))


def _print_rows(rows: Iterable[dict[str, Any]], headers: list[str]) -> None:
    rows = list(rows)
    if not rows:
        print("(empty)")
        return
    widths = {h: len(h) for h in headers}
    rendered = []
    for row in rows:
        out_row = {}
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                sval = f"{val:.3f}"
            elif isinstance(val, (int, bool)):
                sval = str(val)
            elif isinstance(val, datetime):
                sval = val.isoformat()
            else:
                sval = str(val)
            out_row[h] = sval
            widths[h] = max(widths[h], len(sval))
        rendered.append(out_row)

    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rendered:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
