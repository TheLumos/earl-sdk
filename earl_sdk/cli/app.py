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
from typing import Any

from earl_sdk import DoctorApiConfig, EarlClient, SimulationStatus, __version__, auth_storage
from earl_sdk.interactive.storage.config_store import AuthProfile, ConfigStore, DoctorConfig
from earl_sdk.interactive.storage.run_store import LocalRun, RunStore

from . import schema as _schema_mod

logger = logging.getLogger("earl_sdk.cli")


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

Authentication is resolved from (in order): command-line flags, ``--profile``
name in ``~/.earl/config.json``, then env vars
(``EARL_CLIENT_ID``/``EARL_CLIENT_SECRET``/``EARL_ORGANIZATION``/``EARL_ENVIRONMENT``).
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
  echo "$CLIENT_SECRET" | earl auth profile add --name ci --client-id ... --env test --client-secret -

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
        "--env", choices=["local", "dev", "test", "prod"], help="Environment override"
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
            "  earl auth profile add --name ci --client-id abc --env test --client-secret -\n"
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
    prof_add.add_argument("--env", required=True, choices=["local", "dev", "test", "prod"])

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
            "Requires a public 'Native' Auth0 application per env with the "
            "Device Code + Refresh Token grants enabled. Register its client "
            "ID with `earl auth device-clients set --env <env> <client_id>` "
            "or pass --device-client-id each run."
        ),
        epilog=(
            "Examples:\n"
            "  earl auth login --env test\n"
            "  earl auth login --env prod --organization org_abc123\n"
            "  earl auth login --env dev --device-client-id abc123 --no-browser\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    auth_login.add_argument("--env", required=True, choices=["local", "dev", "test", "prod"])
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
            "Auth0 tenant. One ID per env: local/dev/test/prod."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    devclients_sub = auth_devclients.add_subparsers(dest="devclients_cmd", required=True)
    devclients_sub.add_parser("list", help="List configured device-flow client IDs")
    dc_set = devclients_sub.add_parser("set", help="Register a client ID for an env")
    dc_set.add_argument("--env", required=True, choices=["local", "dev", "test", "prod"])
    dc_set.add_argument("client_id")
    dc_clear = devclients_sub.add_parser("clear", help="Remove the client ID for an env")
    dc_clear.add_argument("--env", required=True, choices=["local", "dev", "test", "prod"])

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
                    "message_preview": (message[:120] + "…") if len(message) > 120 else message,
                },
            )
    return False


def _dispatch(args: argparse.Namespace, store: ConfigStore, runs: RunStore) -> None:
    if args.command == "auth":
        _handle_auth(args, store)
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
        _emit(
            args,
            {
                "environment": client.environment,
                "organization": client.organization,
                "api_url": client.api_url,
                "service_api_urls": client.service_api_urls,
            },
        )
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


def _build_client(args: argparse.Namespace, store: ConfigStore) -> EarlClient:
    profile = None
    if args.profile:
        profile = store.config.profiles.get(args.profile)
        if not profile:
            raise ValueError(f"Profile not found: {args.profile}")
    elif store.config.active_profile:
        profile = store.get_active_profile()

    auth_kind = "m2m"
    refresh_token: str | None = None
    if profile and profile.auth_kind == "device":
        auth_kind = "device"
        client_id = args.client_id or profile.client_id
        client_secret = ""  # device flow uses a public client
        refresh_token = profile.refresh_token_clear()
        if not refresh_token:
            raise ValueError(
                f"Profile '{profile.name}' is a device-flow profile but its refresh token "
                f"is missing from storage. Run `earl auth login --env {profile.environment}` "
                "to re-authenticate."
            )
    else:
        client_id = args.client_id or (
            profile.client_id if profile else os.getenv("EARL_CLIENT_ID", "")
        )
        client_secret = args.client_secret or (
            profile.secret_clear() if profile else os.getenv("EARL_CLIENT_SECRET", "")
        )

    organization = args.organization
    if organization is None:
        if profile:
            organization = profile.organization or ""
        else:
            organization = os.getenv("EARL_ORGANIZATION", "")

    environment = args.env or (
        profile.environment if profile else os.getenv("EARL_ENVIRONMENT", "prod")
    )
    if auth_kind == "m2m" and (not client_id or not client_secret):
        raise ValueError(
            "Missing credentials. Use --client-id/--client-secret, configure an auth profile, "
            "or run `earl auth login --env <env>` for an interactive Device-Flow login."
        )
    if auth_kind == "device" and not client_id:
        raise ValueError("Device-flow profile is missing a client_id; re-run `earl auth login`.")

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


def _handle_auth_login(args: argparse.Namespace, store: ConfigStore) -> None:
    """Run the OAuth Device Authorization Grant and persist the result.

    The Auth0 application is expected to be configured with
    ``Organization: Prompt For Credentials`` (see AGENTS.md / sdk/README.md),
    which means:

    - If the signing-in user belongs to exactly one organization, Auth0 picks
      it automatically and the access token arrives pre-scoped to that org.
    - If they belong to several (typical for internal users), Auth0 shows an
      "Select your organization" screen in the browser and the CLI just waits.

    In both cases we extract the ``org_id`` / ``org_name`` claims from the
    returned token and populate the profile accordingly, so the user never has
    to type an org ID.  Explicit ``--organization`` is still supported for
    cases where the admin wants to pre-select an org, and it takes precedence.
    """
    from earl_sdk.client import EnvironmentConfig
    from earl_sdk.device_flow import (
        DEFAULT_SCOPES,
        extract_org_info,
        poll_for_token,
        slugify_org,
        start_device_authorization,
    )

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
    scopes = args.scopes.split() if args.scopes else list(DEFAULT_SCOPES)
    if "offline_access" not in scopes:
        # Without offline_access Auth0 will not issue a refresh token, which
        # defeats the whole point of persisting the login.
        scopes.append("offline_access")

    # When --organization is omitted Auth0 drives the org picker in the
    # browser. When it's provided we pre-select that org (useful for admins
    # scripting multi-org setups).
    pre_selected_org = args.organization or ""

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
        },
    ):
        return

    # 1. Request a user/device code.
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
        # Print a single progress dot per poll; a periodic line keeps the user
        # aware without spamming.
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

    # 4. Pull org metadata out of the access token. The token was just minted
    # by Auth0 over HTTPS, so the claims are trustworthy for UX use.
    org_info = extract_org_info(token.access_token)
    resolved_org = pre_selected_org or org_info.org_id
    if not resolved_org:
        print(
            "\nwarning: the access token did not include an org_id claim. "
            "The orchestrator will likely reject requests. Either:\n"
            "  • ensure the Native Auth0 app has 'Type of Users: Business Users'\n"
            "    and 'Login Flow: Prompt For Credentials' so the browser shows\n"
            "    an organization picker, or\n"
            "  • re-run `earl auth login` with an explicit --organization.\n",
            file=sys.stderr,
        )

    # Name the profile deterministically from the resolved org so repeat
    # logins for the same (env, org) overwrite the existing profile rather
    # than leave stale entries behind.
    if args.name:
        profile_name = args.name
    else:
        slug = slugify_org(org_info.org_name, resolved_org)
        profile_name = f"device-{args.env}-{slug}" if slug else f"device-{args.env}"

    # 5. Persist the new profile + token.
    prof = AuthProfile.create_device(
        name=profile_name,
        client_id=client_id,
        clear_refresh_token=token.refresh_token,
        organization=resolved_org,
        environment=args.env,
    )
    store.upsert_profile(prof)
    if args.activate:
        store.set_active_profile(prof.name)

    # Seed the on-disk access-token cache so the next `earl` call skips Auth0.
    cache_key = auth_storage.token_cache_key(client_id, audience, resolved_org, domain, "device")
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
    org_label = org_info.org_name or resolved_org or "no organization"
    _say(
        args,
        f"Signed in to env={args.env}{who} as profile '{prof.name}' "
        f"→ {org_label}. Refresh token stored in {backend_note}.",
    )


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
            pipe = client.pipelines.update(args.pipeline_name, **kwargs)
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
                "message_preview": (message[:120] + "…") if len(message) > 120 else message,
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
