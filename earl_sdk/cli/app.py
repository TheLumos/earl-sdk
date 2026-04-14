"""Command-line interface for Earl SDK.

The ``earl`` command provides a scriptable CLI surface on top of existing SDK
APIs and local interactive storage primitives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterable

from earl_sdk import DoctorApiConfig, EarlClient, SimulationStatus, __version__
from earl_sdk.interactive.storage.config_store import AuthProfile, ConfigStore, DoctorConfig
from earl_sdk.interactive.storage.run_store import LocalRun, RunStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earl",
        description="EARL command-line interface",
    )
    parser.add_argument("--profile", help="Use saved auth profile name from ~/.earl/config.json")
    parser.add_argument("--env", choices=["local", "dev", "test", "prod"], help="Environment override")
    parser.add_argument("--client-id", help="Auth0 client ID")
    parser.add_argument("--client-secret", help="Auth0 client secret")
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
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # auth
    auth = sub.add_parser("auth", help="Manage auth profiles and connectivity")
    auth_sub = auth.add_subparsers(dest="auth_cmd", required=True)
    auth_prof = auth_sub.add_parser("profile", help="Manage profiles")
    auth_prof_sub = auth_prof.add_subparsers(dest="profile_cmd", required=True)

    prof_add = auth_prof_sub.add_parser("add", help="Add a profile")
    prof_add.add_argument("--name", required=True)
    prof_add.add_argument("--client-id", required=True)
    prof_add.add_argument("--client-secret", required=True)
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

    # doctor
    doctor = sub.add_parser("doctor", help="Manage saved doctor endpoint configs")
    doctor_sub = doctor.add_subparsers(dest="doctor_cmd", required=True)
    doctor_add = doctor_sub.add_parser("add", help="Add doctor config")
    doctor_add.add_argument("--name", required=True)
    doctor_add.add_argument("--type", required=True, choices=["internal", "external", "client_driven"])
    doctor_add.add_argument("--api-url")
    doctor_add.add_argument("--api-key")
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
    v_list.add_argument("--type", choices=["all", "hard-gates", "scoring-dimensions"], default="all")

    # pipelines
    pipelines = sub.add_parser("pipelines", help="Manage pipelines")
    pipelines_sub = pipelines.add_subparsers(dest="pipelines_cmd", required=True)
    pl_list = pipelines_sub.add_parser("list", help="List pipelines")
    pl_list.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True)
    pl_get = pipelines_sub.add_parser("get", help="Get pipeline")
    pl_get.add_argument("pipeline_name")
    pl_create = pipelines_sub.add_parser("create", help="Create pipeline")
    _add_pipeline_common_args(pl_create, creating=True)
    pl_update = pipelines_sub.add_parser("update", help="Update pipeline")
    pl_update.add_argument("pipeline_name")
    _add_pipeline_common_args(pl_update, creating=False)
    pl_delete = pipelines_sub.add_parser("delete", help="Delete pipeline")
    pl_delete.add_argument("pipeline_name")
    pl_validate = pipelines_sub.add_parser("validate-doctor", help="Validate external doctor endpoint")
    pl_validate.add_argument("--api-url", required=True)
    pl_validate.add_argument("--api-key")
    pl_validate.add_argument("--timeout", type=float, default=10.0)

    # simulations
    sims = sub.add_parser("simulations", help="Manage simulations")
    sims_sub = sims.add_subparsers(dest="simulations_cmd", required=True)
    s_list = sims_sub.add_parser("list", help="List simulations")
    s_list.add_argument("--status", choices=[s.value for s in SimulationStatus], help="Status filter")
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
    sims_sub.add_parser("pending", help="List pending episodes (requires --simulation-id)").add_argument(
        "--simulation-id", required=True
    )
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
    s_respond = sims_sub.add_parser("respond", help="Submit doctor response for client-driven episode")
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
    parser.add_argument(
        "--validate-doctor", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--conversation-initiator", choices=["patient", "doctor"])
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--verifiers", choices=["lumos", "legacy"])


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    store = ConfigStore()
    runs = RunStore()

    if not args.command:
        parser.print_help()
        return

    try:
        _dispatch(args, store, runs)
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


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

    client_id = args.client_id or (profile.client_id if profile else os.getenv("EARL_CLIENT_ID", ""))
    client_secret = args.client_secret or (
        profile.secret_clear() if profile else os.getenv("EARL_CLIENT_SECRET", "")
    )
    organization = args.organization
    if organization is None:
        if profile:
            organization = profile.organization or ""
        else:
            organization = os.getenv("EARL_ORGANIZATION", "")

    environment = args.env or (profile.environment if profile else os.getenv("EARL_ENVIRONMENT", "prod"))
    if not client_id or not client_secret:
        raise ValueError(
            "Missing credentials. Use --client-id/--client-secret or configure an auth profile."
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
    )


def _handle_auth(args: argparse.Namespace, store: ConfigStore) -> None:
    if args.auth_cmd == "profile":
        if args.profile_cmd == "add":
            prof = AuthProfile(
                name=args.name,
                client_id=args.client_id,
                client_secret=AuthProfile.obfuscate(args.client_secret),
                organization=args.organization,
                environment=args.env,
            )
            store.upsert_profile(prof)
            print(f"Saved profile '{args.name}'")
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
            store.set_active_profile(args.name)
            print(f"Active profile: {args.name}")
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
            store.delete_profile(args.name)
            print(f"Deleted profile: {args.name}")
            return
    if args.auth_cmd == "test":
        client = _build_client(args, store)
        ok = client.test_connection()
        print("Connection OK" if ok else "Connection failed")
        return
    raise ValueError("Unsupported auth command")


def _handle_doctor(args: argparse.Namespace, store: ConfigStore) -> None:
    if args.doctor_cmd == "add":
        if args.type == "external" and not args.api_url:
            raise ValueError("--api-url is required for external doctor")
        cfg = DoctorConfig(
            name=args.name,
            type=args.type,
            api_url=args.api_url or "",
            api_key=DoctorConfig.obfuscate(args.api_key) if args.api_key else "",
            auth_type=args.auth_type,
        )
        store.upsert_doctor_config(cfg)
        print(f"Saved doctor config: {cfg.name}")
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
        store.delete_doctor_config(args.name)
        print(f"Deleted doctor config: {args.name}")
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
        _emit(args, [_to_dict(d) for d in client.dimensions.list(include_custom=args.include_custom)])
        return
    if args.dimensions_cmd == "get":
        _emit(args, _to_dict(client.dimensions.get(args.dimension_id)))
        return
    if args.dimensions_cmd == "create":
        created = client.dimensions.create(
            name=args.name,
            description=args.description,
            category=args.category,
            weight=args.weight,
        )
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
        client.pipelines.delete(args.pipeline_name)
        print(f"Deleted pipeline: {args.pipeline_name}")
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
        sim = client.simulations.create(
            pipeline_name=args.pipeline,
            num_episodes=args.num_episodes,
            parallel_count=args.parallel_count,
        )
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
            with open(args.message_file, "r", encoding="utf-8") as f:
                message = f.read()
        if not message:
            raise ValueError("Provide --message or --message-file")
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
        _print_rows(entries, headers=["simulation_id", "pipeline_name", "status", "average_score", "saved_at"])
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


def _handle_chat(args: argparse.Namespace, client: EarlClient, store: ConfigStore, runs: RunStore) -> None:
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
