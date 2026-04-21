"""Tests for ``earl schema`` and agent-friendly global CLI flags.

These tests exercise the CLI entry point in-process (no subprocess) so they
stay fast and don't require network or Auth0 credentials. The ``--dry-run``
tests in particular verify that no EarlClient is constructed, which is the
contract agents rely on.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import pytest

from earl_sdk.cli import app as cli_app
from earl_sdk.cli import schema as schema_mod

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(argv: list[str]) -> tuple[str, str, int]:
    """Invoke the CLI and capture (stdout, stderr, exit_code)."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            cli_app.main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    return stdout.getvalue(), stderr.getvalue(), code


# ── schema command ───────────────────────────────────────────────────────────


def test_schema_json_has_top_level_commands():
    out, _, code = _run(["schema", "--format", "json"])
    assert code == 0
    spec = json.loads(out)
    assert spec["name"] == "earl"
    assert "commands" in spec
    # Every non-trivial consumer-facing command should be discoverable.
    for cmd in ("auth", "pipelines", "simulations", "schema", "doctor"):
        assert cmd in spec["commands"], f"missing top-level command: {cmd}"


def test_schema_json_describes_nested_subcommands():
    out, _, code = _run(["schema", "--format", "json"])
    assert code == 0
    spec = json.loads(out)

    profile_add = spec["commands"]["auth"]["subcommands"]["profile"]["subcommands"]["add"]
    flags = {a["flags"][0] for a in profile_add["arguments"] if a["flags"]}
    assert {"--name", "--client-id", "--env"}.issubset(flags)


def test_schema_json_has_examples_from_epilog():
    out, _, code = _run(["schema", "--format", "json"])
    assert code == 0
    spec = json.loads(out)
    sch_node = spec["commands"]["schema"]
    assert any("earl schema" in ex for ex in sch_node["examples"])


def test_schema_markdown_renders_headings():
    out, _, code = _run(["schema", "--format", "markdown"])
    assert code == 0
    assert out.startswith("# `earl` CLI reference")
    assert "## Global flags" in out
    # H2 for a top-level command
    assert "## `earl auth`" in out
    # Nested subcommand heading
    assert "#### `earl auth profile add`" in out


def test_schema_out_writes_file(tmp_path):
    dest = tmp_path / "spec.json"
    out, err, code = _run(["schema", "--format", "json", "--out", str(dest)])
    assert code == 0
    assert dest.exists()
    json.loads(dest.read_text())  # must be valid JSON
    assert "wrote schema" in err
    assert out == ""


def test_schema_dict_stable_structure():
    parser = cli_app._parser()
    spec = schema_mod.schema_dict(parser, version="1.2.3")
    assert spec["version"] == "1.2.3"
    assert isinstance(spec["global_arguments"], list)
    for arg in spec["global_arguments"]:
        assert {"name", "flags", "required", "type", "help"}.issubset(arg.keys())


# ── Global flag plumbing ─────────────────────────────────────────────────────


def test_json_flag_aliases_output_json():
    out, _, code = _run(["--json", "schema", "--format", "json"])
    assert code == 0
    # Still valid JSON, regardless of whether the subcommand explicitly respects
    # --output. This just checks the global flag doesn't blow up parsing.
    json.loads(out)


def test_debug_flag_configures_logger():
    import logging

    _, _, code = _run(["--debug", "schema", "--format", "json"])
    assert code == 0
    assert logging.getLogger("earl_sdk").level == logging.DEBUG
    # Reset so other tests aren't affected.
    logging.getLogger("earl_sdk").setLevel(logging.WARNING)


# ── --dry-run for mutating commands ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_earl_home(tmp_path, monkeypatch):
    """Keep tests away from the developer's real ~/.earl dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")
    # The dirs are module-level constants pinned at import time from
    # ``Path.home()``; override them explicitly for isolation.
    import earl_sdk.auth_storage as storage

    earl_dir = tmp_path / ".earl"
    monkeypatch.setattr(storage, "EARL_DIR", earl_dir)
    monkeypatch.setattr(storage, "TOKEN_CACHE_DIR", earl_dir / "tokens")
    yield


def test_dry_run_pipelines_create_emits_payload_without_credentials():
    out, _, code = _run(
        [
            "--dry-run",
            "pipelines",
            "create",
            "--name",
            "test-pipeline",
            "--case-id",
            "case-xyz",
            "--doctor",
            "external",
            "--doctor-api-url",
            "https://example.com/chat",
            "--doctor-api-key",
            "s3cret",
        ]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["action"] == "pipelines.create"
    assert payload["payload"]["name"] == "test-pipeline"
    # api_key must be redacted in --dry-run output
    assert payload["payload"]["doctor_config"]["api_key"] == "<redacted>"


def test_dry_run_simulations_start_skips_api():
    # If the client were built we'd see a credentials error instead of JSON.
    with patch.object(cli_app, "_build_client") as m:
        out, _, code = _run(
            [
                "--dry-run",
                "simulations",
                "start",
                "--pipeline",
                "my-eval",
                "--num-episodes",
                "3",
            ]
        )
    assert code == 0
    m.assert_not_called()
    payload = json.loads(out)
    assert payload["action"] == "simulations.start"
    assert payload["payload"]["num_episodes"] == 3


def test_dry_run_dimensions_create_emits_payload():
    out, _, code = _run(
        [
            "--dry-run",
            "dimensions",
            "create",
            "--name",
            "my-dim",
            "--description",
            "custom",
        ]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "dimensions.create"
    assert payload["payload"]["name"] == "my-dim"


def test_dry_run_auth_profile_add_redacts_secret():
    out, _, code = _run(
        [
            "--dry-run",
            "auth",
            "profile",
            "add",
            "--name",
            "testprof",
            "--client-id",
            "abc",
            "--env",
            "test",
            "--client-secret",
            "super-secret",
        ]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "auth.profile.add"
    assert payload["payload"]["client_secret"] == "<redacted>"


def test_dry_run_simulations_respond_requires_message():
    _, err, code = _run(
        [
            "--dry-run",
            "simulations",
            "respond",
            "sim-1",
            "ep-1",
        ]
    )
    # Missing message → error (exit 1), no payload emitted.
    assert code == 1
    assert "--message" in err or "message" in err.lower()


def test_dry_run_simulations_respond_truncates_preview():
    long_msg = "x" * 500
    out, _, code = _run(
        [
            "--dry-run",
            "simulations",
            "respond",
            "sim-1",
            "ep-1",
            "--message",
            long_msg,
        ]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["payload"]["message_length"] == 500
    assert len(payload["payload"]["message_preview"]) <= 121  # 120 + ellipsis


# ── Argument coverage for schema ─────────────────────────────────────────────


def test_schema_reports_correct_argument_types():
    parser = cli_app._parser()
    spec = schema_mod.schema_dict(parser)
    sim_start = spec["commands"]["simulations"]["subcommands"]["start"]
    types = {a["name"]: a["type"] for a in sim_start["arguments"]}
    assert types["num_episodes"] == "integer"
    assert types["parallel_count"] == "integer"
    assert types["pipeline"] == "string"


def test_schema_preserves_choices_for_env_flag():
    parser = cli_app._parser()
    spec = schema_mod.schema_dict(parser)
    env_arg = next(a for a in spec["global_arguments"] if a["name"] == "env")
    assert set(env_arg["choices"]) == {"local", "dev", "test", "prod"}
