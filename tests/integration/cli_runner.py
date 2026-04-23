"""Subprocess wrapper that drives the ``earl`` CLI.

The goals of this helper are:

1. Run actual ``earl`` commands as subprocesses so tests exercise the full
   stack (argparse, profile resolution, HTTP client, backend, response
   formatting) rather than poking :class:`earl_sdk.EarlClient` directly.
2. Print every command before running it, so a human reading the test output
   sees the exact ``earl ...`` invocation and can copy-paste it. This makes
   the integration test double as living documentation.
3. Optionally parse ``--json`` output into a Python object for assertions.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class CliResult:
    """Result of a single ``earl`` CLI invocation."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def json(self) -> Any:
        """Parse stdout as JSON. Raises ``ValueError`` if stdout is not JSON."""
        stripped = self.stdout.strip()
        if not stripped:
            raise ValueError(
                f"expected JSON from `{self.cmdline}`, got empty stdout "
                f"(stderr={self.stderr!r})"
            )
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"failed to parse JSON from `{self.cmdline}`:\n"
                f"stdout: {self.stdout!r}\n"
                f"stderr: {self.stderr!r}\n"
                f"error: {e}"
            ) from e

    @property
    def cmdline(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


class CliRunner:
    """Run ``earl`` subprocess commands with shared state (env, profile).

    Every invocation prints the resolved command to stderr-ish (stdout here,
    since pytest captures it and surfaces it on failure) before executing.
    """

    @property
    def env(self) -> str:
        return self._env

    @property
    def profile(self) -> str | None:
        return self._profile

    def __init__(
        self,
        env: str,
        *,
        profile: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        json_global: bool = True,
    ):
        """
        Args:
            env: Earl environment label (``dev``, ``staging``, ...).
            profile: CLI profile name to pin via ``--profile``. If ``None``,
                the CLI falls back to the active profile in ``~/.earl/config.json``.
            extra_env: Extra environment variables to set for the subprocess
                (e.g. ``EARL_CLIENT_ID`` / ``EARL_CLIENT_SECRET`` for M2M).
            json_global: If True, every command gets ``--json`` so output is
                machine-parseable. Set ``False`` for commands that don't
                support ``--json`` (e.g. ``login``).
        """
        self._env = env
        self._profile = profile
        self._extra_env = dict(extra_env or {})
        self._json_global = json_global

    def with_profile(self, profile: str | None) -> "CliRunner":
        """Return a shallow copy bound to a different profile."""
        return CliRunner(
            env=self._env,
            profile=profile,
            extra_env=self._extra_env,
            json_global=self._json_global,
        )

    def with_env_vars(self, **kw: str) -> "CliRunner":
        """Return a shallow copy with additional env vars merged in."""
        merged = {**self._extra_env, **kw}
        return CliRunner(
            env=self._env,
            profile=self._profile,
            extra_env=merged,
            json_global=self._json_global,
        )

    def without_profile_and_env(self) -> "CliRunner":
        """Return a runner with no profile / no M2M env vars (simulates unauth)."""
        stripped = {
            k: v
            for k, v in self._extra_env.items()
            if not k.startswith("EARL_CLIENT") and k != "EARL_ORGANIZATION"
        }
        return CliRunner(
            env=self._env,
            profile=None,
            extra_env=stripped,
            json_global=self._json_global,
        )

    def run(
        self,
        *args: str,
        check: bool = True,
        json_output: bool | None = None,
        timeout: float = 60.0,
        extra_env: Mapping[str, str] | None = None,
    ) -> CliResult:
        """Invoke ``earl`` with ``args``. Asserts exit 0 unless ``check=False``."""
        use_json = self._json_global if json_output is None else json_output

        argv: list[str] = ["earl"]
        if use_json:
            argv.append("--json")
        if self._profile:
            argv += ["--profile", self._profile]
        argv += list(args)

        proc_env = os.environ.copy()
        # Strip any stray M2M env vars from the caller's shell so tests don't
        # silently pick up a client-credentials flow when they intended to use
        # the PKCE profile. Callers opt-in to M2M via ``with_env_vars(...)``
        # which re-populates these.
        for k in ("EARL_CLIENT_ID", "EARL_CLIENT_SECRET", "EARL_ORGANIZATION"):
            proc_env.pop(k, None)
        proc_env.update(self._extra_env)
        if extra_env:
            proc_env.update(extra_env)

        print(f"\n$ {_format_argv(argv, proc_env, self._extra_env)}")

        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
        )
        result = CliResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and not result.ok:
            raise AssertionError(
                f"`{result.cmdline}` exited {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        return result

    def run_expect_fail(
        self,
        *args: str,
        json_output: bool | None = None,
        timeout: float = 60.0,
        extra_env: Mapping[str, str] | None = None,
    ) -> CliResult:
        """Invoke ``earl`` expecting a non-zero exit code.

        Returns the :class:`CliResult`. Raises if the command unexpectedly
        succeeds.
        """
        result = self.run(
            *args,
            check=False,
            json_output=json_output,
            timeout=timeout,
            extra_env=extra_env,
        )
        if result.ok:
            raise AssertionError(
                f"`{result.cmdline}` was expected to fail but exited 0\n"
                f"stdout: {result.stdout}"
            )
        return result


def _format_argv(
    argv: Iterable[str], proc_env: Mapping[str, str], extra_env: Mapping[str, str]
) -> str:
    """Render ``argv`` as a copy-pasteable shell line, prefixing any
    subprocess-only env vars that the user would need to export themselves."""
    overridden = {
        k: v
        for k, v in extra_env.items()
        if os.environ.get(k) != v
    }
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in overridden.items())
    cmd = " ".join(shlex.quote(a) for a in argv)
    return f"{prefix} {cmd}".strip()
