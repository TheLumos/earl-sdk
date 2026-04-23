# CLI-driven integration tests

These tests exercise the **full stack** end-to-end — `earl` CLI → orchestrator
on Cloud Run → Lumos case-service → Firestore / Auth0 — against a real
deployment. Every test shells out to the installed `earl` binary so it
doubles as executable documentation: read a test to see the exact CLI
incantation.

Scope (matches the refactor plan):

| File | Covers |
|---|---|
| `test_01_auth.py` | unauth 401, `whoami`, `auth my-orgs`, cross-env audience rejection, refresh token reuse across subprocesses, `auth profile list` |
| `test_02_tenancy.py` | single `org_id` per token, `service-account list` scoped to active org, cross-org pipeline invisibility (requires a second profile) |
| `test_03_roles.py` | non-admin user denied on mutating endpoints: `service-account create`, `pipelines delete`; non-admin still gets read access (requires a non-admin profile) |
| `test_04_catalog.py` | `patients list / --limit / --offset / get`, `cases list / get`, `dimensions list / get`, `verifiers list` |
| `test_05_pipeline.py` | authoring: catalog case + extra hard gate + scoring dims + patient pin → `create / list / get / update`, plus `--max-turns` client-side validation |
| `test_06_simulation.py` | start → wait → get report; asserts **every** attached verifier produced a numeric score in the report. Also a cancel path. |
| `test_07_sa_lifecycle.py` | `service-account create` → mint M2M token directly at Auth0 → use M2M via CLI → `service-account list` contains it → `service-account revoke` → Auth0 rejects further token exchanges → SA removed from list |
| `test_08_org_admin.py` | org / member / invitation CRUD via CLI against a live tenant |
| `test_09_org_admin_cassettes.py` | Same surface as `test_08` but via the typed ``client.admin`` API, with pytest-recording cassettes so CI runs offline |

## Cassette-backed tests (offline-replayable)

`test_09_org_admin_cassettes.py` uses
[`pytest-recording`](https://pypi.org/project/pytest-recording/) / VCR to
record real Auth0 traffic once, then replay it deterministically in CI. No
credentials are needed for replay.

Install the recording-only dev deps:

```bash
pip install -r sdk/tests/integration/requirements-dev.txt
```

**Replay (default, no env needed):**

```bash
pytest sdk/tests/integration -k cassette -v
```

**Refresh cassettes against a live tenant:**

```bash
RECORD_CASSETTES=1 EARL_INTEGRATION_ENV=dev \
  pytest sdk/tests/integration -k cassette -v
```

See `cassettes/README.md` for scrubbing, commit, and review guidance.

## Prerequisites

1. `earl` is on `PATH` (`pip install -e sdk`).
2. You have an **EARL_Admin** or **EARL_Org_Admin** PKCE profile for the target
   env:
   ```bash
   earl login --env dev --organization org_XXXX   # or --env staging
   earl whoami                                     # sanity
   ```
3. The target env is fully deployed (Firestore + Auth0 mgmt secrets wired,
   service accounts feature flag on, Native app callback URLs include the
   loopback ports 8484–8499).

Optional, needed for specific tests:

- `EARL_INTEGRATION_NONADMIN_PROFILE` — a profile for a user **without**
  `EARL_Admin` / `EARL_Org_Admin` in the target env. Required for
  `test_03_roles.py`. Otherwise those tests skip.
- `EARL_INTEGRATION_SECOND_ORG_PROFILE` — a profile bound to a **different**
  Auth0 organization in the target env. Required for the cross-org
  isolation check in `test_02_tenancy.py::test_cross_org_pipeline_invisible`.

## Run

```bash
# Entire suite against dev
EARL_INTEGRATION_ENV=dev pytest sdk/tests/integration -v

# Entire suite against staging (same tests, different env)
EARL_INTEGRATION_ENV=staging pytest sdk/tests/integration -v

# Just the auth / tenancy / role layer (fast)
EARL_INTEGRATION_ENV=dev pytest sdk/tests/integration \
  -v -k "auth or tenancy or roles"

# Skip the slow simulation-end-to-end test (~2-3 min)
EARL_INTEGRATION_ENV=dev pytest sdk/tests/integration \
  -v --deselect sdk/tests/integration/test_06_simulation.py::test_simulation_end_to_end

# With both role + cross-org profiles configured
EARL_INTEGRATION_ENV=dev \
EARL_INTEGRATION_NONADMIN_PROFILE=pkce-dev-readonly \
EARL_INTEGRATION_SECOND_ORG_PROFILE=pkce-dev-other-org \
pytest sdk/tests/integration -v
```

Every CLI invocation prints its full command line before executing, e.g.:

```
$ earl --json --profile pkce-dev-5O2o4T7R whoami
$ earl --json --profile pkce-dev-5O2o4T7R cases list
$ EARL_CLIENT_ID=abc EARL_CLIENT_SECRET=*** EARL_ORG_ID=org_... earl --json cases list
```

so if a test fails, the last line of `pytest` output is a copy-pasteable
repro.

## Companion demo wrapper

A human-readable walkthrough of the same flow, runnable from a shell and
printing every step, is in `scripts/earl-integration-demo.sh`. Point it at
a target env and it runs the happy-path sequence linearly with verbose
commentary — useful for demos, onboarding new users, or sanity-checking a
fresh deployment.

## What this suite does **not** cover

- Upstream **case authoring** (creating a new case in the Lumos
  case-service). That's not an Earl API surface.
- Exhaustive dimension editor flows (create / update / test). The
  CRUD shape is already covered by `test_smoke_full.py`.
- Load / concurrency tests.
