# Cassettes for Auth0-backed integration tests

This folder stores recorded HTTP exchanges (pytest-recording / VCR format) for
tests marked `@pytest.mark.vcr`. In replay mode — the default — the test suite
never touches the live Auth0 tenant; it reads cassettes from this folder
instead. That lets CI run offline on every PR without a secret.

## When to refresh

Refresh cassettes when:

* The orchestrator's Auth0 Management API interactions change shape
  (new endpoint, new body field, new header).
* The Auth0 tenant config changes (scopes, connection, org list).
* A cassette-backed test fails with a "no cassette entry" error.

## How to refresh

1. Make sure you have a live **dev** or **staging** Auth0 tenant and the
   relevant `earl` PKCE profile (`earl login --env dev --organization …`).
2. Install the dev deps:

   ```bash
   pip install -r sdk/tests/integration/requirements-dev.txt
   ```
3. Run the cassette-backed tests in record mode:

   ```bash
   RECORD_CASSETTES=1 EARL_INTEGRATION_ENV=dev \
     pytest sdk/tests/integration -k cassette -v
   ```
4. Inspect the diff — cassette files live next to this README. Scrub any
   accidental tokens in the `Authorization:` header (the fixture does this
   automatically, but double-check).
5. Commit the new cassettes alongside the test change. PR reviewers can
   run the replay mode locally without any credentials.

## Why pytest-recording

`pytest-recording` wraps VCR.py with saner defaults (automatic `@pytest.mark.vcr`
marker, per-test cassette paths, opt-in `--record-mode=rewrite`). We prefer
it over the older `pytest-vcr` package, which has been unmaintained since
2020.
