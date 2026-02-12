#!/usr/bin/env python3
"""
List recent simulation IDs from the EARL API (e.g. to recover IDs after an early exit).

Use when test_doctors.py --repeat N exited before printing all EARL_SIMULATION_ID lines.
Simulations are listed by most recent first; filter by pipeline name prefix if needed.

Usage:
  python3 list_recent_simulations.py --env test --limit 50 \\
      --client-id "..." --client-secret "..."

  # Only IDs (one per line), e.g. for scripts:
  python3 list_recent_simulations.py --env test --limit 50 --ids-only ...

  # Filter by pipeline name prefix (e.g. from sdk-test-external-1738...):
  python3 list_recent_simulations.py --env test --limit 100 --pipeline-prefix "sdk-test-external"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient


def main():
    parser = argparse.ArgumentParser(description="List recent simulation IDs from EARL API")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test")
    parser.add_argument("--limit", type=int, default=100, help="Max simulations to fetch (default 100)")
    parser.add_argument("--pipeline-prefix", type=str, default=None,
                        help="Filter by pipeline_name start (e.g. sdk-test-external)")
    parser.add_argument("--ids-only", action="store_true", help="Print only simulation IDs, one per line")
    parser.add_argument("--client-id", type=str)
    parser.add_argument("--client-secret", type=str)
    parser.add_argument("--organization", type=str)
    args = parser.parse_args()

    client_id = args.client_id or os.environ.get("EARL_CLIENT_ID")
    client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Need --client-id and --client-secret (or EARL_CLIENT_ID, EARL_CLIENT_SECRET)", file=sys.stderr)
        sys.exit(1)

    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=args.organization or os.environ.get("EARL_ORGANIZATION", ""),
        environment=args.env,
    )

    # API returns simulations; SDK list() returns Simulation objects (id, status, pipeline_name, etc.)
    sims = client.simulations.list(limit=args.limit)
    if args.pipeline_prefix:
        sims = [s for s in sims if getattr(s, "pipeline_name", None) and str(s.pipeline_name).startswith(args.pipeline_prefix)]

    if args.ids_only:
        for s in sims:
            print(s.id)
        return

    print(f"Found {len(sims)} simulation(s) (env={args.env}, limit={args.limit})")
    if args.pipeline_prefix:
        print(f"Filtered by pipeline_name starting with: {args.pipeline_prefix!r}")
    print()
    for s in sims:
        pname = getattr(s, "pipeline_name", "") or ""
        status = getattr(s, "status", None)
        status_str = str(status.value) if status else "?"
        print(f"  {s.id}  pipeline={pname}  status={status_str}")


if __name__ == "__main__":
    main()
