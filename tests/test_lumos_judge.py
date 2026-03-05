#!/usr/bin/env python3
"""
Earl SDK — Case-based evaluation test.

Demonstrates the recommended workflow:
  1. List available cases
  2. Create a pipeline from a case (auto-includes verifiers + gates + dims)
  3. Run simulation with internal doctor
  4. Get structured report with hard gates, scoring dimensions, case verifiers

Usage:
    python3 test_lumos_judge.py --env test
    python3 test_lumos_judge.py --env test --max-turns 10 --timeout 900
"""

import os
import sys
import argparse
import json
import time
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.exceptions import EarlError


def get_credentials(args):
    """Load credentials from CLI args, env vars, or ~/.earl/config.json."""
    client_id = args.client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = args.organization or os.environ.get("EARL_ORGANIZATION", "")
    if not client_id or not client_secret:
        config_path = os.path.expanduser("~/.earl/config.json")
        if os.path.exists(config_path):
            try:
                cfg = json.load(open(config_path))
                profile = cfg.get("profiles", {}).get(cfg.get("active_profile", "test"), {})
                if profile.get("client_id") and profile.get("client_secret"):
                    client_id = profile["client_id"]
                    raw = profile["client_secret"]
                    try:
                        client_secret = base64.b64decode(raw).decode()
                    except Exception:
                        client_secret = raw
                    organization = organization or profile.get("organization", "")
                    print(f"  Loaded credentials from ~/.earl/config.json")
            except Exception as e:
                print(f"  Warning: could not read ~/.earl/config.json: {e}")
    if not client_id or not client_secret:
        print("ERROR: Missing credentials.")
        print("  Use --client-id/--client-secret, env vars, or ~/.earl/config.json")
        sys.exit(1)
    return client_id, client_secret, organization


def main():
    parser = argparse.ArgumentParser(description="Earl SDK — Case-based evaluation test")
    parser.add_argument("--env", default="test", choices=["test", "prod", "local"])
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    parser.add_argument("--organization", default=None)
    parser.add_argument("--case", default="carla-hypertension-yasmin", help="Case ID to use")
    parser.add_argument("--max-turns", type=int, default=4, help="Max conversation turns")
    parser.add_argument("--timeout", type=int, default=900, help="Max wait seconds")
    args = parser.parse_args()

    client_id, client_secret, organization = get_credentials(args)

    print("=" * 60)
    print("Earl SDK — Case-Based Evaluation Test")
    print("=" * 60)

    client = EarlClient(
        client_id=client_id, client_secret=client_secret,
        environment=args.env, organization=organization,
    )
    print(f"  Environment: {args.env}")
    print(f"  API: {client.api_url}")

    # 1. List cases
    print(f"\n[1] Available cases")
    cases = client.cases.list()
    if not cases:
        print("  ERROR: No cases available")
        sys.exit(1)
    for c in cases:
        print(f"  • {c['case_id']}: {c['name']}")
        print(f"    {c.get('case_verifiers', 0)} case verifiers, "
              f"{c.get('hard_gates', 0)} hard gates, "
              f"{c.get('scoring_dimensions', 0)} scoring dims")

    # 2. Get case detail
    print(f"\n[2] Case: {args.case}")
    detail = client.cases.get(args.case)
    totals = detail.get("totals", {})
    print(f"  Name:     {detail['name']}")
    print(f"  Patient:  {detail['patient_id']}")
    print(f"  Verifiers: {totals.get('case_verifiers', 0)} case verifiers, "
          f"{totals.get('hard_gates', 0)} hard gates, "
          f"{totals.get('scoring_dimensions', 0)} scoring dims")

    # 3. Create pipeline
    pipeline_name = f"case-test-{int(time.time())}"
    print(f"\n[3] Creating pipeline: {pipeline_name}")
    pipeline = client.pipelines.create(
        name=pipeline_name,
        case_id=args.case,
        max_turns=args.max_turns,
        verifiers="lumos",
    )
    print(f"  Created: {pipeline.name}")

    # 4. Run simulation
    print(f"\n[4] Starting simulation...")
    sim = client.simulations.create(pipeline_name=pipeline_name, num_episodes=1)
    print(f"  SIM ID: {sim.id}")

    # 5. Wait
    print(f"\n[5] Waiting (timeout={args.timeout}s)...")
    start = time.time()
    last = None
    while time.time() - start < args.timeout:
        sim = client.simulations.get(sim.id)
        s = sim.status.value if hasattr(sim.status, "value") else str(sim.status)
        if s != last:
            print(f"  [{int(time.time() - start)}s] {s}")
            last = s
        if s in ("completed", "failed"):
            break
        time.sleep(15)

    # 6. Report
    print(f"\n[6] Report")
    report = client.simulations.get_report(sim.id)
    status = report.get("status", "unknown")
    score = report.get("summary", {}).get("average_score")
    print(f"  Status: {status}  Score: {score}")

    ok = True
    for ep in report.get("episodes", []):
        print(f"\n  Episode {ep.get('episode_number', '?')}: "
              f"score={ep.get('total_score')}, "
              f"turns={ep.get('dialogue_turns')}, "
              f"status={ep.get('status')}")

        # Hard gates
        hg = ep.get("hard_gates", [])
        if hg:
            passed = sum(1 for g in hg if g.get("passed"))
            print(f"\n  Hard Gates: {passed}/{len(hg)} passed")
            for g in hg:
                tag = "PASS" if g.get("passed") else "FAIL"
                print(f"    [{tag}] {g['id']}: {g.get('score')}/{g.get('max_score')}")
                if g.get("rationale"):
                    print(f"          {g['rationale'][:100]}")

        # Scoring dimensions
        sd = ep.get("scoring_dimensions", [])
        if sd:
            activated = [d for d in sd if d.get("activated", True)]
            skipped = len(sd) - len(activated)
            avg = sum(d.get("score", 0) for d in activated) / len(activated) if activated else 0
            print(f"\n  Scoring Dimensions: {avg:.1f}/4 avg ({len(activated)} evaluated, {skipped} skipped)")
            for d in activated:
                print(f"    {d.get('score', '?')}/{d.get('max_score', 4)}  {d['id']}")
                if d.get("rationale"):
                    print(f"          {d['rationale'][:100]}")

        # Case verifiers
        cv = ep.get("case_verifiers", [])
        if cv:
            triggered = [v for v in cv if v.get("triggered")]
            total_pts = sum(v.get("points_awarded", 0) for v in cv)
            print(f"\n  Case Verifiers: {len(triggered)}/{len(cv)} triggered ({total_pts:+d} pts)")
            for v in triggered:
                pts = v.get("points_awarded", 0)
                print(f"    [{pts:+d}] {v['id']}")
                if v.get("rationale"):
                    print(f"          {v['rationale'][:100]}")

        if ep.get("status") == "failed":
            ok = False
            print(f"\n  ERROR: {ep.get('error', 'unknown')[:200]}")

    print("\n" + "=" * 60)
    if ok and status == "completed":
        print("PASSED")
    else:
        print("FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
