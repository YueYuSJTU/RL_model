"""Run the fixed goal-point battery for one benchmark opponent.

The subject is placed in the opponent seat and shown a designed set of fixed
goal points (see goalpoint.py).  Only the subject's own flight is measured.

Usage:
  python scripts/opponent_behavior/run_goalpoints.py \
      --subject 11 --out experiments/opponent_behavior/raw/gp_11.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import OPPONENTS, load_policy, make_eval_env, write_json  # noqa: E402
from goalpoint import (GOAL_POINTS, configure_goalpoint_env,  # noqa: E402
                       goal_point_ned, rollout_goalpoint, summarize_goalpoint)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, choices=sorted(OPPONENTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    vec = make_eval_env(seed=args.seed)
    subject = load_policy(OPPONENTS[args.subject]["path"])

    per_point = {}
    print("[goalpoint] subject=%s  battery=%d points" % (args.subject, len(GOAL_POINTS)),
          flush=True)
    for pt in GOAL_POINTS:
        eps = rollout_goalpoint(vec, subject, pt, repeats=args.repeats)
        s = summarize_goalpoint(eps)
        s.update(name=pt["name"], bearing_deg=pt["bearing_deg"],
                 dist_ft=pt["dist_ft"], alt_ft=pt["alt_ft"],
                 goal_point_ned=list(goal_point_ned(pt)))
        per_point[pt["name"]] = s
        print("   %-22s point=%5.2f gun=%5.2f closure=%+8.0f dmin=%6.0f "
              "straight=%5.2f steps=%3.0f" % (
                  pt["name"], s["point_frac"]["mean"], s["gun_frac"]["mean"],
                  s["closure_mean_fps"]["mean"], s["dist_min_ft"]["mean"],
                  s["straightness"]["mean"], s["avg_steps"]), flush=True)
    vec.close()

    write_json(Path(args.out), dict(
        subject=args.subject,
        subject_path=OPPONENTS[args.subject]["path"],
        subject_family=OPPONENTS[args.subject]["family"],
        subject_reward_set=OPPONENTS[args.subject]["reward_set"],
        repeats=args.repeats, seed=args.seed, wall_s=time.time() - t0,
        goal_points=per_point,
    ))
    print("[goalpoint] %s done in %.0fs -> %s" % (args.subject, time.time() - t0, args.out),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
