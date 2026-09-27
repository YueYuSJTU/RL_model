#!/usr/bin/env bash
# Run the fixed goal-point battery for all benchmark opponents.
#
# Every opponent is shown the same designed set of fixed goal points and only
# its OWN flight is measured.  No external model is used as a measuring stick.
#
# Usage:
#   bash scripts/opponent_behavior/run_goalpoints_all.sh [repeats] [jobs]

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

REPEATS="${1:-2}"
JOBS="${2:-6}"
PY=/home/ubuntu/miniconda3/envs/js_gpu/bin/python
export PY
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/dev/shm/mplcfg
export HOME=/dev/shm/home
export TMPDIR=/dev/shm/tmp
export OMP_NUM_THREADS=2
mkdir -p /dev/shm/mplcfg /dev/shm/home /dev/shm/tmp
mkdir -p experiments/opponent_behavior/raw experiments/opponent_behavior/logs

echo "=== goal-point battery: repeats=${REPEATS}, ${JOBS} jobs ==="
for s in 1 2 3 11 16 17 23 616; do echo "$s"; done | xargs -P "$JOBS" -n 1 bash -c '
  s="$0"
  nice -n 10 "$PY" scripts/opponent_behavior/run_goalpoints.py \
      --subject "$s" --repeats '"$REPEATS"' \
      --out "experiments/opponent_behavior/raw/gp_${s}.json" \
      >> "experiments/opponent_behavior/logs/goalpoints.log" 2>&1'
echo "=== done ==="
