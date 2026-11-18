#!/usr/bin/env bash
# Re-render a folder of rollout pickles with several Isaac Sim processes in parallel.
#
# Usage (from anywhere; runs from the repo root):
#   scripts/rerender_batch.sh <load_dir> <save_dir> <n_workers> [extra isaac_lab_rerender args ...]
#
# Example:
#   scripts/rerender_batch.sh \
#     data/raw/diffik/sim/one_leg/rollout/med/<run>/success \
#     data/raw/diffik/sim/one_leg_render_fr3/rollout/med/<run>/success \
#     4 --sub-steps 3 --furniture one_leg --num-parts 5 --robot fr3 --save-size 240x320 \
#     -dr rand --part-random full --table-random base
#
# - Episode i (in sorted file order, as isaac_lab_rerender.py picks it with -i) goes to worker i % n_workers.
# - Episodes whose output already exists are skipped, so rerunning the same command resumes.
# - Stop: Ctrl+C (kills the running episodes; they are redone next time), or `touch <logs>/STOP` to let the
#   running episodes finish first.
# - Logs: <save_dir>/../rerender_logs/<i>.log per episode, worker_<k>.log per worker, failures.txt.
#   The extra args must end with the `rand ...` subcommand if you use it (argparse requires it last).
#
# Environment: PY (default ~/miniconda3/envs/isaaclab/bin/python), START_STAGGER seconds between worker
# starts (default 20), RETRIES extra attempts for a failed episode (default 2), RETRY_SLEEP seconds before
# a retry (default 10).
#
# Workers: on a 61 GB / 16-core machine use 4 (at most 5). Each Isaac Sim process peaks at ~8 GB RAM;
# 8 workers ran out of memory and the OOM killer took down the desktop session.

set -uo pipefail

if [ $# -lt 3 ]; then
  sed -n 2,25p "$0"
  exit 1
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

LOAD=$1
SAVE=$2
N=$3
shift 3
EXTRA=("$@")

PY=${PY:-$HOME/miniconda3/envs/isaaclab/bin/python}
START_STAGGER=${START_STAGGER:-20}
RETRIES=${RETRIES:-2}
RETRY_SLEEP=${RETRY_SLEEP:-10}
LOGS=$(dirname "$SAVE")/rerender_logs
LIST=$LOGS/files.txt
STOP=$LOGS/STOP

mkdir -p "$SAVE" "$LOGS"
rm -f "$STOP"

# Same file list and order as isaac_lab_rerender.py (sorted Paths of *.pkl and *.pkl.xz under LOAD)
"$PY" - "$LOAD" > "$LIST" <<'EOF'
import sys
from pathlib import Path
d = Path(sys.argv[1])
for f in sorted(list(d.rglob("*.pkl")) + list(d.rglob("*.pkl.xz"))):
    print(f.name)
EOF
TOTAL=$(wc -l < "$LIST")
DONE=$(while IFS= read -r name; do [ -f "$SAVE/${name%.xz}" ] && echo; done < "$LIST" | wc -l)
echo "$TOTAL episodes in $LOAD, $DONE already rendered in $SAVE, $N workers"
echo "logs: $LOGS   stop gracefully: touch $STOP   stop now: Ctrl+C"

export PYTHONPATH=$REPO:$REPO/furniture-bench
export RARL_SOURCE_DIR=$REPO/src
export PYTHONUNBUFFERED=1

worker() {
  local k=$1 i=0 name out t0 attempt
  sleep $((k * START_STAGGER))  # don't start all Isaac Sim instances at once
  while IFS= read -r name; do
    if (( i % N == k )); then
      [ -f "$STOP" ] && { echo "[w$k] stop file found"; break; }
      out="$SAVE/${name%.xz}"
      attempt=0
      [ -f "$out" ] || : > "$LOGS/$i.log"
      while [ ! -f "$out" ] && [ ! -f "$STOP" ]; do
        attempt=$((attempt + 1))
        echo "[w$k] $(date +%T) start i=$i $name (attempt $attempt)"
        echo "===== attempt $attempt =====" >> "$LOGS/$i.log"
        t0=$SECONDS
        if "$PY" -m src.sim2real.isaac_lab_rerender --headless --enable_cameras -i "$i" \
            --load-dir "$LOAD" --save --save-dir "$SAVE" "${EXTRA[@]}" \
            < /dev/null >> "$LOGS/$i.log" 2>&1 && [ -f "$out" ]; then
          echo "[w$k] $(date +%T) done  i=$i in $((SECONDS - t0)) s"
        elif (( attempt > RETRIES )); then
          echo "[w$k] $(date +%T) FAILED i=$i $name after $attempt attempts (see $LOGS/$i.log)" | tee -a "$LOGS/failures.txt"
          break
        else
          # Isaac Sim occasionally segfaults while loading extensions at startup; a retry usually works
          echo "[w$k] $(date +%T) failed i=$i (attempt $attempt), retrying in $RETRY_SLEEP s"
          sleep "$RETRY_SLEEP"
        fi
      done
    fi
    i=$((i + 1))
  done < "$LIST"
}

WPIDS=()
stop_all() {
  echo "stopping workers..."
  touch "$STOP"
  for p in "${WPIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null; done
  sleep 5
  # Isaac Sim often ignores SIGINT/SIGTERM
  for p in "${WPIDS[@]}"; do pkill -KILL -P "$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
  echo "stopped; rerun the same command to resume"
  exit 130
}
trap stop_all INT TERM

for ((k = 0; k < N; k++)); do
  worker "$k" > >(tee -a "$LOGS/worker_$k.log") 2>&1 &
  WPIDS+=($!)
done
wait

DONE=$(while IFS= read -r name; do [ -f "$SAVE/${name%.xz}" ] && echo; done < "$LIST" | wc -l)
echo "finished: $DONE / $TOTAL rendered"
[ -f "$LOGS/failures.txt" ] && echo "failures listed in $LOGS/failures.txt"
