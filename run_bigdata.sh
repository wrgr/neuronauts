#!/usr/bin/env bash
# Relaunch loop for the checkpointed Track-B fetch. The agent proxy restarts (new port)
# kill in-flight network jobs; fetch_bigdata.py checkpoints every box / svmap chunk, so
# re-invoking resumes from disk. Loops until the final SideTable exists.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
n=0
until [ -f data/sidetable_big.npz ]; do
  n=$((n+1))
  echo "=== [run_bigdata] attempt $n  $(date -u +%H:%M:%S) ==="
  OMP_NUM_THREADS=4 python -u -m experiments.pcfg_synapse_partitions.fetch_bigdata \
      --n-boxes 27 --side-um 40 --out data/sidetable_big.npz || true
  [ -f data/sidetable_big.npz ] && break
  # Back off on repeated fast failures (e.g. CAVE 503) so we don't hammer the service;
  # cap at 60s. Cached boxes/svmap persist, so resuming loses nothing.
  back=$(( n < 12 ? 5*n : 60 ))
  echo "=== [run_bigdata] exited without sidetable; resuming in ${back}s (attempt $n) ==="
  sleep "$back"
done
echo "=== [run_bigdata] DONE: data/sidetable_big.npz present ==="
