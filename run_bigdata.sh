#!/usr/bin/env bash
# Good-citizen Track-B watcher. CAVE rejects flooding/recursive-loop traffic outright (503/429)
# rather than slowing it, so we do NOT hammer: while materialize is unhealthy we make ONE cheap
# unauthenticated curl probe every PROBE_S seconds, and only spin up the heavy checkpointed fetch
# once materialize answers 200. The fetch itself is per-box checkpointed, so a mid-run failure
# just drops us back to probing and resumes from the next uncached box. Stops when the SideTable
# exists.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
PROBE_S="${PROBE_S:-300}"
HEALTH="https://minnie.microns-daf.com/materialize/version"

while [ ! -f data/sidetable_big.npz ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "=== [run_bigdata] materialize healthy (200) $(date -u +%H:%M:%S); running one fetch pass ==="
    OMP_NUM_THREADS=4 python -u -m experiments.pcfg_synapse_partitions.fetch_bigdata \
        --n-boxes 27 --side-um 40 --out data/sidetable_big.npz || true
    [ -f data/sidetable_big.npz ] && break
    echo "=== [run_bigdata] pass ended without SideTable; re-probing in ${PROBE_S}s ==="
  else
    echo "=== [run_bigdata] materialize HTTP $code $(date -u +%H:%M:%S); waiting ${PROBE_S}s (no load) ==="
  fi
  sleep "$PROBE_S"
done
echo "=== [run_bigdata] DONE: data/sidetable_big.npz present ==="
