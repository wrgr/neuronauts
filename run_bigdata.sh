#!/usr/bin/env bash
# Track-B fetch: self-healing loop. Runs in bash (no model cost). The agent proxy restarts and
# CHANGES PORT periodically; a baked-in HTTPS_PROXY then points at a dead port -- that stale-port
# assumption is what caused the earlier debug loop. So each iteration re-derives the LIVE port
# from /root/.ccr/README.md (the proxy rewrites that file on restart). Per-box checkpointed:
# a CAVE 500/503 or a port blip just means a box isn't cached and the next iteration retries it.
# A hard container restart can still kill this process; the supervising cron relaunches it then.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
mkdir -p data/bigdata
echo $$ > data/bigdata/watcher.pid

while [ ! -f data/sidetable_big.npz ]; do
  port=$(grep -oE '127\.0\.0\.1:[0-9]+' /root/.ccr/README.md 2>/dev/null | head -1)
  [ -n "$port" ] && export HTTPS_PROXY="http://$port" https_proxy="http://$port"
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://minnie.microns-daf.com/materialize/version" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "=== [run_bigdata] $(date -u +%H:%M:%S) proxy=$port materialize=200; fetch pass ==="
    OMP_NUM_THREADS=4 python -u -m experiments.pcfg_synapse_partitions.fetch_bigdata \
        --n-boxes 27 --side-um 40 --out data/sidetable_big.npz || true
  else
    echo "=== [run_bigdata] $(date -u +%H:%M:%S) proxy=$port materialize=$code; wait (no load) ==="
  fi
  [ -f data/sidetable_big.npz ] && break
  sleep 180
done
echo "=== [run_bigdata] DONE: sidetable_big.npz present ==="
rm -f data/bigdata/watcher.pid
