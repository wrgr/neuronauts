#!/usr/bin/env bash
# Track-B fetch: ONE resumable pass, driven by a cron every 10 min. Background loops kept getting
# reaped between turns and the agent proxy changes port on restart -- so no persistent process and
# no baked-in proxy. Each invocation: re-derive the LIVE proxy port from /root/.ccr/README.md,
# then run fetch_bigdata once (it skips already-cached boxes, so it just advances). flock prevents
# two overlapping cron ticks from racing on the per-box cache writes.
set -u
cd "$(dirname "$0")"
exec 9>data/bigdata/.fetch.lock 2>/dev/null || { mkdir -p data/bigdata; exec 9>data/bigdata/.fetch.lock; }
flock -n 9 || { echo "[run_bigdata] another pass holds the lock; skip"; exit 0; }
source .venv/bin/activate 2>/dev/null || true

[ -f data/bigdata/DONE ] && { echo "[run_bigdata] DONE sentinel present; nothing to do"; exit 0; }

port=$(grep -oE '127\.0\.0\.1:[0-9]+' /root/.ccr/README.md 2>/dev/null | head -1)
[ -n "$port" ] && export HTTPS_PROXY="http://$port" https_proxy="http://$port"
code=$(curl -s -o /dev/null -w "%{http_code}" "https://minnie.microns-daf.com/materialize/version" 2>/dev/null)
echo "[run_bigdata] $(date -u +%H:%M:%S) proxy=$port materialize=$code"
[ "$code" = "200" ] || { echo "[run_bigdata] materialize down; skip this pass (no load)"; exit 0; }

OMP_NUM_THREADS=4 python -u -m experiments.pcfg_synapse_partitions.fetch_bigdata \
    --n-boxes 27 --side-um 40 --out data/sidetable_big.npz
