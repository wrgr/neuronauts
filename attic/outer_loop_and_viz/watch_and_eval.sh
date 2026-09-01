#!/bin/bash
# Watch for checkpoint files and evaluate each one as it appears.
# Runs threshold sweep {0.90, 0.95, 0.97, 0.99, 0.995} per checkpoint
# and appends a one-line summary to run_logs/ckpt_eval_summary.tsv.
#
# Usage:
#   bash scripts/watch_and_eval.sh &
#   # or: nohup bash scripts/watch_and_eval.sh > /tmp/watch_eval.log 2>&1 &
set -e
CACHE=data/boxes
OUTDIR=run_logs/ckpt_evals
SUMMARY=run_logs/ckpt_eval_summary.tsv
THRESHOLDS="0.90 0.95 0.97 0.99 0.995"
POLL_INTERVAL=20  # seconds

mkdir -p $OUTDIR
# Header (skip if file already exists)
if [ ! -f $SUMMARY ]; then
    printf "checkpoint\tt_best\tmean_F1_best\tmean_F1_t099\n" > $SUMMARY
fi

evaluated=""

eval_checkpoint() {
    local ckpt=$1
    local tag=$(basename $ckpt .pt)
    echo "[watch_eval] Evaluating $tag ..."

    best_mean=-1
    best_t=""
    t099_mean=""

    for t in $THRESHOLDS; do
        local logf=$OUTDIR/${tag}_t${t}.log
        python scripts/train.py evaluate \
            --cell-gnn-checkpoint "$ckpt" \
            --cache-dir $CACHE \
            --partition-threshold $t \
            --split test \
            > "$logf" 2>&1
        mean=$(grep -oP "mean=\K[0-9.]+" "$logf" | head -1)
        [ -z "$mean" ] && mean="0"
        echo "  t=$t  mean_F1=$mean"
        if python3 -c "import sys; sys.exit(0 if float('$mean') > float('$best_mean') else 1)" 2>/dev/null; then
            best_mean=$mean
            best_t=$t
        fi
        [ "$t" = "0.99" ] && t099_mean=$mean
    done

    printf "%s\t%s\t%s\t%s\n" "$tag" "$best_t" "$best_mean" "$t099_mean" >> $SUMMARY
    echo "[watch_eval] $tag  best: t=$best_t F1=$best_mean  t=0.99: F1=$t099_mean"
}

echo "[watch_eval] Started. Watching models/ for checkpoint files every ${POLL_INTERVAL}s"
echo "[watch_eval] Summary → $SUMMARY"

while true; do
    for ckpt in models/cell_gnn_path16_ep*.pt models/cell_gnn_path16_no_grammar_ep*.pt; do
        [ -f "$ckpt" ] || continue
        # Skip if already evaluated
        echo "$evaluated" | grep -qF "$ckpt" && continue
        eval_checkpoint "$ckpt"
        evaluated="$evaluated $ckpt"
    done

    # Exit once both final models exist and are evaluated
    done16=false
    done_ng=false
    echo "$evaluated" | grep -qF "models/cell_gnn_path16_ep50.pt" && done16=true
    echo "$evaluated" | grep -qF "models/cell_gnn_path16_no_grammar_ep50.pt" && done_ng=true
    if $done16 && $done_ng; then
        echo "[watch_eval] Both final checkpoints evaluated. Done."
        echo ""
        echo "=== Final summary ==="
        cat $SUMMARY
        break
    fi

    sleep $POLL_INTERVAL
done
