#!/bin/bash
# Evaluate Option-2 path-embedding models.
#
# For each new checkpoint:
#   1. Sweep thresholds {0.90, 0.95, 0.97, 0.99, 0.995} to find the model's
#      own best-mean threshold.
#   2. Also evaluate at t=0.99 for apples-to-apples comparison with the
#      ablation table in docs/ablation_results.md.
#
# Run after training completes:
#   bash scripts/eval_path_models.sh
set -e
THRESHOLDS="0.90 0.95 0.97 0.99 0.995"
CACHE=data/boxes
OUTDIR=run_logs/eval_path_models
mkdir -p $OUTDIR

sweep_and_report() {
    local ckpt=$1
    local tag=$2
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $ckpt not found"
        return
    fi

    echo ""
    echo "======================================================"
    echo "  Model: $tag"
    echo "======================================================"

    best_mean=-1
    best_t=""
    for t in $THRESHOLDS; do
        local logf=$OUTDIR/${tag}_t${t}.log
        python scripts/train.py evaluate \
            --cell-gnn-checkpoint $ckpt \
            --cache-dir $CACHE \
            --partition-threshold $t \
            --split test \
            2>&1 | tee $logf | grep -E "Test F1:|mean=" | tail -1
        mean=$(grep "mean=" $logf | grep -oP "mean=\K[0-9.]+" | head -1)
        echo "  t=$t  mean_F1=$mean"
        if [ -n "$mean" ] && python3 -c "import sys; sys.exit(0 if float('$mean') > float('$best_mean') else 1)" 2>/dev/null; then
            best_mean=$mean
            best_t=$t
        fi
    done

    echo ""
    echo "  Best threshold for $tag: t=$best_t  mean_F1=$best_mean"
    echo "  (Fixed comparison at t=0.99 is above)"
}

# Option-2: 6 scalars + path_emb_dim=16
sweep_and_report models/cell_gnn_path16.pt path16

# Option-2 grammar-ablated: 5 scalars + path_emb_dim=16
sweep_and_report models/cell_gnn_path16_no_grammar.pt path16_no_grammar

# Reference baseline (for sanity check)
sweep_and_report models/cell_gnn_seg.pt baseline_seg

echo ""
echo "======================================================"
echo "  Fixed t=0.99 comparison (append to ablation table)"
echo "======================================================"
for ckpt_tag in \
    "models/cell_gnn_path16.pt:path16_6feat" \
    "models/cell_gnn_path16_no_grammar.pt:path16_5feat_no_grammar" \
    "models/cell_gnn_seg.pt:baseline_seg_t099"; do
    ckpt="${ckpt_tag%%:*}"
    tag="${ckpt_tag##*:}"
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $ckpt not found"
        continue
    fi
    logf=$OUTDIR/${tag}_t099.log
    python scripts/train.py evaluate \
        --cell-gnn-checkpoint $ckpt \
        --cache-dir $CACHE \
        --partition-threshold 0.99 \
        --split test \
        2>&1 | tee $logf | grep "Test F1:" | head -1
done

echo ""
echo "Done. Full logs in $OUTDIR/"
