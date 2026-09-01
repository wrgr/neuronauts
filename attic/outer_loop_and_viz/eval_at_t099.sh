#!/bin/bash
# Evaluate all ablation checkpoints at the calibrated threshold t=0.99.
# Intended to be run after run_k_ablation.sh and run_feature_ablation.sh complete.
set -e
mkdir -p run_logs/eval_t099

evaluate() {
    local ckpt=$1
    local tag=$2
    local log=run_logs/eval_t099/${tag}.log
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $ckpt not found"
        return
    fi
    echo "=== Evaluating $tag at t=0.99 ==="
    python scripts/train.py evaluate \
        --cell-gnn-checkpoint $ckpt \
        --cache-dir data/boxes \
        --partition-threshold 0.99 \
        2>&1 | tee $log | grep -E "F1:|Precision:|Recall:" | head -3
    echo ""
}

# K ablation
for K in 1 2 3 4 5; do
    evaluate models/cell_gnn_seg_K${K}.pt K${K}
done

# Per-feature ablation
for F in distance same_scaffold grammar_score shared_agents shared_partners seg_connectivity; do
    evaluate models/cell_gnn_seg_drop_${F}.pt drop_${F}
done

# Reference model
evaluate models/cell_gnn_seg.pt baseline_seg

echo "=== All evaluations complete ==="
