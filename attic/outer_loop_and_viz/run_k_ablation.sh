#!/bin/bash
# K-hop ablation: train models with varying n_layers
set -e
mkdir -p run_logs/k_ablation

for K in 1 2 3 4 5; do
    out=models/cell_gnn_seg_K${K}.pt
    log=run_logs/k_ablation/K${K}.log
    if [ -f "$out" ]; then
        echo "[K=$K] checkpoint exists, skipping"
        continue
    fi
    echo "=== Training K=$K ==="
    python scripts/train.py train-cell-gnn \
        --cache-dir data/boxes \
        --epochs 50 \
        --d-model 64 --n-layers $K --n-heads 4 --embedding-dim 32 \
        --cell-gnn-output $out \
        --seg-scores-cache data/seg_scores.json \
        --no-hard-neg-mining \
        --seed 42 \
        2>&1 | tee $log
    echo ""
done
echo "=== K-hop ablation complete ==="
