#!/bin/bash
# Per-feature ablation: train models with each edge feature zeroed out.
# Indices: 0=distance, 1=same_scaffold, 2=grammar_score,
#          3=shared_agents, 4=shared_partners, 5=seg_connectivity
set -e
mkdir -p run_logs/feature_ablation

NAMES=(distance same_scaffold grammar_score shared_agents shared_partners seg_connectivity)

for IDX in 0 1 2 3 4 5; do
    NAME=${NAMES[$IDX]}
    out=models/cell_gnn_seg_drop_${NAME}.pt
    log=run_logs/feature_ablation/drop_${NAME}.log
    if [ -f "$out" ]; then
        echo "[drop=$NAME] checkpoint exists, skipping"
        continue
    fi
    echo "=== Training without $NAME (idx=$IDX) ==="
    python scripts/train.py train-cell-gnn \
        --cache-dir data/boxes \
        --epochs 50 \
        --d-model 64 --n-layers 3 --n-heads 4 --embedding-dim 32 \
        --cell-gnn-output $out \
        --seg-scores-cache data/seg_scores.json \
        --no-hard-neg-mining \
        --seed 42 \
        --ablate-feature $IDX \
        2>&1 | tee $log
    echo ""
done
echo "=== Per-feature ablation complete ==="
