#!/usr/bin/env bash
# Full-pipeline timing harness.
# Runs each step sequentially, capturing wall/user/sys time.
# On the first failing step, halts and records the failure in timing.tsv.
#
# Usage:
#   bash scripts/run_timing_pipeline.sh <run_dir>
#
# The caller is expected to wrap this in `caffeinate -i nohup ... &`.

set -u
set -o pipefail

PROJECT_ROOT="/Users/wgray13/projects/neuronauts"
cd "$PROJECT_ROOT"

RUN_DIR="${1:-run_logs/timing_profile_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

TIMING_TSV="$RUN_DIR/timing.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
PIPELINE_LOG="$RUN_DIR/pipeline.log"

printf "step\twall_s\tuser_s\tsys_s\texit_code\n" > "$TIMING_TSV"

source "$PROJECT_ROOT/.venv/bin/activate"

# Ensure SSL cert verification works for caveclient -> global.daf-apis.com.
# Python 3.14's bundled SSL doesn't find a system CA bundle on macOS by
# default; without this the orchestrated steps see SSL: CERTIFICATE_VERIFY_FAILED
# even though `cloudvolume` (used by build-dataset) tunnels around the issue.
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

log()  { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$PIPELINE_LOG"; }

# Extract the most informative summary line(s) from a step's log file.
# Patterns are step-specific so the Monitor surface stays signal-heavy.
summarise_step() {
  local step="$1"
  local logfile="$2"
  local out=""
  case "$step" in
    step1_build_dataset)
      local n_ok n_skip last
      n_ok=$(grep -cE " ok \(n_synapses=" "$logfile" 2>/dev/null || echo 0)
      n_skip=$(grep -cE " skip " "$logfile" 2>/dev/null || echo 0)
      last=$(grep -E "Dataset build complete|Done\." "$logfile" 2>/dev/null | tail -1)
      out="boxes ok=$n_ok skip=$n_skip; $last"
      ;;
    step1b_verify_lineage)
      out=$(grep -E "cache != v117 root|FAIL:|OK:" "$logfile" 2>/dev/null | head -3 | tr '\n' '|')
      ;;
    step2_fetch_edits)
      out=$(grep -E "done:|No edit pairs|Saved.*pairs|Saved.*chains" "$logfile" 2>/dev/null | tail -3 | tr '\n' '|')
      ;;
    step3_path_encoder|step4_grammar|step5_cell_gnn)
      # Last few epoch / val / best lines
      out=$(grep -iE "^epoch |val_|best|saved|loss=|acc=" "$logfile" 2>/dev/null | tail -4 | tr '\n' '|')
      ;;
    step6_eval_30um|step7_eval_v117)
      out=$(grep -E "F1:|Precision:|Recall:|Delta F1|n_boxes" "$logfile" 2>/dev/null | tail -6 | tr '\n' '|')
      ;;
    *)
      out=$(tail -2 "$logfile" 2>/dev/null | tr '\n' '|')
      ;;
  esac
  [ -n "$out" ] && log "    SUMMARY: $out"
}

# Run a single step under /usr/bin/time -p.
# Args: step_name, log_file, command...
run_step() {
  local step="$1"; shift
  local logfile="$1"; shift
  local timefile="$RUN_DIR/${step}.time"

  log "=== START $step ==="
  log "    cmd: $*"

  # Run command in the background so we can heartbeat box-fetch progress
  # for long-running data builds without losing the live update.
  /usr/bin/time -p -o "$timefile" "$@" >"$logfile" 2>&1 &
  local pid=$!

  # Periodic heartbeat: emit recent progress every 30s while the step runs.
  # Heartbeat content is step-specific (defined alongside summarise_step).
  local last_hb_count=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    case "$step" in
      step1_build_dataset)
        local cur=$(grep -cE " ok \(n_synapses=" "$logfile" 2>/dev/null || echo 0)
        local skp=$(grep -cE " skip " "$logfile" 2>/dev/null || echo 0)
        if [ "$cur" -ne "$last_hb_count" ]; then
          log "    HB step1: ok=$cur skip=$skp  (latest: $(grep -E '^\s*\[' "$logfile" 2>/dev/null | tail -1))"
          last_hb_count=$cur
        fi
        ;;
      step2_fetch_edits)
        local resolved=$(grep -E "svids resolved" "$logfile" 2>/dev/null | tail -1)
        [ -n "$resolved" ] && log "    HB step2: $resolved"
        ;;
      step3_path_encoder|step4_grammar|step5_cell_gnn)
        local last_ep
        last_ep=$(grep -iE "^epoch |val_|loss=" "$logfile" 2>/dev/null | tail -1)
        [ -n "$last_ep" ] && log "    HB ${step}: $last_ep"
        ;;
      step6_eval_30um|step7_eval_v117)
        local last_box
        last_box=$(grep -E "GNN F1=|F1=" "$logfile" 2>/dev/null | tail -1)
        [ -n "$last_box" ] && log "    HB ${step}: $last_box"
        ;;
    esac
  done
  wait "$pid"
  local rc=$?

  local wall user sys
  wall=$(awk '$1=="real"{print $2}' "$timefile")
  user=$(awk '$1=="user"{print $2}' "$timefile")
  sys=$(awk  '$1=="sys" {print $2}' "$timefile")

  printf "%s\t%s\t%s\t%s\t%d\n" "$step" "${wall:-NA}" "${user:-NA}" "${sys:-NA}" "$rc" >> "$TIMING_TSV"
  summarise_step "$step" "$logfile"
  log "=== END   $step  exit=$rc  wall=${wall}s ==="

  return $rc
}

# A failing step aborts the chain.
must() {
  if ! run_step "$@"; then
    log "FAILED: $1 — aborting pipeline."
    finalize 1
  fi
}

finalize() {
  local rc="${1:-0}"
  {
    echo "# Pipeline timing summary"
    echo
    echo "Run dir: \`$RUN_DIR\`"
    echo "Started: $START_TS"
    echo "Ended:   $(date)"
    echo
    echo "| step | wall (s) | wall (h:mm:ss) | exit |"
    echo "|------|---------:|---------------:|-----:|"
    awk -F'\t' 'NR>1 {
      wall = $2 + 0
      h = int(wall/3600); m = int((wall%3600)/60); s = int(wall%60)
      printf "| %s | %.2f | %d:%02d:%02d | %d |\n", $1, wall, h, m, s, $5
    }' "$TIMING_TSV"
    echo
    echo "## Logs"
    echo
    for f in "$RUN_DIR"/*.log; do
      [ "$f" = "$PIPELINE_LOG" ] && continue
      echo "- \`$(basename "$f")\`"
    done
  } > "$SUMMARY_MD"
  log "Summary written to $SUMMARY_MD"
  exit "$rc"
}

START_TS="$(date)"
log "Pipeline starting. Run dir: $RUN_DIR"
log "Python: $(python -V 2>&1)  $(which python)"

# Step 0: install editable + dev extras.
must step0_install     "$RUN_DIR/step0_install.log" \
  pip install -e ".[dev]"

# Step 1: build box cache, seeded from proofread cells.
# --max-synapses 100000 (default 20000 skips ~44% of dense proofread-cell
# boxes; observed median 19K, max 32K).
must step1_build_dataset "$RUN_DIR/step1_build_dataset.log" \
  python scripts/train.py build-dataset \
    --cache-dir data/boxes_30um \
    --counts-tsv run_logs/synapse_root_counts_static.tsv \
    --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv \
    --n-boxes 300 \
    --box-side-um 30 \
    --no-em \
    --min-synapses 500 \
    --max-synapses 100000 \
    --seed 42

# Step 1b: verify the new cache has v117 lineage edits.  Aborts the pipeline if
# fewer than 5% of probed svids show lineage divergence — that means the cache
# would yield no edit pairs and step 2 would silently produce zero training data.
must step1b_verify_lineage "$RUN_DIR/step1b_verify_lineage.log" \
  python attic/one_off_analyses/verify_cache_lineage.py \
    --cache-dir data/boxes_30um \
    --min-edit-fraction 0.05 \
    --n-svids 1000

# Step 2: fetch edit pairs from the cache.
must step2_fetch_edits "$RUN_DIR/step2_fetch_edits.log" \
  python scripts/train.py fetch-cave-edits-from-cache \
    --cache-dir data/boxes_30um \
    --output-tsv data/cave_edit_pairs_v3.tsv \
    --output-chains data/cave_edit_chains_v3.npz

# Step 3: train path encoder (~2h).
must step3_path_encoder "$RUN_DIR/step3_path_encoder.log" \
  python scripts/train.py train-path-encoder \
    --cache-dir data/boxes_30um \
    --epochs 10 \
    --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
    --edit-chains-npz data/cave_edit_chains_v3.npz \
    --output models/path_encoder_v3.pt \
    --checkpoint-every 2 \
    --seed 42

# Step 4: train grammar (~6h).
must step4_grammar "$RUN_DIR/step4_grammar.log" \
  python scripts/train.py train \
    --cache-dir data/boxes_30um \
    --epochs 10 \
    --grammar-output models/grammar_30um_v1.pt

# Step 5: train CellGNN (~8h).
must step5_cell_gnn "$RUN_DIR/step5_cell_gnn.log" \
  python scripts/train.py train-cell-gnn \
    --cache-dir data/boxes_30um \
    --epochs 10 \
    --path-encoder-checkpoint models/path_encoder_v3_best.pt \
    --pretrained-path-emb-dim 16 \
    --cell-gnn-output models/cell_gnn_v3.pt \
    --checkpoint-every 2 \
    --n-layers 2 \
    --seed 42

# Step 6: evaluate on the 30µm test split.
must step6_eval_30um "$RUN_DIR/step6_eval_30um.log" \
  python scripts/train.py evaluate \
    --cache-dir data/boxes_30um \
    --cell-gnn-checkpoint models/cell_gnn_v3.pt \
    --split test

# Step 7: cross-version F1 on v117 boxes (whole cache, no split filter).
must step7_eval_v117 "$RUN_DIR/step7_eval_v117.log" \
  python scripts/train.py evaluate \
    --cache-dir data/boxes_v117 \
    --cell-gnn-checkpoint models/cell_gnn_v3.pt

finalize 0
