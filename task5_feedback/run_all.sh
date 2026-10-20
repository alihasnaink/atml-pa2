#!/usr/bin/env bash
# Run every Task 5 experiment end to end (used by the Kaggle notebook). Nothing is trained:
# SFT, RLVR and RLAIF are frozen policies (checkpoints/rlvr_policy, checkpoints/rlaif_policy).
#
# With >= 2 GPUs the work is split into two independent chains that run in parallel:
#   GPU 0: Step 1 - GSM8K in-domain generation (3 policies) + pairwise judge
#   GPU 1: Step 3 - SVAMP transfer generation (3 policies) + pairwise judge, then Step 2 - diagnostic set
# Each chain has its own judge cache file. Finished generations and cached judge calls are reused,
# so the script can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/feedback.yaml}
LOG=results/task5_feedback/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Fetch the public policy / judge models once, before two processes race for the cache.
python - <<PY
from huggingface_hub import snapshot_download
from common.data import load_yaml
cfg = load_yaml("$CFG")
for repo in {cfg["base_model"], cfg["ai_judge_model"]}:
    snapshot_download(repo)
PY

chain_a() {
  python -m task5_feedback.evaluate_math --config "$CFG" --dataset gsm
}

chain_b() {
  python -m task5_feedback.evaluate_math --config "$CFG" --dataset transfer
  python -m task5_feedback.score_perturbations --config "$CFG"
}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs visible: $NGPU"
if [ "$NGPU" -ge 2 ]; then
  ( export CUDA_VISIBLE_DEVICES=0; chain_a ) 2>&1 | sed -u 's/^/[gpu0] /' | tee "$LOG/gpu0.log" &
  PID_A=$!
  ( export CUDA_VISIBLE_DEVICES=1; chain_b ) 2>&1 | sed -u 's/^/[gpu1] /' | tee "$LOG/gpu1.log" &
  PID_B=$!
  FAIL=0
  wait $PID_A || FAIL=1
  wait $PID_B || FAIL=1
  [ $FAIL -eq 0 ] || { echo "A Task 5 chain failed; see $LOG/"; exit 1; }
else
  chain_a 2>&1 | tee "$LOG/gpu0.log"
  chain_b 2>&1 | tee "$LOG/gpu1.log"
fi

# 2) Tables, figures and qualitative candidates for the report (CPU).
python -m task5_feedback.compare_feedback --config "$CFG"
echo "Task 5 complete."
