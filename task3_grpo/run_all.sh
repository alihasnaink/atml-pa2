#!/usr/bin/env bash
# Run every Task 3 experiment end to end (used by the Kaggle notebook).
#
# With >= 2 GPUs the work is split into two independent chains that run in parallel:
#   GPU 0: standard 20-update GRPO continuation -> held-out eval of SFT reference, midpoint, standard
#   GPU 1: group-size study on the K=8 cache (CPU) -> canonical GRPO and Dr. GRPO forks (train + eval)
# With one GPU the chains run one after the other. Finished runs are skipped, so the
# script can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/grpo.yaml}
LOG=results/task3_grpo/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Validate the corrected GRPO helpers (CPU, seconds).
python -m task3_grpo.validate_grpo --config "$CFG"

# 2) Fetch the public base/reward models once, before two processes race for the cache.
python - <<EOF
from huggingface_hub import snapshot_download
from common.data import load_yaml
cfg = load_yaml("$CFG")
for repo in {cfg["base_model"], cfg["reward_model"], cfg["reward_tokenizer"]}:
    snapshot_download(repo)
EOF

chain_a() {
  [ -f outputs/task3_grpo/standard/adapter_config.json ] || \
    python -m task3_grpo.continue_train --config "$CFG" --run-name standard
  [ -f results/task3_grpo/eval/sft/summary.json ] || \
    python -m task3_grpo.evaluate --config "$CFG" --name sft
  [ -f results/task3_grpo/eval/midpoint/summary.json ] || \
    python -m task3_grpo.evaluate --config "$CFG" --adapter checkpoints/grpo_midpoint_policy --name midpoint
  [ -f results/task3_grpo/eval/standard/summary.json ] || \
    python -m task3_grpo.evaluate --config "$CFG" --adapter outputs/task3_grpo/standard --name standard
}

chain_b() {
  python -m task3_grpo.analyze_group_size --config "$CFG"
  python -m task3_grpo.compare_normalization --config "$CFG"
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
  [ $FAIL -eq 0 ] || { echo "A Task 3 chain failed; see $LOG/"; exit 1; }
else
  chain_a 2>&1 | tee "$LOG/gpu0.log"
  chain_b 2>&1 | tee "$LOG/gpu1.log"
fi

# 3) Tables, figures and qualitative candidates for the report (CPU).
python -m task3_grpo.make_report --config "$CFG"
echo "Task 3 complete."
