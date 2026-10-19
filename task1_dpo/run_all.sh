#!/usr/bin/env bash
# Run every Task 1 experiment end to end (used by the Kaggle notebook).
#
# With >= 2 GPUs the work is split into two independent chains that run in parallel:
#   GPU 0: standard DPO (train + eval) -> SFT baseline eval -> length-balanced DPO (train + eval) + length analysis
#   GPU 1: beta forks 0.03 / 0.10 / 0.30 (train + eval each)
# With one GPU the chains run one after the other. Finished runs are skipped, so the
# script can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/dpo.yaml}
LOG=results/task1_dpo/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Validate the corrected DPO objective (CPU, seconds).
python -m task1_dpo.validate_dpo --config "$CFG"

# 2) Fetch the public base/reward models once, before two processes race for the cache.
python - <<EOF
from huggingface_hub import snapshot_download
from common.data import load_yaml
cfg = load_yaml("$CFG")
for repo in {cfg["base_model"], cfg["reward_model"], cfg["reward_tokenizer"]}:
    snapshot_download(repo)
EOF

chain_a() {
  if [ ! -f outputs/task1_dpo/standard/adapter_config.json ]; then
    python -m task1_dpo.train --config "$CFG" --run-name standard
  fi
  [ -f results/task1_dpo/eval/standard/summary.json ] || \
    python -m task1_dpo.evaluate --config "$CFG" --adapter outputs/task1_dpo/standard --name standard
  [ -f results/task1_dpo/eval/sft/summary.json ] || \
    python -m task1_dpo.evaluate --config "$CFG" --name sft
  python -m task1_dpo.analyze_length --config "$CFG"
}

chain_b() {
  python -m task1_dpo.ablate_beta --config "$CFG"
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
  [ $FAIL -eq 0 ] || { echo "A Task 1 chain failed; see $LOG/"; exit 1; }
else
  chain_a 2>&1 | tee "$LOG/gpu0.log"
  chain_b 2>&1 | tee "$LOG/gpu1.log"
fi

# 3) Tables, figures and qualitative candidates for the report (CPU).
python -m task1_dpo.make_report --config "$CFG"
echo "Task 1 complete."
