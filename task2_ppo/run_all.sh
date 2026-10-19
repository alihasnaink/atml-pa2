#!/usr/bin/env bash
# Run every Task 2 experiment end to end (used by the Kaggle notebook).
#
# With >= 2 GPUs the work is split into two independent chains that run in parallel:
#   GPU 0: standard 20-update continuation -> held-out eval of SFT reference, midpoint, standard
#          -> KL forks beta_KL = 0.0 and 0.20 (train + eval)
#   GPU 1: cached-batch clipping study -> epsilon forks 0.05 / 0.20 / 0.50 (train + eval);
#          the eps=0.20 fork (beta_KL=0.10) is also the beta_KL=0.10 condition of the KL study
# With one GPU the chains run one after the other. Finished runs are skipped, so the
# script can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/ppo.yaml}
LOG=results/task2_ppo/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Validate the corrected PPO objective and helpers (CPU, seconds).
python -m task2_ppo.validate_ppo --config "$CFG"

# 2) Fetch the public base/reward models once, before two processes race for the cache.
python - <<EOF
from huggingface_hub import snapshot_download
from common.data import load_yaml
cfg = load_yaml("$CFG")
for repo in {cfg["base_model"], cfg["reward_model"], cfg["reward_tokenizer"]}:
    snapshot_download(repo)
EOF

chain_a() {
  [ -f outputs/task2_ppo/standard/adapter_config.json ] || \
    python -m task2_ppo.continue_train --config "$CFG" --run-name standard
  [ -f results/task2_ppo/eval/sft/summary.json ] || \
    python -m task2_ppo.evaluate --config "$CFG" --name sft
  [ -f results/task2_ppo/eval/midpoint/summary.json ] || \
    python -m task2_ppo.evaluate --config "$CFG" --adapter checkpoints/ppo_midpoint_policy --name midpoint
  [ -f results/task2_ppo/eval/standard/summary.json ] || \
    python -m task2_ppo.evaluate --config "$CFG" --adapter outputs/task2_ppo/standard --name standard
  python -m task2_ppo.ablate_kl --config "$CFG" --kl-values 0.0 0.20
}

chain_b() {
  python -m task2_ppo.analyze_clipping --config "$CFG"
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
  [ $FAIL -eq 0 ] || { echo "A Task 2 chain failed; see $LOG/"; exit 1; }
else
  chain_a 2>&1 | tee "$LOG/gpu0.log"
  chain_b 2>&1 | tee "$LOG/gpu1.log"
fi

# 3) All KL conditions now exist (beta_KL=0.10 comes from the eps=0.20 fork); this only checks/skips.
python -m task2_ppo.ablate_kl --config "$CFG"

# 4) Tables, figures and qualitative candidates for the report (CPU).
python -m task2_ppo.make_report --config "$CFG"
echo "Task 2 complete."
