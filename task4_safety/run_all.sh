#!/usr/bin/env bash
# Task 4 GPU work (used by the Kaggle notebook): deterministic XSTest responses + reward scores for
# SFT/DPO/PPO/GRPO, then the fixed AI judge on every response. The manual audit and the final
# evaluation run locally on CPU afterwards:
#   python -m task4_safety.make_audit_sheet   # blinded sheet -> label it by hand
#   python -m task4_safety.evaluate_safety    # all metrics, agreement, report/task4/
#
# With >= 2 GPUs: GPU 0 handles SFT + DPO, GPU 1 handles PPO + GRPO. Finished files are skipped / resumed.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/feedback.yaml}
LOG=results/task4_safety/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Stage the three standard adapters from Tasks 1-3 (git-ignored, so not in a fresh clone).
python - <<'EOF'
import glob, os, shutil, zipfile
need = ["outputs/task1_dpo/standard", "outputs/task2_ppo/standard", "outputs/task3_grpo/standard"]
missing = [d for d in need if not os.path.exists(os.path.join(d, "adapter_config.json"))]
roots = [os.environ.get("ADAPTER_SOURCE", "/kaggle/input")]
for root in roots:
    if not missing or not os.path.isdir(root):
        continue
    for z in glob.glob(f"{root}/**/*.zip", recursive=True):
        with zipfile.ZipFile(z) as f:
            members = [m for m in f.namelist() if any(m.startswith(d + "/") for d in missing) and "/value/" not in m]
            if members:
                f.extractall(".", members)
                print(f"extracted {len(members)} files from {z}")
    for d in list(missing):
        hits = glob.glob(f"{root}/**/{d}/adapter_config.json", recursive=True)
        if hits and not os.path.exists(os.path.join(d, "adapter_config.json")):
            shutil.copytree(os.path.dirname(hits[0]), d, ignore=shutil.ignore_patterns("value"), dirs_exist_ok=True)
            print(f"copied {os.path.dirname(hits[0])} -> {d}")
    missing = [d for d in need if not os.path.exists(os.path.join(d, "adapter_config.json"))]
if missing:
    raise SystemExit(f"Missing standard adapters: {missing}. Attach the Task 1-3 artifacts (zip or folders) as notebook input.")
print("All standard adapters present:", need)
EOF

# 2) Fetch the public policy / reward / judge models once, before two processes race for the cache.
python - <<EOF
from huggingface_hub import snapshot_download
from common.data import load_yaml
cfg = load_yaml("$CFG")
for repo in {cfg["base_model"], cfg["reward_model"], cfg["reward_tokenizer"], cfg["ai_judge_model"]}:
    snapshot_download(repo)
EOF

chain() {  # $@ = policies
  python -m task4_safety.generate_responses --config "$CFG" --policies "$@"
  python -m task4_safety.judge_responses --config "$CFG" --policies "$@"
}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs visible: $NGPU"
if [ "$NGPU" -ge 2 ]; then
  ( export CUDA_VISIBLE_DEVICES=0; chain sft dpo ) 2>&1 | sed -u 's/^/[gpu0] /' | tee "$LOG/gpu0.log" &
  PID_A=$!
  ( export CUDA_VISIBLE_DEVICES=1; chain ppo grpo ) 2>&1 | sed -u 's/^/[gpu1] /' | tee "$LOG/gpu1.log" &
  PID_B=$!
  FAIL=0
  wait $PID_A || FAIL=1
  wait $PID_B || FAIL=1
  [ $FAIL -eq 0 ] || { echo "A Task 4 chain failed; see $LOG/"; exit 1; }
else
  chain sft dpo ppo grpo 2>&1 | tee "$LOG/gpu0.log"
fi

# 3) Judge-based metrics/figures now (CPU); rerun locally after the manual audit to add agreement.
python -m task4_safety.evaluate_safety --config "$CFG"
echo "Task 4 GPU stage complete."
