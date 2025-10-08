#!/usr/bin/env bash
set -euo pipefail

# 进入仓库根目录（当前脚本在 scripts/single_node/）
cd "$(dirname "$0")/../.."

# 激活项目虚拟环境
source .venv/bin/activate

# 统一缓存与并行设置（不要再用 TRANSFORMERS_CACHE）
export HF_HOME="$PWD/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false

# -------- 从 .secrets/ 读取密钥（不要提交到 Git）--------
[[ -f .secrets/wandb.env ]] && source .secrets/wandb.env
[[ -f .secrets/hf.env    ]] && source .secrets/hf.env

# 必要性检查（缺失就报错退出）
: "${WANDB_API_KEY:?WANDB_API_KEY not set; put it in .secrets/wandb.env}"
: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN not set; put it in .secrets/hf.env}"

# 兼容部分库
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"

# -------- W&B 运行命名等，可按需保留/修改 --------
: "${WANDB_PROJECT:=flow-grpo}"           # 若未在 wandb.env 指定则默认
export WANDB_PROJECT
export WANDB_NAME="sd3_ocr_$(date +%m%d_%H%M)_g1"

# （可选）静默登录一次，便于子进程/工具复用
command -v wandb >/dev/null 2>&1 && wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
command -v huggingface-cli >/dev/null 2>&1 && \
  huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential --non-interactive >/dev/null 2>&1 || true

# 启动（等价于 accelerate launch ...）
python -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes 1 \
  --main_process_port 29501 \
  scripts/train_sd3.py \
  --config config/grpo.py:general_ocr_sd3_1gpu


# 1 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
