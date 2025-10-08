#!/usr/bin/env bash
set -euo pipefail

# ===== 可调项 =====
PY_VER="${PY_VER:-3.10}"             # Python 版本
VENV_DIR="${VENV_DIR:-.venv}"        # venv 位置（项目内）
CUDA_STREAM="${CUDA_STREAM:-auto}"   # auto / cu124 / cu118 / cpu

# 把各种缓存指到项目目录（可选，但利于随项目保留）
export UV_CACHE_DIR="${PWD}/.cache/uv"
export HF_HOME="${PWD}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${PWD}/.cache/torch"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

echo "==> [1/5] 安装/准备 Python ${PY_VER}"
uv python install "${PY_VER}"

echo "==> [2/5] 创建 venv: ${VENV_DIR}"
uv venv --python "${PY_VER}" "${VENV_DIR}"

torch_index() {
  local s="${CUDA_STREAM}"
  if [[ "$s" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      local v; v=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)
      case "$v" in
        12.*) s="cu124" ;;
        11.8) s="cu118" ;;
        *)    s="cu124" ;;
      esac
    else
      s="cpu"
    fi
  fi
  case "$s" in
    cu124) echo "https://download.pytorch.org/whl/cu124" ;;
    cu118) echo "https://download.pytorch.org/whl/cu118" ;;
    cpu)   echo "https://download.pytorch.org/whl/cpu" ;;
    *)     echo "https://download.pytorch.org/whl/cpu" ;;
  esac
}

IDX="$(torch_index)"
echo "==> [3/5] 安装 PyTorch from: $IDX"
uv pip install --index-url "$IDX" \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio"

echo "==> [4/5] 安装项目及依赖 (editable)"
uv pip install -e .

echo "==> [5/5] 验证"
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "Done. 激活: source ${VENV_DIR}/bin/activate  或直接用  uv run <cmd>"
