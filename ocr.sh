#!/usr/bin/env bash
set -euo pipefail

# 固定参数（如需临时改再用环境变量覆盖：LANG_CODE=ch ./ocr.sh）
VENV_DIR="${VENV_DIR:-.venv}"   # 已存在的虚拟环境
LANG_CODE="${LANG_CODE:-en}"    # 默认英文
PADDLE_VER="${PADDLE_VER:-2.6.2}"
PADDLEOCR_VER="${PADDLEOCR_VER:-2.9.1}"

# 不使用镜像：INDEX_URL/EXTRA_INDEX_URL 默认为空
INDEX_URL="${INDEX_URL:-}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-}"

# 缓存（可选）
export PADDLE_HOME="${PWD}/.cache/paddle"
mkdir -p "${PADDLE_HOME}"

# 1) 激活已有 venv
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "❌ 未找到 ${VENV_DIR}/bin/python。请先创建并激活虚拟环境。" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# 2) 选 uv（有则用 uv，更快），否则用 pip
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
  RUNPY="uv run python"
else
  PIP="pip"
  RUNPY="python"
fi

$PIP install -U pip wheel setuptools

# 3) 固定安装 GPU 版 Paddle（Meta GPU 场景）
PIP_ARGS=()
[[ -n "$INDEX_URL" ]] && PIP_ARGS+=(--index-url "$INDEX_URL")
[[ -n "$EXTRA_INDEX_URL" ]] && PIP_ARGS+=(--extra-index-url "$EXTRA_INDEX_URL")

echo "==> 安装 GPU 版 Paddle: paddlepaddle-gpu==${PADDLE_VER}"
$PIP install "${PIP_ARGS[@]}" "paddlepaddle-gpu==${PADDLE_VER}"

echo "==> 安装 PaddleOCR 与 Levenshtein"
$PIP install "${PIP_ARGS[@]}" "paddleocr==${PADDLEOCR_VER}" "python-Levenshtein"

# 4) 预下载模型（固定 use_gpu=True）
echo "==> 预下载 OCR 模型 (lang='${LANG_CODE}', use_gpu=True)"
$RUNPY - <<PY
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="${LANG_CODE}", use_gpu=True, show_log=False)
print("PaddleOCR model ready.")
PY

echo "✅ 完成。当前环境：${VENV_DIR} ；语言：${LANG_CODE}；GPU：已启用"
