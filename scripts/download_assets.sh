#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/models}"
ASSETS_DIR="${TARGET}/mxbai_xs_preproc/1/assets"
ORT_MODEL_DIR="${TARGET}/mxbai_xs_ort/1"

rm -rf "${TARGET}/mxbai_assets" "${ASSETS_DIR}"
mkdir -p "${ASSETS_DIR}" "${ORT_MODEL_DIR}"

ASSETS_DIR="${ASSETS_DIR}" ORT_MODEL_DIR="${ORT_MODEL_DIR}" python3 - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

repo_id = "mixedbread-ai/mxbai-rerank-xsmall-v1"
assets_dir = os.environ["ASSETS_DIR"]
ort_model_dir = os.environ["ORT_MODEL_DIR"]

local_dir = snapshot_download(
    repo_id,
    local_dir=assets_dir,
    allow_patterns=[
        "tokenizer.*",
        "vocab.*",
        "merges.*",
        "*.json",
        "*.txt",
        "onnx/model.onnx",
    ],
)

src = os.path.join(local_dir, "onnx", "model.onnx")
dst = os.path.join(ort_model_dir, "model.onnx")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
print(f"Assets ready: {assets_dir} ONNX copied to: {dst}")
PY
