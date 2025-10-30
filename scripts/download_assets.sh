#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/models}"
TMP_DIR="${TARGET}/mxbai_assets"
PREPROC_ASSETS_DIR="${TARGET}/mxbai_xs_preproc/1/assets"
HF_ASSETS_DIR="${TARGET}/mxbai_xs_hf/1/assets"
ORT_MODEL_DIR="${TARGET}/mxbai_xs_ort/1"

rm -rf "${TMP_DIR}" "${PREPROC_ASSETS_DIR}" "${HF_ASSETS_DIR}"
mkdir -p "${PREPROC_ASSETS_DIR}" "${HF_ASSETS_DIR}" "${ORT_MODEL_DIR}"

TMP_DIR="${TMP_DIR}" PREPROC_ASSETS_DIR="${PREPROC_ASSETS_DIR}" HF_ASSETS_DIR="${HF_ASSETS_DIR}" ORT_MODEL_DIR="${ORT_MODEL_DIR}" python3 - <<'PY'
import os
import shutil
from huggingface_hub import snapshot_download

repo_id = "mixedbread-ai/mxbai-rerank-xsmall-v1"
tmp_dir = os.environ["TMP_DIR"]
preproc_assets_dir = os.environ["PREPROC_ASSETS_DIR"]
hf_assets_dir = os.environ["HF_ASSETS_DIR"]
ort_model_dir = os.environ["ORT_MODEL_DIR"]

local_dir = snapshot_download(
    repo_id,
    local_dir=tmp_dir,
    allow_patterns=[
        "tokenizer.*",
        "vocab.*",
        "merges.*",
        "*.json",
        "*.txt",
        "*.bin",
        "*.safetensors",
        "*.model",
        "onnx/model.onnx",
    ],
)

src = os.path.join(local_dir, "onnx", "model.onnx")
dst = os.path.join(ort_model_dir, "model.onnx")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)

def copy_tree(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

copy_tree(local_dir, preproc_assets_dir)
copy_tree(local_dir, hf_assets_dir)

shutil.rmtree(local_dir, ignore_errors=True)

print(f"Assets ready: {preproc_assets_dir} and {hf_assets_dir} ONNX copied to: {dst}")
PY
