# Repository Guidelines

## Project Structure & Module Organization
Core Triton models live under `models/`, with Python backends in `mxbai_xs_preproc/1/model.py` and `mxbai_xs_postproc/1/model.py`, the ONNX runtime bundle in `mxbai_xs_ort/`, and the ensemble wiring in `mxbai_xs_ensemble/`. Keep tokenizer assets inside each model’s `1/assets/` folder; `scripts/download_assets.sh` populates them. Reference payload samples (`payload_24.json`, `payload_full.json`, `rerank_query_full.json`) when crafting inference requests.

## Build, Test, and Development Commands
Run the asset bootstrapper before building to ensure tokenizer files are available:
```bash
scripts/download_assets.sh
```
Build the Triton image locally and tag it for iteration:
```bash
docker build -t mxbai-triton:xs .
```
Start Triton with GPU access and the default ports exposed:
```bash
docker run --rm --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 mxbai-triton:xs
```
Use the health and inference calls from the README once the container is live.

## Coding Style & Naming Conventions
Python backends follow PEP 8 with four-space indentation, explicit type hints, and descriptive snake_case names (e.g., `max_batch_size`). Favor pure functions and keep Triton-specific utilities (`triton_python_backend_utils`) at the edges. For `config.pbtxt`, continue the existing lowercase, underscore-separated model names, and align tensor names (`QUERY`, `DOCUMENTS`, `top_n`) with the ensemble wiring.

## Testing Guidelines
There is no standalone unit test suite; validate changes by running the container and replaying sample payloads via `curl` or the JSON files in the root directory. Confirm `scores` and `indices` tensors preserve shape `[1, TOP_N]`, and exercise edge cases such as empty documents, oversized batches, or omitted `TOP_N`. When modifying preprocessing, assert tokenizer assets are refreshed and logsoftmax outputs remain stable by comparing against previous snapshots.

## Commit & Pull Request Guidelines
Commit messages should stay concise, start with a capitalized imperative verb (e.g., “Enhance model configurations”), and summarize the primary change in one line. Pull requests must describe motivation, notable implementation decisions, and manual verification steps; link issue IDs when available. Attach before/after metrics or sample responses if behavior changes, and mention any new assets or configuration tweaks that operators must apply.
