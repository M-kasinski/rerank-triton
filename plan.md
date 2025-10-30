Got it Mika — on va au but et **on te file l’image Docker clé en main**.
Tu n’écris que des fichiers, tu build, tu run. Pas de magie noire.

---

# 0) Arborescence à créer

```
.
├── Dockerfile
├── scripts/
│   └── download_assets.sh
└── models/
    ├── mxbai_xs_preproc/
    │   ├── 1/
    │   │   └── model.py
    │   └── config.pbtxt
    ├── mxbai_xs_ort/
    │   ├── 1/
    │   │   └── model.onnx    # <-- sera téléchargé par le script
    │   └── config.pbtxt
    ├── mxbai_xs_postproc/
    │   ├── 1/
    │   │   └── model.py
    │   └── config.pbtxt
    └── mxbai_xs_ensemble/
        └── config.pbtxt
```

---

# 1) `Dockerfile`

Colle ça tel quel.

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Dépendances pour le backend Python (tokenization)
RUN pip3 install --no-cache-dir \
    transformers==4.44.2 tokenizers==0.19.1 huggingface_hub==0.24.6 onnx==1.16.1

ENV TOKENIZERS_PARALLELISM=true

# Copie du repo de modèles + scripts
COPY models /models
COPY scripts /opt/scripts

# Télécharge les poids ONNX + fichiers tokenizer dans l'image
RUN bash /opt/scripts/download_assets.sh /models

EXPOSE 8000 8001 8002
CMD ["tritonserver","--model-repository=/models","--strict-model-config=true","--exit-on-error=true"]
```

---

# 2) `scripts/download_assets.sh`

Télécharge le **tokenizer** et `onnx/model.onnx` depuis Hugging Face et les place où il faut.

```bash
#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/models}"
ASSETS_DIR="${TARGET}/mxbai_assets"
ORT_MODEL_DIR="${TARGET}/mxbai_xs_ort/1"

mkdir -p "${ASSETS_DIR}" "${ORT_MODEL_DIR}"

python3 - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download

repo_id = "mixedbread-ai/mxbai-rerank-xsmall-v1"
assets_dir = os.environ.get("ASSETS_DIR", "/models/mxbai_assets")
local_dir = snapshot_download(
    repo_id,
    local_dir=assets_dir,
    allow_patterns=[
        "tokenizer.*","vocab.*","merges.*","*.json","*.txt",
        "onnx/model.onnx"
    ],
)
src = os.path.join(local_dir, "onnx", "model.onnx")
dst = os.path.join("/models","mxbai_xs_ort","1","model.onnx")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copyfile(src, dst)
print("Assets ready:", assets_dir, "ONNX copied to:", dst)
PY
```

Rends-le exécutable :
`chmod +x scripts/download_assets.sh`

---

# 3) `models/mxbai_xs_preproc/config.pbtxt`

Préprocessing (tokenization) côté **backend Python**.

```proto
name: "mxbai_xs_preproc"
backend: "python"
max_batch_size: 32

input [
  { name: "QUERY",     data_type: TYPE_BYTES, dims: [1] },
  { name: "DOCUMENTS", data_type: TYPE_BYTES, dims: [-1] }
]

output [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [-1, 256] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [-1, 256] }
]

instance_group [{ kind: KIND_GPU, count: 1 }]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 2000
}
```

> Onsort en **INT64** parce que l’ONNX officiel attend int64. (Quand tu passeras à TensorRT, on passera en int32 côté préproc.)

---

# 4) `models/mxbai_xs_preproc/1/model.py`

Tokenize `(query, doc)` comme **pair**. Pas de prompt exotique.

```python
import numpy as np
import triton_python_backend_utils as pb
from transformers import AutoTokenizer

# On charge localement ce qui a été téléchargé dans l'image
TOK = AutoTokenizer.from_pretrained("/models/mxbai_assets", use_fast=True)
MAX_LEN = 512

class TritonPythonModel:
    def initialize(self, args): 
        pass

    def execute(self, requests):
        rsps = []
        for req in requests:
            q = pb.get_input_tensor_by_name(req, "QUERY").as_numpy()[0].decode("utf-8")
            docs = [b.decode("utf-8") for b in pb.get_input_tensor_by_name(req, "DOCUMENTS").as_numpy()]
            enc = TOK([q]*len(docs), text_pair=docs, padding="max_length",
                      truncation=True, max_length=MAX_LEN, return_tensors="np")
            out = [
                pb.Tensor("input_ids",      enc["input_ids"].astype(np.int64)),
                pb.Tensor("attention_mask", enc["attention_mask"].astype(np.int64)),
            ]
            rsps.append(pb.InferenceResponse(output_tensors=out))
        return rsps
```

---

# 5) `models/mxbai_xs_ort/config.pbtxt`

Modèle **ONNXRuntime** (GPU) — simple et robuste.

```proto
name: "mxbai_xs_ort"
backend: "onnxruntime"
max_batch_size: 32

input [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [256] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [256] }
]

# Le modèle sort 1 logit par pair
output [
  { name: "logits", data_type: TYPE_FP32, dims: [1] }
]

instance_group [{ kind: KIND_GPU, count: 1 }]

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 2000
}
```

---

# 6) `models/mxbai_xs_postproc/config.pbtxt`

Petit post-traitement pour transformer les logits en score [0,1].

```proto
name: "mxbai_xs_postproc"
backend: "python"
max_batch_size: 32

input  [ { name: "logits", data_type: TYPE_FP32, dims: [1] } ]
output [ { name: "scores", data_type: TYPE_FP32, dims: [1] } ]

instance_group [{ kind: KIND_GPU, count: 1 }]
```

---

# 7) `models/mxbai_xs_postproc/1/model.py`

```python
import numpy as np
import triton_python_backend_utils as pb

class TritonPythonModel:
    def initialize(self, args): 
        pass

    def execute(self, requests):
        rsps = []
        for req in requests:
            logits = pb.get_input_tensor_by_name(req, "logits").as_numpy().astype(np.float32) # [B,1]
            scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            rsps.append(pb.InferenceResponse(output_tensors=[
                pb.Tensor("scores", scores)
            ]))
        return rsps
```

---

# 8) `models/mxbai_xs_ensemble/config.pbtxt`

Chaîne préproc → ONNX → postproc en un seul modèle.

```proto
name: "mxbai_xs_ensemble"
platform: "ensemble"
max_batch_size: 32

input  [
  { name: "QUERY",     data_type: TYPE_BYTES, dims: [1] },
  { name: "DOCUMENTS", data_type: TYPE_BYTES, dims: [-1] }
]

output [ { name: "scores", data_type: TYPE_FP32, dims: [-1,1] } ]

ensemble_scheduling {
  step [
    {
      model_name: "mxbai_xs_preproc"
      input_map  { key: "QUERY"     value: "QUERY" }
      input_map  { key: "DOCUMENTS" value: "DOCUMENTS" }
      output_map { key: "input_ids"      value: "input_ids" }
      output_map { key: "attention_mask" value: "attention_mask" }
    },
    {
      model_name: "mxbai_xs_ort"
      input_map  { key: "input_ids"      value: "input_ids" }
      input_map  { key: "attention_mask" value: "attention_mask" }
      output_map { key: "logits" value: "logits" }
    },
    {
      model_name: "mxbai_xs_postproc"
      input_map  { key: "logits" value: "logits" }
      output_map { key: "scores" value: "scores" }
    }
  ]
}
```

---

# 9) Build & Run

**Build**

```bash
docker build -t mxbai-triton:xs .
```

**Run**

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mxbai-triton:xs
```

**Healthchecks**

```bash
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/mxbai_xs_ensemble
```

**Inference (exemple)**

```bash
curl -s -X POST localhost:8000/v2/models/mxbai_xs_ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":[
      {"name":"QUERY","datatype":"BYTES","shape":[1,1],"data":["python async http client"]},
      {"name":"DOCUMENTS","datatype":"BYTES","shape":[1,3],
       "data":["doc A ...","doc B ...","doc C ..."]}
    ],
    "outputs":[{"name":"scores"}]
  }' | jq .
```

Tu auras un tableau `scores` (un score par doc). Trie côté client si tu veux le `top_n`.

---

## Bonus latence (quand tu seras à l’aise)

* **Passer en TensorRT FP16**: build un `model.plan` avec `trtexec`, crée `models/mxbai_xs_trt/` (inputs en `TYPE_INT32`), et pointe l’étape 2 de l’**ensemble** vers `mxbai_xs_trt`. Dans ce cas, change le préproc pour sortir des `INT32`.
