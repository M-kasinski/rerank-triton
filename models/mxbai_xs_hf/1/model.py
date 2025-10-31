import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import triton_python_backend_utils as pb


class TritonPythonModel:
    def initialize(self, args):
        model_dir = Path(__file__).parent
        assets_dir = model_dir / "assets"
        if not assets_dir.is_dir():
            raise RuntimeError(f"Assets directory not found: {assets_dir}")

        config = json.loads(args.get("model_config", "{}"))
        self.max_length = int(config.get("parameters", {}).get("max_length", 512))
        self.max_batch_size = int(config.get("max_batch_size") or 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(str(assets_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(assets_dir), torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _decode_string_tensor(tensor: np.ndarray) -> List[str]:
        flat = tensor.reshape(-1)
        decoded: List[str] = []
        for item in flat:
            if isinstance(item, bytes):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    def execute(self, requests):
        responses = []
        with torch.inference_mode():
            for request in requests:
                query_tensor = pb.get_input_tensor_by_name(request, "QUERY").as_numpy()
                documents_tensor = pb.get_input_tensor_by_name(request, "DOCUMENTS").as_numpy()

                queries = self._decode_string_tensor(query_tensor)
                documents = self._decode_string_tensor(documents_tensor)

                if len(queries) != 1:
                    raise ValueError("Expected exactly one query per request")
                if not documents:
                    raise ValueError("Received empty documents tensor")

                if len(documents) > self.max_batch_size:
                    raise ValueError(
                        f"Number of documents ({len(documents)}) exceeds max_batch_size={self.max_batch_size}"
                    )

                tokenized = self.tokenizer(
                    [queries[0]] * len(documents),
                    documents,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenized = {name: tensor.to(self.device) for name, tensor in tokenized.items()}

                logits = self.model(**tokenized).logits.squeeze(-1)
                scores = torch.sigmoid(logits)

                try:
                    top_n_tensor = pb.get_input_tensor_by_name(request, "TOP_N").as_numpy()
                    top_n = int(top_n_tensor.reshape(-1)[0])
                except Exception:
                    top_n = len(documents)

                top_n = max(0, min(top_n, len(documents)))
                if top_n == 0:
                    top_n = len(documents)

                scores_np = scores.detach().cpu().numpy().astype(np.float32)
                indices = np.argsort(scores_np)[::-1][:top_n]

                top_scores = np.ascontiguousarray(scores_np[indices])
                top_indices = np.ascontiguousarray(indices.astype(np.int64))

                response = pb.InferenceResponse(
                    output_tensors=[
                        pb.Tensor("scores", top_scores),
                        pb.Tensor("indices", top_indices),
                    ]
                )
                responses.append(response)
        return responses

    def finalize(self):
        pass
