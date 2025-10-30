import os
from typing import List

import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb


class TritonPythonModel:
    def initialize(self, args):
        model_repository = args["model_repository"]
        assets_dir = os.path.join(model_repository, "mxbai_assets")
        if not os.path.isdir(assets_dir):
            raise RuntimeError(f"Assets directory not found: {assets_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(assets_dir, use_fast=True)
        self.max_length = 512

    def _decode_bytes_tensor(self, tensor: np.ndarray) -> List[str]:
        flat = tensor.reshape(-1)
        return [item.decode("utf-8") for item in flat]

    def execute(self, requests):
        responses = []
        for request in requests:
            query_tensor = pb.get_input_tensor_by_name(request, "QUERY").as_numpy()
            documents_tensor = pb.get_input_tensor_by_name(request, "DOCUMENTS").as_numpy()

            queries = self._decode_bytes_tensor(query_tensor)
            documents = self._decode_bytes_tensor(documents_tensor)

            if len(queries) != 1:
                raise ValueError("Expected exactly one query per request")
            query = queries[0]

            if not documents:
                raise ValueError("Received empty documents tensor")

            encodings = self.tokenizer(
                [query] * len(documents),
                documents,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            input_ids = encodings["input_ids"].astype(np.int64)
            attention_mask = encodings["attention_mask"].astype(np.int64)

            outputs = [
                pb.Tensor("input_ids", input_ids),
                pb.Tensor("attention_mask", attention_mask),
            ]
            responses.append(pb.InferenceResponse(output_tensors=outputs))
        return responses

    def finalize(self):
        pass
