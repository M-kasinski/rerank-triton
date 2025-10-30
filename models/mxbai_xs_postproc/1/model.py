import numpy as np
import triton_python_backend_utils as pb


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb.get_input_tensor_by_name(request, "logits").as_numpy().astype(np.float32)
            scores = 1.0 / (1.0 + np.exp(-logits))

            try:
                top_n_tensor = pb.get_input_tensor_by_name(request, "top_n").as_numpy()
                top_n = int(top_n_tensor.reshape(-1)[0])
            except Exception:
                top_n = logits.shape[0]

            top_n = max(0, min(top_n, logits.shape[0]))
            if top_n == 0:
                top_n = logits.shape[0]

            flat_scores = scores.reshape(-1)
            indices = np.argsort(flat_scores)[::-1][:top_n]

            top_scores = flat_scores[indices][:, np.newaxis].astype(np.float32)
            top_indices = indices.astype(np.int64)

            score_tensor = pb.Tensor("scores", top_scores)
            index_tensor = pb.Tensor("indices", top_indices)
            responses.append(pb.InferenceResponse(output_tensors=[score_tensor, index_tensor]))
        return responses

    def finalize(self):
        pass
