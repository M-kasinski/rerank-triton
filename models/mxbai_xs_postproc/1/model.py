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
            tensor = pb.Tensor("scores", scores)
            responses.append(pb.InferenceResponse(output_tensors=[tensor]))
        return responses

    def finalize(self):
        pass
