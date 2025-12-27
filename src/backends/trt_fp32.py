from typing import Tuple

import numpy as np

from backends.trt_fp16 import TensorRTBackend


class TensorRTFP32Backend(TensorRTBackend):
    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().infer(frame)
