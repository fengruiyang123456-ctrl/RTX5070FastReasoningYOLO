from typing import Tuple

import numpy as np

from common.postprocess import postprocess_yolov8
from common.preprocess import letterbox, to_tensor


class TensorRTBackend:
    def __init__(
        self,
        engine_path: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
    ) -> None:
        try:
            import tensorrt as trt  # noqa: F401
            import pycuda.driver as cuda  # noqa: F401
            import pycuda.autoinit  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TensorRT backend requires tensorrt and pycuda. "
                "Generate an engine and install dependencies first."
            ) from exc

        self.engine_path = engine_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        raise RuntimeError(
            "TensorRT backend is a stub. Implement engine load and execution "
            "for your environment, or use ORT FP16."
        )

    def warmup(self) -> None:
        return None

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized, ratio, pad = letterbox(frame, (self.imgsz, self.imgsz))
        _tensor = to_tensor(resized, half=True)
        boxes, scores, class_ids = postprocess_yolov8(
            np.empty((0, 0)),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            ratio=ratio,
            pad=pad,
            orig_shape=frame.shape[:2],
        )
        return boxes, scores, class_ids
