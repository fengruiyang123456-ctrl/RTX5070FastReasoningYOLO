from typing import Tuple

import numpy as np
import onnxruntime as ort

from common.postprocess import postprocess_yolov8
from common.preprocess import letterbox, to_tensor


class OrtFP16Backend:
    def __init__(
        self,
        onnx_path: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
    ) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.infer(dummy)

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized, ratio, pad = letterbox(frame, (self.imgsz, self.imgsz))
        tensor = to_tensor(resized, half=True)
        pred = self.session.run(None, {self.input_name: tensor})[0]
        boxes, scores, class_ids = postprocess_yolov8(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            ratio=ratio,
            pad=pad,
            orig_shape=frame.shape[:2],
        )
        return boxes, scores, class_ids
