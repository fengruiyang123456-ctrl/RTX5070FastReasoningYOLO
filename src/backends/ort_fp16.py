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
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        self.providers = self.session.get_providers()
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.use_fp16 = input_meta.type == "tensor(float16)"
        self.input_dtype = np.float16 if self.use_fp16 else np.float32
        if "CUDAExecutionProvider" not in self.providers and self.use_fp16:
            raise RuntimeError(
                "CUDAExecutionProvider not available but ONNX expects float16. "
                "Install onnxruntime-gpu or export a float32 ONNX model."
            )
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.infer(dummy)

    @staticmethod
    def _is_nms_output(pred: np.ndarray) -> bool:
        if not isinstance(pred, np.ndarray):
            return False
        if pred.ndim == 3 and pred.shape[-1] == 6:
            return True
        if pred.ndim == 2 and pred.shape[1] == 6:
            return True
        return False

    @staticmethod
    def _rescale_boxes(
        boxes: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes /= ratio
        h, w = orig_shape
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    def _postprocess_nms_export(
        self,
        pred: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pred.ndim == 3:
            pred = pred[0]
        if pred.size == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)
        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(np.int64)
        keep = scores >= self.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if boxes.size == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)
        boxes = self._rescale_boxes(boxes, ratio, pad, orig_shape)
        return boxes, scores, class_ids

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized, ratio, pad = letterbox(frame, (self.imgsz, self.imgsz))
        tensor = to_tensor(resized, half=self.use_fp16)
        tensor = tensor.astype(self.input_dtype, copy=False)
        pred = self.session.run(None, {self.input_name: tensor})[0]
        if self._is_nms_output(pred):
            return self._postprocess_nms_export(pred, ratio, pad, frame.shape[:2])
        boxes, scores, class_ids = postprocess_yolov8(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            ratio=ratio,
            pad=pad,
            orig_shape=frame.shape[:2],
        )
        return boxes, scores, class_ids
