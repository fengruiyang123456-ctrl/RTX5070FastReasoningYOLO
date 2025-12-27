from typing import Tuple

import numpy as np

from ultralytics import YOLO


class TorchFP16Backend:
    def __init__(
        self,
        weights_path: str,
        device: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
    ) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def warmup(self) -> None:
        _ = self.model.predict(
            np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8),
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            half=True,
            verbose=False,
        )

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            half=True,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int64)
        return boxes, scores, class_ids
