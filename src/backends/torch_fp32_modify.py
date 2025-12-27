from typing import Tuple

import numpy as np

from common.postprocess import postprocess_yolov8

try:
    import torch
    import torch.nn.functional as torch_f
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch_fp32_modify requires torch.") from exc

from ultralytics import YOLO


class TorchFP32ModifyBackend:
    def __init__(
        self,
        weights_path: str,
        device: str,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
    ) -> None:
        if "cuda" not in device:
            raise RuntimeError("torch_fp32_modify requires a CUDA device.")
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

        self.model = YOLO(weights_path)
        self.torch_model = self.model.model.to(self.device).eval()

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.infer(dummy)

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        ratio = min(self.imgsz / h, self.imgsz / w)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        dw = (self.imgsz - new_w) / 2
        dh = (self.imgsz - new_h) / 2
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))

        frame_t = torch.from_numpy(frame).to(device=self.device, non_blocking=True)
        frame_t = frame_t[..., [2, 1, 0]]
        frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)
        frame_t = frame_t.float().div_(255.0)
        if (new_h, new_w) != (h, w):
            frame_t = torch_f.interpolate(
                frame_t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        frame_t = torch_f.pad(
            frame_t, (left, right, top, bottom), value=114.0 / 255.0
        )
        frame_t = frame_t.contiguous()

        with torch.no_grad():
            pred = self.torch_model(frame_t)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
        pred = pred.detach().cpu().numpy()

        boxes, scores, class_ids = postprocess_yolov8(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            ratio=ratio,
            pad=(dw, dh),
            orig_shape=frame.shape[:2],
        )
        return boxes, scores, class_ids
