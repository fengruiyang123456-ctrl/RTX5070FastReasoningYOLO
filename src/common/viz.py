from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return frame

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls_id)} {score:.2f}"
        if class_names and 0 <= int(cls_id) < len(class_names):
            label = f"{class_names[int(cls_id)]} {score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def overlay_metrics(frame: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    y = 20
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 20
    return frame
