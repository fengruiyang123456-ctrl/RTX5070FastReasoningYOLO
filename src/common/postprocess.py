from typing import Tuple

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return np.array(keep, dtype=np.int64)


def postprocess_yolov8(
    pred: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        if pred.shape[0] in (84, 85):
            pred = pred.T

    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all.max(axis=1)

    keep = scores >= conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)

    boxes = xywh_to_xyxy(boxes)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio

    h, w = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

    keep = nms(boxes, scores, iou_thres)
    return boxes[keep], scores[keep], class_ids[keep]


def postprocess_yolov8_resize(
    pred: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    orig_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        if pred.shape[0] in (84, 85):
            pred = pred.T

    boxes = pred[:, :4]
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all.max(axis=1)

    keep = scores >= conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int64)

    boxes = xywh_to_xyxy(boxes)
    h0, w0 = orig_shape
    new_h, new_w = new_shape
    gain_w = new_w / w0
    gain_h = new_h / h0
    boxes[:, [0, 2]] /= gain_w
    boxes[:, [1, 3]] /= gain_h

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

    keep = nms(boxes, scores, iou_thres)
    return boxes[keep], scores[keep], class_ids[keep]
