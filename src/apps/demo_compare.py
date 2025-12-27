import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from backends.ort_fp16 import OrtFP16Backend
from backends.torch_fp16 import TorchFP16Backend
from backends.torch_fp32 import TorchFP32Backend
from backends.torch_fp32_modify import TorchFP32ModifyBackend
from backends.trt_fp16 import TensorRTBackend
from backends.trt_fp32 import TensorRTFP32Backend
from common.config import AppConfig
from common.timer import Timer
from common.video_io import iter_frames, open_video
from common.viz import draw_detections, overlay_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-screen demo comparison")
    parser.add_argument("--baseline", default="torch_fp32")
    parser.add_argument("--optimized", default="ort_fp16")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--weights", default=str(AppConfig().weights_path("yolo.pt")))
    parser.add_argument("--onnx", default=str(AppConfig().weights_path("yolo.onnx")))
    parser.add_argument("--trt-engine", default=str(AppConfig().outputs_dir / "trt_engines/yolo_fp16.plan"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def build_backend(name: str, args: argparse.Namespace):
    if name == "torch_fp32":
        return TorchFP32Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)
    if name == "torch_fp32_modify":
        return TorchFP32ModifyBackend(args.weights, args.device, args.conf, args.iou, args.imgsz)
    if name == "torch_fp16":
        return TorchFP16Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)
    if name == "ort_fp16":
        return OrtFP16Backend(args.onnx, args.conf, args.iou, args.imgsz)
    if name == "trt_fp16":
        return TensorRTBackend(args.trt_engine, args.conf, args.iou, args.imgsz)
    if name == "trt_fp32":
        return TensorRTFP32Backend(args.trt_engine, args.conf, args.iou, args.imgsz)
    raise ValueError(f"Unknown backend: {name}")


def main() -> None:
    args = parse_args()
    cfg = AppConfig(conf_thres=args.conf, iou_thres=args.iou)
    baseline = build_backend(args.baseline, args)
    optimized = build_backend(args.optimized, args)

    baseline.warmup()
    optimized.warmup()

    cap = open_video(args.source)
    timer = Timer(use_cuda="cuda" in args.device)

    for ok, frame in iter_frames(cap):
        if not ok:
            break

        timer.start()
        b_boxes, b_scores, b_cls = baseline.infer(frame)
        b_ms = timer.stop()

        timer.start()
        o_boxes, o_scores, o_cls = optimized.infer(frame)
        o_ms = timer.stop()

        left = draw_detections(frame.copy(), b_boxes, b_scores, b_cls, cfg.get_class_names())
        right = draw_detections(frame.copy(), o_boxes, o_scores, o_cls, cfg.get_class_names())

        overlay_metrics(left, [f"{args.baseline} {b_ms:.2f} ms"])
        overlay_metrics(right, [f"{args.optimized} {o_ms:.2f} ms"])

        stacked = np.hstack([left, right])
        cv2.imshow("Baseline | Optimized", stacked)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
