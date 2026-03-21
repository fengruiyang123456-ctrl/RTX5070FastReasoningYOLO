import argparse

import cv2
import numpy as np

from src.backends.factory import available_backends, build_backend
from src.common.config import AppConfig
from src.common.timer import Timer
from src.common.video_io import iter_frames, open_video
from src.common.viz import draw_detections, overlay_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-screen demo comparison")
    backend_choices = sorted(set(available_backends()))
    parser.add_argument("--baseline", default="torch_fp32", choices=backend_choices)
    parser.add_argument("--optimized", default="ort_fp16", choices=backend_choices)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--weights", default=str(AppConfig().weights_path("yolo.pt")))
    parser.add_argument("--onnx", default=str(AppConfig().weights_path("yolo.onnx")))
    parser.add_argument("--trt-engine", default=str(AppConfig().outputs_dir / "trt_engines/yolo_fp16.plan"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


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
