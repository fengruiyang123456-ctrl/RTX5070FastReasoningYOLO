import argparse

import cv2

from src.backends.factory import available_backends, build_backend
from src.common.config import AppConfig
from src.common.video_io import iter_frames, open_video
from src.common.viz import draw_detections, overlay_metrics
from src.common.timer import Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-backend live demo")
    parser.add_argument("--backend", default="torch_fp32", choices=sorted(set(available_backends())))
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--weights", default=str(AppConfig().weights_path("yolo.pt")))
    parser.add_argument("--onnx", default=str(AppConfig().weights_path("yolo.onnx")))
    parser.add_argument("--trt-engine", default=str(AppConfig().outputs_dir / "trt_engines/yolo_fp16.plan"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig(conf_thres=args.conf, iou_thres=args.iou)
    backend = build_backend(args.backend, args)
    backend.warmup()

    cap = open_video(args.source)
    timer = Timer(use_cuda="cuda" in args.device)

    for ok, frame in iter_frames(cap, max_frames=args.max_frames):
        if not ok:
            break
        timer.start()
        boxes, scores, class_ids = backend.infer(frame)
        latency_ms = timer.stop()

        annotated = draw_detections(frame.copy(), boxes, scores, class_ids, cfg.get_class_names())
        overlay_metrics(annotated, [f"{args.backend} {latency_ms:.2f} ms"])

        if not args.no_display:
            cv2.imshow("RTX5070 Fast Reasoning YOLO", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
