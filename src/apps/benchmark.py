import argparse
import sys
from pathlib import Path
from typing import List

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from backends.ort_fp16 import OrtFP16Backend
from backends.torch_fp16 import TorchFP16Backend
from backends.torch_fp32 import TorchFP32Backend
from backends.trt_fp16 import TensorRTBackend
from common.config import AppConfig
from common.stats import compute_stats, write_benchmark
from common.timer import Timer
from common.video_io import iter_frames, open_video

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a backend")
    parser.add_argument("--backend", default="torch_fp32")
    parser.add_argument("--source", default="0", help="camera index, video path, or image path")
    parser.add_argument("--weights", default=str(AppConfig().weights_path("yolo.pt")))
    parser.add_argument("--onnx", default=str(AppConfig().weights_path("yolo.onnx")))
    parser.add_argument("--trt-engine", default=str(AppConfig().outputs_dir / "trt_engines/yolo_fp16.plan"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def build_backend(args: argparse.Namespace):
    if args.backend == "torch_fp32":
        return TorchFP32Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)
    if args.backend == "torch_fp16":
        return TorchFP16Backend(args.weights, args.device, args.conf, args.iou, args.imgsz)
    if args.backend == "ort_fp16":
        return OrtFP16Backend(args.onnx, args.conf, args.iou, args.imgsz)
    if args.backend == "trt_fp16":
        return TensorRTBackend(args.trt_engine, args.conf, args.iou, args.imgsz)
    raise ValueError(f"Unknown backend: {args.backend}")


def get_gpu_mem_mb() -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024 * 1024))


def load_source_frames(source: str, max_frames: int) -> List:
    if Path(source).is_file():
        image = cv2.imread(source)
        if image is None:
            raise RuntimeError(f"Failed to read image: {source}")
        return [image] * max_frames

    cap = open_video(source)
    frames = []
    for ok, frame in iter_frames(cap, max_frames=max_frames):
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def main() -> None:
    args = parse_args()
    backend = build_backend(args)
    timer = Timer(use_cuda="cuda" in args.device)

    frames = load_source_frames(args.source, args.num_frames)
    if not frames:
        raise RuntimeError("No frames loaded for benchmarking.")

    for _ in range(args.warmup):
        _ = backend.infer(frames[0])

    latencies_ms: List[float] = []
    for frame in frames:
        timer.start()
        _ = backend.infer(frame)
        latencies_ms.append(timer.stop())

    stats = compute_stats(latencies_ms)
    mem_mb = get_gpu_mem_mb()
    name = args.backend if not args.tag else f"{args.backend}_{args.tag}"
    out_dir = AppConfig().outputs_dir / "benches"
    write_benchmark(out_dir, name, stats, extra={"gpu_mem_mb": mem_mb, "frames": len(frames)})


if __name__ == "__main__":
    main()
