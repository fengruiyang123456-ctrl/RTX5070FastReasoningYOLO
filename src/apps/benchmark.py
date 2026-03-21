import argparse
from pathlib import Path
from typing import List

import cv2

from src.backends.factory import available_backends, build_backend
from src.common.config import AppConfig
from src.common.stats import compute_stats, write_benchmark
from src.common.timer import Timer
from src.common.video_io import iter_frames, open_video

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a backend")
    parser.add_argument("--backend", default="torch_fp32", choices=sorted(set(available_backends())))
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
    backend = build_backend(args.backend, args)
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
