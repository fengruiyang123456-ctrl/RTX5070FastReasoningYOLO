Project_Song
============

Minimal, reproducible baseline vs optimized inference comparison for YOLO.

Highlights
----------
- Baseline: PyTorch FP32
- Optimized: ONNX Runtime FP16 (optional TensorRT FP16)
- Outputs: FPS, avg latency, P95, P99, GPU memory when available
- Demo: split-screen comparison from a webcam or video

Quick start (Ubuntu)
-------------------
1) System deps:
   `bash scripts/00_system_deps.sh`

2) Python env:
   `bash scripts/01_create_env.sh`

3) Export ONNX (fixed input size):
   `bash scripts/02_export_onnx.sh`

4) Run benchmarks:
   `bash scripts/03_benchmark_all.sh`

5) Run split-screen demo:
   `bash scripts/04_demo_split_screen.sh`

Weights
-------
- Place baseline weights at `weights/yolo.pt`
- Place exported ONNX at `weights/yolo.onnx`

Notes
-----
- The ORT backend uses a simple YOLOv8-style postprocess; it is meant to be
  a clean, minimal template.
- TensorRT backend is a stub until an engine is generated; see
  `src/backends/trt_fp16.py` for details.
