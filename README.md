# RTX5070 Fast Reasoning YOLO

Benchmark-oriented YOLO inference project for comparing baseline PyTorch execution against optimized ONNX Runtime and TensorRT deployments.

## What this project does

- Runs a reproducible latency and throughput benchmark across multiple inference backends
- Supports side-by-side visual comparison for baseline and optimized pipelines
- Separates preprocessing, postprocessing, timing, visualization, and backend implementations
- Exports benchmark results as JSON and CSV for later analysis

## Supported backends

- `torch_fp32`: baseline PyTorch inference
- `torch_fp16`: PyTorch half-precision inference
- `torch_fp32_custom_preproc`: PyTorch inference with custom CUDA preprocessing
- `ort_fp16`: ONNX Runtime CUDA inference
- `trt_fp16`: TensorRT engine inference
- `trt_fp32`: TensorRT engine wrapper for FP32-compatible plans

Legacy backend name `torch_fp32_modify` is still accepted for compatibility.

## Repository layout

```text
src/
  apps/        CLI entry points
  backends/    Inference backend implementations and registry
  common/      Shared config, preprocessing, postprocessing, timing, visualization
scripts/       Environment setup and benchmark/demo helpers
env/           Python environment definitions
weights/       Local model artifacts (not versioned in a real deployment)
outputs/       Generated benchmark results and TensorRT engines
```

## Quick start

### 1. Create the environment

```bash
bash scripts/00_system_deps.sh
bash scripts/01_create_env.sh
```

### 2. Prepare model artifacts

Place your YOLO weights at `weights/yolo.pt`, then export ONNX:

```bash
bash scripts/02_export_onnx.sh
```

The checked-in files under `weights/` are placeholders only. They are not usable model artifacts.

### 3. Run the benchmark

```bash
bash scripts/03_benchmark_all.sh
```

Outputs are written to `outputs/benches/` as timestamped JSON and CSV files.

### 4. Run the comparison demo

```bash
bash scripts/04_demo_split_screen.sh
```

### 5. Run a single backend manually

```bash
python -m src.apps.run_camera --backend torch_fp32 --source 0
python -m src.apps.benchmark --backend ort_fp16 --source demo.mp4
```

## Project goals

This repository is intended to answer a practical engineering question:

How much latency reduction can be achieved by moving from a baseline PyTorch YOLO pipeline to ONNX Runtime or TensorRT on NVIDIA hardware, while keeping the evaluation flow reproducible and easy to inspect?

## Current limitations

- TensorRT requires a prebuilt engine and local TensorRT dependencies
- Benchmarking currently focuses on single-stream inference
- No automated test suite is included yet
- Model accuracy metrics are not tracked in this repository; the focus is deployment-side performance

## Resume-friendly framing

If you reference this project in a resume, position it as:

"Built a reproducible YOLO inference benchmarking pipeline comparing PyTorch, ONNX Runtime, and TensorRT backends, with benchmark export, split-screen visual validation, and modular preprocessing/postprocessing components."
