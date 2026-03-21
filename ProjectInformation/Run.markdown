# Run Guide

## Required Files

Make sure these files exist before running:

- `weights/yolo.pt`
- `weights/yolo.onnx`

If `weights/yolo.onnx` does not exist yet:

```bash
bash scripts/02_export_onnx.sh
```

## Local Run

Run all enabled benchmarks:

```bash
bash scripts/03_benchmark_all.sh
```

Run the split-screen comparison demo:

```bash
bash scripts/04_demo_split_screen.sh
```

Run a single backend:

```bash
python -m src.apps.run_camera --backend torch_fp32 --source 0
```

Run an ONNX Runtime benchmark manually:

```bash
python -m src.apps.benchmark --backend ort_fp16 --source 0
```

Run TensorRT manually if you already have an engine file:

```bash
python -m src.apps.benchmark \
  --backend trt_fp16 \
  --trt-engine outputs/trt_engines/yolo_fp16.plan
```

## Docker Run

Build the image:

```bash
docker compose build
```

Run benchmarks inside the container:

```bash
docker compose run --rm app bash -lc "bash scripts/03_benchmark_all.sh"
```

Run the split-screen demo inside the container:

```bash
docker compose run --rm app bash -lc "bash scripts/04_demo_split_screen.sh"
```

Open a shell inside the container:

```bash
docker compose run --rm app bash
```

## Outputs

Benchmark results are written to:

```text
outputs/benches/
```

TensorRT engine files are expected under:

```text
outputs/trt_engines/
```
