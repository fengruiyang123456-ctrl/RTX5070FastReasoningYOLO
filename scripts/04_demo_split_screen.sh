#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="weights/yolo.pt"
ONNX="weights/yolo.onnx"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS"
  exit 1
fi
if [[ ! -f "$ONNX" ]]; then
  echo "Missing ONNX: $ONNX"
  echo "Run: bash scripts/02_export_onnx.sh"
  exit 1
fi

python src/apps/demo_compare.py --baseline torch_fp32 --optimized trt_fp32 \
  --weights "$WEIGHTS" --onnx "$ONNX" \
  --trt-engine "outputs/trt_engines/yolo_fp32.plan"
