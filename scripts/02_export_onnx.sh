#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="weights/yolo.pt"
OUT_ONNX="weights/yolo.onnx"
IMGSZ="${1:-640}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS"
  exit 1
fi

python - <<PY
from ultralytics import YOLO

model = YOLO("$WEIGHTS")
model.export(format="onnx", imgsz=$IMGSZ, dynamic=False, half=True, simplify=True, opset=17)
PY

if [[ -f "weights/yolo.onnx" ]]; then
  echo "Exported ONNX: $OUT_ONNX"
else
  echo "Export did not produce weights/yolo.onnx"
  exit 1
fi
