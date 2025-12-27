#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="weights/yolo.pt"
OUT_ONNX="weights/yolo.onnx"
IMGSZ="${1:-640}"
CONF="${2:-0.25}"
IOU="${3:-0.45}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS"
  exit 1
fi

python - <<PY
from ultralytics import YOLO

model = YOLO("$WEIGHTS")
model.export(
    format="onnx",
    imgsz=$IMGSZ,
    dynamic=False,
    half=True,
    simplify=True,
    opset=17,
    nms=True,
    conf=$CONF,
    iou=$IOU,
)
PY

if [[ -f "$OUT_ONNX" ]]; then
  python - <<PY
import onnx
from onnx import TensorProto

model = onnx.load("$OUT_ONNX")
model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
onnx.save(model, "$OUT_ONNX")
print("Patched ONNX input type to FP16.")
PY
  echo "Exported ONNX: $OUT_ONNX"
else
  echo "Export did not produce $OUT_ONNX"
  exit 1
fi
