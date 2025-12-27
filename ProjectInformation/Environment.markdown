# Environment Setup (Project_Song)

This document captures the environment configuration used during optimization and TensorRT tests.
All commands assume Ubuntu 22.04.5 LTS.

## System
- GPU: NVIDIA GeForce RTX 5070 (8 GB)
- CPU: AMD Ryzen 7 9700X
- Driver: 580.95.05
- CUDA Toolkit (nvcc): 11.5
- GPU Driver CUDA Runtime: 13.0 (reported by `nvidia-smi`)

## System Packages (TensorRT)
TensorRT installed from NVIDIA CUDA repo (system-level):
- TensorRT: 10.14.1.48-1+cuda13.0
- tensorrt-dev: provides `trtexec`

Install (if missing):
```
sudo apt-get update
sudo apt-get install -y tensorrt tensorrt-dev python3-libnvinfer python3-libnvinfer-dev
```

`trtexec` path:
```
/usr/src/tensorrt/bin/trtexec
```

## Python Environment (runtime)
Conda environment for runtime:
- Name: rtx5070
- Python: 3.10
- Torch: 2.9.1+cu128
- Ultralytics: 8.3.x
- onnxruntime-gpu
- OpenCV

Create/activate:
```
conda create -n rtx5070 python=3.10 -y
conda activate rtx5070
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics onnx onnxruntime-gpu opencv-python numpy pandas
pip install pycuda
```

Verify TensorRT Python binding:
```
python - <<'PY'
import tensorrt as trt
print("TensorRT:", trt.__version__)
PY
```

## Python Environment (ONNX export, optional clean env)
If you want a clean export environment to avoid protobuf conflicts:
```
conda create -n yolo_export python=3.10 -y
conda activate yolo_export
pip install ultralytics onnx onnxsim "protobuf<5,>=4.21"
```

Export FP32 ONNX:
```
python - <<'PY'
from ultralytics import YOLO
model = YOLO("/home/fry/projects/Project_Song/weights/yolo.pt")
model.export(format="onnx", imgsz=640, dynamic=False, half=False, simplify=True, opset=17)
PY
```

## TensorRT Engine Generation
FP32 engine:
```
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/fry/projects/Project_Song/weights/yolo.onnx \
  --saveEngine=/home/fry/projects/Project_Song/outputs/trt_engines/yolo_fp32.plan \
  --memPoolSize=workspace:4096
```

FP16 engine (if needed):
```
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/fry/projects/Project_Song/weights/yolo.onnx \
  --saveEngine=/home/fry/projects/Project_Song/outputs/trt_engines/yolo_fp16.plan \
  --fp16 --memPoolSize=workspace:4096
```

## Runtime Commands
Split-screen demo (FP32 TRT by default):
```
cd /home/fry/projects/Project_Song
bash scripts/04_demo_split_screen.sh
```

Enable GPU preprocessing in TRT backend:
```
TRT_GPU_PREPROC=1 bash scripts/04_demo_split_screen.sh
```

Enable per-stage timing:
```
TRT_PROFILE=1 TRT_PROFILE_EVERY=30 bash scripts/04_demo_split_screen.sh
```

## Notes
- `tensorrt` is installed system-wide; Python bindings are accessed from the runtime env.
- If `trtexec` is not found in PATH, use `/usr/src/tensorrt/bin/trtexec`.
