Ubuntu 环境配置教程（从 RTX 5070 驱动开始）
=========================================

目标
----
- 从 0 配置 Ubuntu 环境，支持 RTX 5070 / 5070 Ti（Blackwell / sm_120）
- 跑通 PyTorch FP32 与 ONNX Runtime FP16
- 可选 TensorRT FP16

说明
----
- 本文档替换旧版教程，已从“驱动安装”开始重新整理。
- 以下以 Ubuntu 22.04 为例。

1) 系统基础依赖
--------------
```bash
sudo apt update
sudo apt install -y \
  git curl wget build-essential cmake pkg-config \
  python3 python3-venv python3-pip \
  ffmpeg v4l-utils \
  libgl1 libglib2.0-0
```

2) 安装 RTX 5070 驱动（推荐 open 分支）
---------------------------------------
说明：RTX 5070 建议使用 570-open 或 575-open 这一类 *open kernel* 驱动。

```bash
sudo apt update
sudo apt install -y nvidia-driver-575-open
sudo reboot
```

重启后验证：
```bash
nvidia-smi
```

3) Python 虚拟环境
------------------
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

4) PyTorch（必须 cu128 或更高）
-------------------------------
RTX 5070 属于 Blackwell（sm_120），必须用 CUDA 12.8+ 轮子。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

5) ONNX Runtime GPU
-------------------
```bash
pip install onnxruntime-gpu
pip install opencv-python numpy ultralytics pandas
```

6) （可选）TensorRT
-------------------
说明：如需使用 TensorRT FP16，请在与 CUDA 12.x 兼容的版本下安装 TensorRT，
并在项目中生成 engine 缓存到 `outputs/trt_engines/`。

7) 项目依赖一键安装
------------------
如果你希望和仓库版本一致，直接执行：
```bash
bash scripts/01_create_env.sh
```

8) 权重与 ONNX 导出
-------------------
1) 准备权重文件 `weights/yolo.pt`  
2) 导出 ONNX：
```bash
bash scripts/02_export_onnx.sh
```

9) 跑基准与 Demo
----------------
```bash
bash scripts/03_benchmark_all.sh
bash scripts/04_demo_split_screen.sh
```
