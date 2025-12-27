下面我按你们这份《ProjectDetial.markdown》里的目标（Baseline：PyTorch FP32；Optimized：FP16 + ONNX Runtime / TensorRT；必须量化 FPS/延迟P95/显存；摄像头 Demo 同屏对比）来给一套**代码结构**和**白板 Ubuntu 的环境配置**方案。   

---

## 1) 推荐代码结构（直接照这个建仓库）

> 目标：同一套输入/预处理/后处理/可视化/指标统计，底下切换 backend（torch / onnxruntime / tensorrt）实现公平对比。

```
project/
  README.md
  .gitignore

  assets/
    videos/                 # 可选：测试视频
    images/                 # 可选：测试图片

  weights/
    yolo.pt                 # baseline 权重（例如 yolov8n.pt）
    yolo.onnx               # 导出的 onnx（固定输入尺寸！）

  outputs/
    benches/                # 每次跑出来的 csv/json 指标
    logs/
    trt_engines/            # TensorRT engine 缓存（可选）

  env/
    requirements.txt
    environment.yml         # 你们喜欢 conda 就用这个

  scripts/
    00_system_deps.sh       # apt 装系统依赖
    01_create_env.sh        # 创建 python 环境 + 装依赖
    02_export_onnx.sh       # 从 pt 导出 onnx（固定尺寸）
    03_benchmark_all.sh     # 一键跑 baseline vs optimized 并输出表
    04_demo_split_screen.sh # 同屏对比 demo

  src/
    common/
      config.py             # 输入尺寸、阈值、设备、路径等统一配置
      timer.py              # 统一计时（CPU + CUDA event）
      stats.py              # avg/p95/p99 统计、写 csv/json
      video_io.py           # webcam/video 读取（解耦可做异步）
      viz.py                # 画框/叠字（FPS/延迟/显存）
      preprocess.py         # resize/normalize/letterbox（固定输入！）
      postprocess.py        # NMS 等（尽量与各 backend 对齐）

    backends/
      torch_fp32.py         # Baseline：PyTorch FP32（同步串行）:contentReference[oaicite:5]{index=5}
      torch_fp16.py         # 可选：PyTorch AMP/FP16
      ort_fp16.py           # Optimized：ONNX Runtime CUDA + FP16 :contentReference[oaicite:6]{index=6}
      trt_fp16.py           # Optimized：TensorRT FP16（时间允许再上）:contentReference[oaicite:7]{index=7}

    apps/
      run_camera.py         # 跑实时摄像头推理（单 backend）
      demo_compare.py        # 同屏对比：左 baseline 右 optimized :contentReference[oaicite:8]{index=8}
      benchmark.py           # 跑 N 次、warmup、输出 benches 表 :contentReference[oaicite:9]{index=9}
```

**你们最终路演最有用的 3 个入口：**

* `scripts/03_benchmark_all.sh`：一键生成 FPS / 平均延迟 / P95 / P99 / 显存占用对比表（对应文档“必须量化”）
* `scripts/04_demo_split_screen.sh`：同屏 demo（左 Baseline PyTorch FP32，右 Optimized FP16 + 引擎）
* `src/apps/run_camera.py`：单路跑通，排障用（摄像头/模型/环境哪个坏一眼看出）

---

## 2) 白板 Ubuntu 环境配置（RTX 5070 重点避坑版）

### 2.1 系统依赖（必装）

```bash
sudo apt update
sudo apt install -y git curl wget build-essential cmake pkg-config \
  python3 python3-venv python3-pip \
  ffmpeg v4l-utils \
  libgl1 libglib2.0-0
```

> `v4l-utils` 用来确认摄像头设备；`ffmpeg`/`libgl1` 常见于 OpenCV 显示/视频读取。

---

### 2.2 NVIDIA 驱动（RTX 5070：优先用 *-open 分支）

有资料显示 **RTX 5070 需要 570-open 或 575-open** 这类 *open kernel* 驱动分支（非 open 的分支可能不工作或更容易翻车）。([Ask Ubuntu][1])

**安装方式（Ubuntu 22.04 示例）：**

```bash
sudo apt update
sudo apt install -y nvidia-driver-575-open
sudo reboot
```

重启后验证：

```bash
nvidia-smi
```

---

### 2.3 PyTorch（RTX 50 系 / sm_120：务必用 CUDA 12.8+ 的轮子）

RTX 5070/5070Ti 属于 Blackwell（常见报错里叫 **sm_120**），旧的 PyTorch 轮子会提示“不兼容 sm_120”。([PyTorch Forums][2])
PyTorch 社区信息提到 **PyTorch 2.7.0 的 CUDA 12.8（cu128）轮子已加入 Blackwell 支持**。([PyTorch Forums][3])

推荐直接走官方“Start Locally”对应的 **cu128** 安装方式。([PyTorch][4])
（命令一般长这样：）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# cu128（示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

### 2.4 ONNX Runtime GPU（建议作为你们的“第一优先优化后端”）

你们文档里把优化引擎优先级放在 ONNX Runtime（CUDA + FP16）前面，很对：更稳、集成成本低。

关键兼容性点（别踩坑）：

* ORT **CUDA 12.x 构建**通常可兼容 **CUDA 12.x**（小版本兼容）([onnxruntime.ai][5])
* 但 **cuDNN 8.x 与 9.x 不互通**：PyTorch 版本用哪个 cuDNN 大版本，你 ORT 也要尽量对齐。([onnxruntime.ai][5])

安装（常见做法）：

```bash
pip install onnxruntime-gpu
pip install opencv-python numpy ultralytics pandas
```

> 如果后面遇到 “libcudnn.so 找不到 / 版本不对”，优先按 ORT 的 CUDA EP 文档去对齐 CUDA/cuDNN 大版本。([onnxruntime.ai][5])

---

### 2.5 TensorRT（可选：最后冲刺加分项）

文档明确“如时间允许”再上 TensorRT，我建议你们也这么排期。
TensorRT 官方支持矩阵会标明它对 CUDA 12.x 的兼容策略（例如单一构建兼容 CUDA 12.x）。([NVIDIA Docs][6])

**建议策略：**

* 第一阶段：Torch FP32 vs ORT FP16 先把表做出来（可交付）
* 第二阶段：再加 TRT FP16（引擎缓存到 `outputs/trt_engines/`）

---

## 3) 你们脚本应该怎么“一键跑出来可交付成果”

这和文档“交付物：可运行代码 + 一键脚本 + 性能对比表 + demo”完全对齐。

**建议你们最终对外只暴露这两个命令：**

```bash
bash scripts/03_benchmark_all.sh   # 输出 outputs/benches/ 对比表（FPS/延迟P95/显存）
bash scripts/04_demo_split_screen.sh  # 摄像头同屏对比 demo
```

---

如果你愿意，我可以按上面结构把**最小可跑通版本**（含：torch_fp32 / ort_fp16 / benchmark 输出 csv / 同屏 demo）每个文件应该写什么、主函数参数怎么设计，给你一套“照抄就能跑”的模板。

[1]: https://askubuntu.com/questions/1554489/rtx-5070-driver-problem?utm_source=chatgpt.com "RTX 5070 driver problem"
[2]: https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509?utm_source=chatgpt.com "NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120"
[3]: https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621?utm_source=chatgpt.com "When will sm120 support be available?"
[4]: https://pytorch.org/get-started/locally/?utm_source=chatgpt.com "Get Started"
[5]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html?utm_source=chatgpt.com "NVIDIA - CUDA | onnxruntime"
[6]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html?utm_source=chatgpt.com "Support Matrix — NVIDIA TensorRT Documentation"
