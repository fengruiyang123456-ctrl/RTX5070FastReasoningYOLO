项目文件说明
==========

项目根目录
----------
- `README.md`：项目概览、使用流程、运行步骤与注意事项。
- `.gitignore`：忽略虚拟环境、输出产物、模型权重等不应入库的文件。

assets/
--------
- `assets/videos/`：可选测试视频目录。
- `assets/images/`：可选测试图片目录。

weights/
---------
- `weights/yolo.pt`：PyTorch 基线权重（需要你自行放置真实文件）。
- `weights/yolo.onnx`：ONNX 模型（由 `scripts/02_export_onnx.sh` 导出）。

outputs/
---------
- `outputs/benches/`：基准测试生成的 CSV/JSON 指标输出。
- `outputs/logs/`：预留日志输出。
- `outputs/trt_engines/`：TensorRT 引擎缓存（可选）。

env/
-----
- `env/requirements.txt`：pip 依赖清单。
- `env/environment.yml`：conda 依赖清单（可选）。

scripts/
---------
- `scripts/00_system_deps.sh`：Ubuntu 系统依赖安装。
- `scripts/01_create_env.sh`：创建虚拟环境并安装 Python 依赖。
- `scripts/02_export_onnx.sh`：从 `.pt` 导出固定输入尺寸 ONNX（FP16）。
- `scripts/03_benchmark_all.sh`：一键跑基准（Torch FP32 + ORT FP16；检测到 TRT 引擎则追加）。
- `scripts/04_demo_split_screen.sh`：同屏对比 demo（基线 vs 优化）。

src/common/
-----------
- `src/common/config.py`：统一配置（输入尺寸、阈值、路径、类别名）。
- `src/common/timer.py`：计时器，支持 CUDA event 计时。
- `src/common/stats.py`：统计平均/95/99 延迟与 FPS，并输出 JSON/CSV。
- `src/common/video_io.py`：统一视频/摄像头读取与帧迭代。
- `src/common/viz.py`：绘制检测框与指标文字。
- `src/common/preprocess.py`：letterbox + 归一化/转张量。
- `src/common/postprocess.py`：YOLOv8 风格后处理 + NMS。

src/backends/
-------------
- `src/backends/torch_fp32.py`：PyTorch FP32 推理（基线）。
- `src/backends/torch_fp16.py`：PyTorch FP16 推理（可选）。
- `src/backends/ort_fp16.py`：ONNX Runtime CUDA FP16 推理。
- `src/backends/trt_fp16.py`：TensorRT 后端占位（未实现，引导报错）。

src/apps/
---------
- `src/apps/run_camera.py`：单后端实时推理（排障用）。
- `src/apps/demo_compare.py`：双后端同屏对比 demo。
- `src/apps/benchmark.py`：基准测试脚本，输出 FPS/延迟/P95/P99/显存。

检查要点
--------
- `weights/yolo.pt` 和 `weights/yolo.onnx` 目前是空文件，运行会失败；需要放入真实权重后再导出 ONNX。
- `src/backends/trt_fp16.py` 目前是 stub，会主动抛错提示未实现；这是预期设计。
- ORT 后处理假设 YOLOv8 输出格式；如果你用的是不同结构或自定义导出，需要调整 `src/common/postprocess.py`。
- 计时与显存统计依赖 CUDA 可用；CPU 环境会显示为 0 或使用 CPU 计时。
