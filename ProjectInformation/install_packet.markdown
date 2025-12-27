好，下面这份我**按“已经装好 Ubuntu 22.04 + 有 RTX 5070 实体卡”这个前提**来写，
目标是：**驱动 → CUDA → cuDNN → PyTorch / ONNX / TensorRT 能正常用**。
你可以**原样拷给队友执行**。

---

## 一、先确认系统是否识别到 RTX 5070（很关键）

```bash
lspci | grep -i nvidia
```

**正常输出示例**：

```
01:00.0 VGA compatible controller: NVIDIA Corporation ...
```

👉 如果这里**看不到 NVIDIA**，说明 BIOS / 硬件 / 插槽有问题，先别往下走。

---

## 二、安装 NVIDIA 官方显卡驱动（强烈推荐方式）

### 1️⃣ 禁用 nouveau（必须）

```bash
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

写入以下内容：

```conf
blacklist nouveau
options nouveau modeset=0
```

保存后执行：

```bash
sudo update-initramfs -u
sudo reboot
```

---

### 2️⃣ 使用 Ubuntu 官方 + NVIDIA 驱动仓库

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
```

自动检测推荐驱动：

```bash
ubuntu-drivers devices
```

你会看到类似：

```
driver   : nvidia-driver-550 (recommended)
```

👉 **5070 必须用 550+，别装 535 / 525**

安装推荐驱动：

```bash
sudo apt install -y nvidia-driver-550
sudo reboot
```

---

### 3️⃣ 验证驱动是否成功

```bash
nvidia-smi
```

**正确示例**：

```
NVIDIA-SMI 550.xx
GPU  Name        RTX 5070
CUDA Version: 12.x
```

![Image](https://global.discourse-cdn.com/nvidia/optimized/4X/8/5/0/850d7d658e4073af4a14cd20ccb9b0541765bc0a_2_690x331.jpeg)

![Image](https://i.sstatic.net/25QCZ.jpg)

👉 **看到这一步，显卡驱动已经 100% OK**

---

## 三、安装 CUDA（只装 Toolkit，不装驱动）

> ⚠️ **原则：驱动用 apt 装，CUDA 用官方 runfile / deb 装**

### 1️⃣ 安装 CUDA 12.4（推荐）

```bash
sudo apt install -y nvidia-cuda-toolkit
```

验证：

```bash
nvcc --version
```

输出类似：

```
Cuda compilation tools, release 12.x
```

---

### 2️⃣ 设置环境变量（一定要做）

```bash
nano ~/.bashrc
```

追加：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

生效：

```bash
source ~/.bashrc
```

---

## 四、安装 cuDNN（TensorRT / PyTorch 必须）

### 方式一（推荐，最稳）：apt 安装

```bash
sudo apt install -y libcudnn8 libcudnn8-dev
```

验证：

```bash
dpkg -l | grep cudnn
```

---

## 五、Python 环境（强烈建议用 Conda）

### 1️⃣ 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

一路回车 → yes → 重启终端

---

### 2️⃣ 创建项目环境

```bash
conda create -n rtx5070 python=3.10 -y
conda activate rtx5070
```

---

## 六、安装 PyTorch（GPU 版）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python - << EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

输出应为：

```
True
NVIDIA RTX 5070
```

---

## 七、ONNX Runtime（CUDA 推理）

```bash
pip install onnx onnxruntime-gpu
```

验证：

```bash
python - << EOF
import onnxruntime as ort
print(ort.get_available_providers())
EOF
```

应包含：

```
CUDAExecutionProvider
```

---

## 八、TensorRT（可选，但你们项目加分项）

```bash
sudo apt install -y tensorrt
pip install nvidia-tensorrt
```

验证：

```bash
trtexec --version
```

---

## 九、最终「一键体检」脚本（强烈建议发给队友）

```bash
nvidia-smi
nvcc --version
python - << EOF
import torch, onnxruntime
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("ONNX Providers:", onnxruntime.get_available_providers())
EOF
```

---

## 十、常见翻车点（一定提醒队友）

❌ **不要装多个 NVIDIA 驱动版本**
❌ **不要用 runfile 装驱动**
❌ **不要混用 conda cudatoolkit 和系统 CUDA**
❌ `nvidia-smi` 不通 → 后面全是白搭

---

## 给你一句实话（项目角度）

你们现在这个阶段：

> **能稳定跑 PyTorch FP16 + ONNX CUDA → 已经是合格作品**
> TensorRT 是加分项，不是必选项

如果你愿意，下一步我可以直接帮你出一份：

* ✅ **“5070 推理 Baseline vs FP16 对比 demo 结构”**
* ✅ **YOLO / 任意视觉模型的一键测速模板**
* ✅ **评委能看懂的性能对比表写法**

你只要说一句：**“下一步做 baseline demo”**
