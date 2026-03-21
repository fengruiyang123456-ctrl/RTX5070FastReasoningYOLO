# Environment Setup

This folder intentionally keeps only two documents:

- `Environment.markdown`: environment preparation
- `Run.markdown`: exact run commands

## Local Python Environment

Recommended host environment:

- Ubuntu 22.04
- Python 3.10
- NVIDIA GPU with a working CUDA driver
- Optional: TensorRT installed on the host if you want to run TRT backends

Install system packages:

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  ffmpeg libgl1 libglib2.0-0
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r env/requirements.txt
```

Prepare model files:

```bash
cp /path/to/your/yolo.pt weights/yolo.pt
bash scripts/02_export_onnx.sh
```

Notes:

- `weights/yolo.pt` and `weights/yolo.onnx` must be real files before running benchmarks
- TensorRT is not installed by the Docker image below; TRT execution is expected on a host that already has TensorRT installed

## Docker Environment

The repository includes:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Build the image:

```bash
docker compose build
```

Open an interactive container:

```bash
docker compose run --rm app bash
```

Inside the container, Python dependencies are already installed during image build. You only need to place model files under `weights/`.

Verify GPU access inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Notes:

- The Docker workflow is intended for PyTorch and ONNX Runtime benchmarking
- TensorRT is intentionally not wired into the container in this version to keep the setup easier to reproduce
