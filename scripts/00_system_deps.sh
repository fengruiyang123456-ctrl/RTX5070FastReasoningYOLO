#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  git curl wget build-essential cmake pkg-config \
  python3 python3-venv python3-pip \
  ffmpeg v4l-utils \
  libgl1 libglib2.0-0
