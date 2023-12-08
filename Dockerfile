# syntax=docker/dockerfile-upstream:1-labs
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN <<-EOF
  set -x
  apt-get update
  apt-get install python-is-python3 python3-pip git build-essential python3-dev
  pip install pdm --break-system-packages
  mkdir /workspace
  apt-get clean
  rm -rf /var/lib/apt/lists/*
EOF

WORKDIR /workspace
COPY pyproject.toml pdm.lock pdm.toml /workspace

RUN <<-EOF
  set -x
  pdm config python.use_venv false
  pdm install
EOF

COPY . /workspace

RUN pdm run python download.py
