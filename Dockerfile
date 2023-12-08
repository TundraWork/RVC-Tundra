# syntax=docker/dockerfile-upstream:1-labs
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV NLTK_DATA /usr/share/nltk_data

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

RUN <<-EOF
  set -x
  pdm run python -m nltk.downloader all -d "$NLTK_DATA"
  pdm run python lib/tools/model_fetcher.py
EOF
