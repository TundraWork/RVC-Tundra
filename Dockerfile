# syntax=docker/dockerfile:1.3-labs
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV NLTK_DATA /usr/share/nltk_data

RUN apt-get update && \
  apt-get install -y python-is-python3 python3-pip git build-essential python3-dev ffmpeg --no-install-recommends && \
  pip install pdm && \
  pdm config python.use_venv false && \
  mkdir /workspace && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml pdm.lock lib/tools/model_fetcher.py /workspace

RUN pdm install && \
  pdm run python -m nltk.downloader punkt -d "$NLTK_DATA" && \
  pdm run python lib/tools/model_fetcher.py

COPY . /workspace

