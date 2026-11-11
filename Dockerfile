FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        git build-essential ca-certificates curl \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["georl-train"]
CMD ["experiment=_smoke"]
