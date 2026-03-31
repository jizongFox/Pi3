FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Pi3

COPY . .

RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.26.4 pillow plyfile huggingface_hub safetensors && \
    pip install -e . --no-deps

ENTRYPOINT ["python", "scripts/docker_inference.py"]
