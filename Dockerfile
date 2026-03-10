# Single-stage build: runtime with ComfyUI + custom nodes
# Models are downloaded at startup (not baked into the image) to keep image small
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# System packages (from base.Dockerfile)
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10 python3.10-venv python3.10-distutils -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Python 3.10 + pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# PyTorch + xformers (CUDA 12.8)
RUN pip install --no-cache-dir torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128

# All other pip dependencies (consolidated into one layer)
# flash_attn pre-built for CUDA 12.8 + PyTorch 2.10 + Python 3.10
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.7.16
RUN pip install --no-cache-dir -U wheel setuptools packaging ninja psutil \
    misaki[en] "huggingface_hub[hf_transfer]" runpod websocket-client librosa && \
    pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-Manager && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-KJNodes && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-MelBandRoFormer && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt && \
    find /ComfyUI -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

COPY . .
RUN chmod +x /entrypoint.sh

ENV RUNPOD_PING_INTERVAL=3000

CMD ["/entrypoint.sh"]
