FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    WANDB_DIR=/cache/wandb \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG INSTALL_FLASH_ATTN=1
ARG BUILD_FLASH_ATTN_FROM_SOURCE=0
ARG FLASH_ATTN_WHEEL_PATH=/opt/vendor/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
ARG INSTALL_HOPPER_CUBLAS_FIX=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libegl1-mesa \
    libgl1 \
    libgl1-mesa-dev \
    libglew-dev \
    libglib2.0-0 \
    libglfw3 \
    libglu1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mesa-utils \
    ninja-build \
    openssh-client \
    openssh-server \
    patchelf \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system && \
    python -m pip install --upgrade pip setuptools wheel packaging ninja

RUN python -m pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url "${TORCH_INDEX_URL}"

COPY vendor /opt/vendor

RUN python -m pip install --no-cache-dir \
    /opt/vendor/transformers-openvla-oft \
    /opt/vendor/dlimp_openvla

# Dependency union for spatial-memory-diffusion, 3dcavla, and MemoryVLA.
# Conflicting pins prefer the OpenVLA-OFT / LIBERO training stack used here.
RUN python -m pip install --no-cache-dir \
    "accelerate>=0.25.0" \
    "draccus==0.8.0" \
    einops \
    "huggingface_hub==0.29.3" \
    json-numpy \
    jsonlines \
    matplotlib \
    "peft==0.11.1" \
    protobuf \
    rich \
    "sentencepiece==0.1.99" \
    "timm==0.9.10" \
    "tokenizers==0.19.1" \
    wandb \
    "tensorflow==2.15.0" \
    "tensorflow_datasets==4.9.3" \
    "tensorflow_graphics==2021.12.3" \
    "diffusers==0.30.3" \
    "imageio[ffmpeg]==2.37.0" \
    uvicorn \
    fastapi \
    flask \
    "mediapy==1.2.0" \
    "hydra-core==1.2.0" \
    easydict \
    thop \
    "bddl==1.0.1" \
    "future==0.18.2" \
    "cloudpickle==2.1.0" \
    "gym==0.25.2" \
    "robosuite==1.4.1" \
    "robomimic==0.2.0" \
    "numpy==1.26.4" \
    "opencv-python==4.11.0.86" \
    "mujoco==2.3.7" \
    "dm_control==1.0.14" \
    pyquaternion \
    pyyaml \
    rospkg \
    pexpect \
    h5py \
    traitlets \
    ipdb \
    ipython \
    modern_robotics \
    pillow \
    termcolor \
    requests \
    psutil \
    tqdm \
    onnxruntime \
    openexr \
    pydantic \
    gradio

RUN if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then \
        if [[ "${BUILD_FLASH_ATTN_FROM_SOURCE}" == "1" ]]; then \
            FLASH_ATTENTION_FORCE_BUILD=TRUE \
            python -m pip install --no-cache-dir --no-build-isolation "flash-attn==2.5.5"; \
        else \
            python -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL_PATH}"; \
        fi; \
    fi

RUN if [[ "${INSTALL_HOPPER_CUBLAS_FIX}" == "1" ]]; then \
        python -m pip install --no-cache-dir nvidia-cublas-cu12==12.4.5.8; \
    fi

WORKDIR /workspace/spatial-memory-diffusion

COPY . /workspace/spatial-memory-diffusion

RUN python -m pip install --no-cache-dir --no-deps -e .

RUN mkdir -p /cache/huggingface /cache/wandb /data /checkpoints /run/sshd /var/run/sshd /root/.ssh && \
    echo 'root:root123' | chpasswd && \
    chmod 700 /root/.ssh && \
    ssh-keygen -A && \
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?UsePAM.*/UsePAM no/' /etc/ssh/sshd_config && \
    sed -i 's|^#\?Subsystem[[:space:]]\+sftp.*|Subsystem sftp /usr/lib/openssh/sftp-server|' /etc/ssh/sshd_config

EXPOSE 22

CMD ["bash", "-lc", "echo 'root:root123' | chpasswd && mkdir -p /run/sshd /var/run/sshd /root/.ssh && chmod 700 /root/.ssh && if [[ ! -f /etc/ssh/ssh_host_rsa_key ]]; then ssh-keygen -A; fi && sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && sed -i 's/^#\\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && sed -i 's/^#\\?UsePAM.*/UsePAM no/' /etc/ssh/sshd_config && exec /usr/sbin/sshd -D"]
