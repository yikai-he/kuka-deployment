# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG FOUNDATIONPOSE_COMMIT=a1b694b83e633c2cb6115b9063d940a687759392
ARG NVDIFFRAST_COMMIT=729261dc64c4241ea36efda84fbf532cc8b425b8
ARG TORCH_CUDA_ARCH_LIST=7.5;8.6
ARG MINICONDA_INSTALLER=Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
ARG MINICONDA_SHA256=634d76df5e489c44ade4085552b97bebc786d49245ed1a830022b0b406de5817

ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=2 \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    PATH=/opt/conda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FOUNDATIONPOSE_ROOT=/opt/FoundationPose \
    PYTHONPATH=/opt/FoundationPose:/opt/FoundationPose/mycpp/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential ca-certificates cmake curl git \
       libboost-program-options-dev libboost-system-dev libeigen3-dev \
       libegl1 libgl1 libglib2.0-0 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --retry 3 \
      "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" \
      --output /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --strict \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda config --system --set auto_update_conda false \
    && python --version \
    && python -m pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt

RUN python -m pip install \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install --requirement /tmp/requirements-gpu.txt

# This exact prebuilt PyTorch3D package is the artifact used by the validated
# Python 3.11 / CUDA 12.1 / PyTorch 2.4.1 environment.
RUN curl --fail --location --retry 3 \
      https://conda.anaconda.org/pytorch3d/linux-64/pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2 \
      --output /tmp/pytorch3d.tar.bz2 \
    && tar -xjf /tmp/pytorch3d.tar.bz2 -C /opt/conda \
    && rm /tmp/pytorch3d.tar.bz2 \
    && python -c "import pytorch3d; assert pytorch3d.__version__ == '0.7.8'"

RUN git clone --filter=blob:none https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast \
    && git -C /opt/nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}" \
    && test "$(git -C /opt/nvdiffrast rev-parse HEAD)" = "${NVDIFFRAST_COMMIT}"

# Pinned nvdiffrast clears TORCH_CUDA_ARCH_LIST immediately before its JIT
# extension build. Apply an explicit patch so the requested fat binary targets
# are retained, then precompile the CUDA plugin into a shared image cache.
COPY docker/patches/nvdiffrast-preserve-cuda-arch-list.patch /tmp/nvdiffrast-preserve-cuda-arch-list.patch
RUN git -C /opt/nvdiffrast apply --check /tmp/nvdiffrast-preserve-cuda-arch-list.patch \
    && git -C /opt/nvdiffrast apply /tmp/nvdiffrast-preserve-cuda-arch-list.patch \
    && git -C /opt/nvdiffrast diff --check \
    && python -m pip install --no-build-isolation /opt/nvdiffrast \
    && mkdir -p "${TORCH_EXTENSIONS_DIR}" \
    && python -c "from nvdiffrast.torch import ops; ops._get_plugin()" \
    && chmod -R a+rwX "${TORCH_EXTENSIONS_DIR}" \
    && chmod 1777 "${TORCH_EXTENSIONS_DIR}"

RUN git clone --filter=blob:none https://github.com/NVlabs/FoundationPose.git /opt/FoundationPose \
    && git -C /opt/FoundationPose checkout --detach "${FOUNDATIONPOSE_COMMIT}" \
    && git -C /opt/FoundationPose submodule update --init --recursive \
    && test "$(git -C /opt/FoundationPose rev-parse HEAD)" = "${FOUNDATIONPOSE_COMMIT}"

# The patch is explicit because the saved successful notebook run contained the
# same dtype conversion. See docs/DOCKER.md; do not replace it with an unrecorded
# edit to the external checkout.
COPY docker/patches/foundationpose-utils-dtype.patch /tmp/foundationpose-utils-dtype.patch
RUN git -C /opt/FoundationPose apply --check /tmp/foundationpose-utils-dtype.patch \
    && git -C /opt/FoundationPose apply /tmp/foundationpose-utils-dtype.patch \
    && git -C /opt/FoundationPose diff --check \
    && python -m py_compile /opt/FoundationPose/Utils.py

RUN cmake -S /opt/FoundationPose/mycpp -B /opt/FoundationPose/mycpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=/opt/conda/bin/python \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /opt/FoundationPose/mycpp/build --parallel 2 \
    && python -c "import mycpp; print(mycpp.__file__)"

WORKDIR /workspace/kuka-deployment
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
RUN python -m pip install --no-deps . \
    && python -m pip check \
    && mkdir -p /output /assets /data /config /tmp/foundationpose-debug \
    && chmod 1777 /output /tmp/foundationpose-debug

# The image deliberately contains no datasets, weights, CAD models, ROS, camera
# drivers, robot interfaces, network service, or display configuration.
ENTRYPOINT ["python", "/workspace/kuka-deployment/scripts/replay_perception.py"]
CMD ["--help"]
