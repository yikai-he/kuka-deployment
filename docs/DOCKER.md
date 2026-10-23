# Phase 1 offline perception container

This image packages the existing recorded RGB-D replay pipeline. It does not add
ROS, live camera acquisition, robot control, policy inference, a network service,
or GUI forwarding. Docker Compose is intentionally not used because Phase 1 has a
single offline process and exposes no ports.

## Compatibility baseline

- Ubuntu 22.04 userspace on `nvidia/cuda:12.1.1-devel-ubuntu22.04`
- Python 3.11 from the checksum-verified Miniconda 23.5.2 installer
- PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, torchaudio 2.4.1+cu121
- PyTorch3D 0.7.8 build `py311_cu121_pyt241`
- nvdiffrast commit `729261dc64c4241ea36efda84fbf532cc8b425b8`
  (the commit recorded for tag v0.3.3)
- FoundationPose commit `a1b694b83e633c2cb6115b9063d940a687759392`
- Direct Python dependencies pinned in `docker/requirements-gpu.txt`

The default `TORCH_CUDA_ARCH_LIST` is `7.5;8.6`, producing native CUDA code for both
the Tesla T4 (SM 7.5) and RTX 3060 (SM 8.6). `MAX_JOBS=2` limits PyTorch extension
compilation, and the FoundationPose C++ build also uses two parallel jobs. The saved
experimental baseline was measured on a T4; compatibility with SM 8.6 does not make
the RTX 3060 numerically or performance-equivalent, so it requires its own regression
run.

### Native extension architecture handling

- **nvdiffrast:** version 0.3.3 packages sources and JIT-compiles its CUDA plugin on
  first use. Its pinned source clears `TORCH_CUDA_ARCH_LIST`, so the image applies
  the explicit patch `docker/patches/nvdiffrast-preserve-cuda-arch-list.patch` and
  compiles the plugin during the image build into `/opt/torch_extensions`. PyTorch's
  extension builder receives `7.5;8.6` and `MAX_JOBS=2`.
- **FoundationPose `mycpp`:** the pinned CMake project contains C++/pybind11/OpenMP
  sources but no CUDA language or `.cu` source. `TORCH_CUDA_ARCH_LIST` does not apply;
  its build is limited with `cmake --build --parallel 2`.
- **PyTorch3D:** the exact `0.7.8=py311_cu121_pyt241` binary is installed rather than
  built in this image. Therefore the Docker build cannot pass architecture flags to
  it. Its internal fatbin targets must be inspected with CUDA binary tools or tested
  on both GPUs; this remains a static-verification gap.
- **PyTorch, torchvision, and torchaudio:** installed as official cu121 binaries;
  they are not extension builds performed by this Dockerfile.

## Explicit FoundationPose dtype patch

The saved notebook first failed in `compute_crop_window_tf_batch()` with a
`float != double` matrix multiplication. Its successful runs occurred only after
adding this conversion before projection:

```python
pts = pts.to(dtype=K.dtype, device=K.device)
```

The application adapter now passes float32 model arrays, but the notebook did not
reset FoundationPose and prove a successful modular run against a clean checkout.
For reproducibility, the image applies the recorded one-line change from
`docker/patches/foundationpose-utils-dtype.patch` after verifying that it applies
cleanly to the pinned revision. The image build will fail if upstream context
changes. This is an explicit modification of the external checkout, not an
unrecorded claim that the pinned FoundationPose commit is clean.

## Build on a cloud server

The build host needs Docker Engine, outbound HTTPS access, enough disk/RAM for CUDA
extension compilation, and Linux x86-64. `nvcc` can normally compile explicit SM
targets without a GPU, and nvdiffrast's build path does not intentionally create a
CUDA context. However, a complete no-GPU image build has not yet been executed for
this repository; confirm that its extension import/link step succeeds on the chosen
cloud builder. Even a successful no-GPU build cannot validate kernel loading, CUDA
driver compatibility, rasterization, or inference.

```bash
cd /path/to/kuka-deployment
docker build --pull \
  --build-arg 'TORCH_CUDA_ARCH_LIST=7.5;8.6' \
  --tag kuka-offline-perception:phase1-t4-rtx3060 .
```

The build downloads public source and package artifacts but no private model/data
assets. Do not remove either architecture from a shared image intended for both GPU
types. The image targets Linux x86-64 because the pinned PyTorch3D artifact is
x86-64.

The base image does not make the host driver optional. The deployment computer must
have an NVIDIA driver new enough for CUDA 12.1 and NVIDIA Container Toolkit configured
for Docker. Validate the host before perception:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

The toolkit supplies host GPU devices and driver libraries to the container; do not
install an NVIDIA kernel driver inside this image.

## Publish to GitHub Container Registry

Choose a lowercase GitHub user or organization name and an immutable version tag.
Create a GitHub personal access token with `write:packages` (and `read:packages` for
subsequent authenticated pulls; a private repository may require additional access).
Do not put the token directly on the command line or in the image.

```bash
export GHCR_OWNER=your-github-user-or-organization
export GHCR_IMAGE=ghcr.io/${GHCR_OWNER}/kuka-offline-perception
export IMAGE_VERSION=phase1-cu121-t4-rtx3060

printf '%s' "${CR_PAT}" | docker login ghcr.io \
  --username "${GHCR_OWNER}" --password-stdin

docker tag kuka-offline-perception:phase1-t4-rtx3060 \
  "${GHCR_IMAGE}:${IMAGE_VERSION}"
docker tag kuka-offline-perception:phase1-t4-rtx3060 \
  "${GHCR_IMAGE}:phase1"

docker push "${GHCR_IMAGE}:${IMAGE_VERSION}"
docker push "${GHCR_IMAGE}:phase1"
```

Prefer deploying the immutable version tag or, after publication, its digest. The
`phase1` tag is a movable convenience tag. Publishing is a separate explicit action;
the Dockerfile performs no registry login or push.

## Pull on another GPU machine

For a private image, authenticate first using a token with `read:packages`. Public
GHCR images can be pulled anonymously. Then pull the immutable tag:

```bash
export GHCR_OWNER=your-github-user-or-organization
export GHCR_IMAGE=ghcr.io/${GHCR_OWNER}/kuka-offline-perception
export IMAGE_VERSION=phase1-cu121-t4-rtx3060

printf '%s' "${CR_PAT}" | docker login ghcr.io \
  --username "${GHCR_OWNER}" --password-stdin
docker pull "${GHCR_IMAGE}:${IMAGE_VERSION}"
docker image inspect "${GHCR_IMAGE}:${IMAGE_VERSION}" --format '{{index .RepoDigests 0}}'
```

Substitute `"${GHCR_IMAGE}:${IMAGE_VERSION}"` for the local image name in the GPU
check, replay, and regression commands below. Runtime data, configuration, weights,
CAD geometry, baselines, and output remain bind mounts and are not published.

## CPU-only unit tests

The normal unit tests use mocks and do not need GPU or private assets. Override
the replay entry point:

```bash
docker run --rm --network none \
  --entrypoint python \
  kuka-offline-perception:phase1 \
  -m pytest -q -m "not gpu_integration" /workspace/kuka-deployment/tests
```

## Check GPU availability

```bash
docker run --rm --gpus all --network none \
  --entrypoint python \
  kuka-offline-perception:phase1 \
  -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
```

This checks visibility only; it is not an inference validation.

## Runtime mounts and effective configuration

Mount datasets, configuration, weights, and meshes read-only. Mount only the
result directory read-write. The YAML paths must use container paths, for example:

```yaml
models:
  yolo_weights: /assets/yolo/red_cube_yoloseg_best.pt
  mesh: /assets/cad/red_cube.obj
  foundationpose_repository: /opt/FoundationPose
  foundationpose_weights: /assets/foundationpose
```

Do not modify the tracked profile merely to set machine-specific paths. Create an
effective YAML outside the repository, preserve it with the results, and mount it
at `/config/perception.yaml`.

## Offline replay

The output directory should be new or empty. No ports, display socket, host network,
USB device, or privileged mode are required.

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/absolute/path/episode_000,dst=/data/episode,readonly \
  --mount type=bind,src=/absolute/path/perception_v7_effective.yaml,dst=/config/perception.yaml,readonly \
  --mount type=bind,src=/absolute/path/red_cube_yoloseg_best.pt,dst=/assets/yolo/red_cube_yoloseg_best.pt,readonly \
  --mount type=bind,src=/absolute/path/red_cube.obj,dst=/assets/cad/red_cube.obj,readonly \
  --mount type=bind,src=/absolute/path/foundationpose_weights,dst=/assets/foundationpose,readonly \
  --mount type=bind,src=/absolute/path/new_replay_output,dst=/output \
  kuka-offline-perception:phase1 \
  --input /data/episode \
  --config /config/perception.yaml \
  --output /output \
  --save-reproducibility-manifest
```

If the runtime user cannot write cache files, add a dedicated writable cache mount;
do not make asset or dataset mounts writable.

## GPU integration regression

The baseline directory must contain `poses.csv` and `poses.npy`. Use a fresh empty
output directory. The example uses the saved modular baseline; use `legacy` only
for the legacy CSV schema.

```bash
docker run --rm --gpus all --network none \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  --env KUKA_GPU_REGRESSION=1 \
  --env KUKA_GPU_DATASET=/data/episode \
  --env KUKA_GPU_CONFIG=/config/perception.yaml \
  --env KUKA_GPU_BASELINE=/baseline \
  --env KUKA_GPU_BASELINE_KIND=modular \
  --env KUKA_GPU_OUTPUT=/output \
  --env KUKA_GPU_TRANSLATION_TOLERANCE_M=0.005 \
  --env KUKA_GPU_ROTATION_TOLERANCE_DEG=2.0 \
  --env KUKA_GPU_FAIL_ON_DEVIATION=1 \
  --mount type=bind,src=/absolute/path/episode_000,dst=/data/episode,readonly \
  --mount type=bind,src=/absolute/path/perception_v7_effective.yaml,dst=/config/perception.yaml,readonly \
  --mount type=bind,src=/absolute/path/red_cube_yoloseg_best.pt,dst=/assets/yolo/red_cube_yoloseg_best.pt,readonly \
  --mount type=bind,src=/absolute/path/red_cube.obj,dst=/assets/cad/red_cube.obj,readonly \
  --mount type=bind,src=/absolute/path/foundationpose_weights,dst=/assets/foundationpose,readonly \
  --mount type=bind,src=/absolute/path/saved_modular_baseline,dst=/baseline,readonly \
  --mount type=bind,src=/absolute/path/new_gpu_test_output,dst=/output \
  kuka-offline-perception:phase1 \
  -m pytest -q -s /workspace/kuka-deployment/tests/test_gpu_integration.py -m gpu_integration
```

The test checks 429 serialized poses, dtype observations, frame correspondence,
validity, tracking/recovery events, and pose tolerances. Preserve the entire output,
especially `preflight.json`, `reproducibility_manifest.json`, and the comparison
JSON/Markdown.

## Standalone comparison

To compare already-produced result directories without GPU access:

```bash
docker run --rm --network none \
  --entrypoint python \
  --mount type=bind,src=/absolute/path/saved_baseline,dst=/baseline,readonly \
  --mount type=bind,src=/absolute/path/new_replay_output,dst=/candidate,readonly \
  --mount type=bind,src=/absolute/path/new_comparison,dst=/comparison \
  kuka-offline-perception:phase1 \
  /workspace/kuka-deployment/scripts/compare_perception_runs.py \
  --legacy /baseline --modular /candidate --output /comparison \
  --translation-tolerance-m 0.005 --rotation-tolerance-deg 2.0
```

Use `--legacy` only for a legacy-schema baseline. If both inputs are modular, invoke
the comparison API or adapt the command after confirming the expected CLI schema.
Do not reinterpret a schema mismatch as an inference regression.

## Validation boundary

A successful image build and CPU test run do not validate FoundationPose, CUDA
rasterization, private weights, or the 429-frame results. Those remain unverified
until the GPU test completes on the intended hardware with the preserved assets.
