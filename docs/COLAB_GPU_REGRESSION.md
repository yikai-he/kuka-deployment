# Phase 1.2 Google Colab GPU regression

This procedure runs the preserved legacy v7 runner and modular Phase 1.1 pipeline
against the same recorded episode. It does not establish real-robot readiness.

## Reference environment

Values below were read from `FoundationPose_KUKA_v7_YOLOSeg_drive_batch(4).ipynb`:

| Component | Reference value |
|---|---|
| Python | 3.11 |
| FoundationPose | `a1b694b83e633c2cb6115b9063d940a687759392` |
| PyTorch | `2.4.1+cu121` |
| torchvision | `0.19.1` |
| torchaudio | `2.4.1` |
| PyTorch3D | `0.7.8=py311_cu121_pyt241` |
| nvdiffrast | `v0.3.3` |
| Ultralytics | `8.4.138` |
| MKL | `2024.0.0` |
| CUDA architecture used for builds | `7.5` |

The notebook searches `/usr/local/cuda-12.8` and `/usr/local/cuda`, but the exact
Colab NVIDIA driver and installed toolkit build are runtime-dependent and therefore
**UNKNOWN until the session starts**. FoundationPose's remaining requirements come
from the pinned checkout's `requirements.txt`; their resolved versions are not fully
pinned by the notebook.

## 1. Start and identify the runtime

Select **Runtime → Change runtime type → NVIDIA GPU**, then run:

```bash
!nvidia-smi
!/usr/local/cuda/bin/nvcc --version
```

Record both outputs with the experiment. If the GPU compute capability is not 7.5,
replace `TORCH_CUDA_ARCH_LIST=7.5` with the actual capability before native builds.

## 2. Clone and pin this repository

Replace `<KUKA_COMMIT>` with an reviewed commit hash—never use an unrecorded moving
branch for a formal comparison:

```bash
!git clone https://github.com/yikai-he/kuka-deployment.git /content/kuka-deployment
%cd /content/kuka-deployment
!git checkout --detach <KUKA_COMMIT>
!git rev-parse HEAD
```

## 3. Mount Drive and define private inputs

Keep these paths in the notebook/session, not application source:

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
DRIVE_INPUT = Path("/content/drive/MyDrive/<YOUR_PRIVATE_INPUT_DIRECTORY>")
EPISODE = DRIVE_INPUT / "<EPISODE_DIRECTORY>"
YOLO_WEIGHTS = DRIVE_INPUT / "red_cube_yoloseg_best.pt"
CUBE_MESH = DRIVE_INPUT / "red_cube.obj"
FP_WEIGHTS = DRIVE_INPUT / "foundationpose_weights"
LEGACY_RUNNER = DRIVE_INPUT / "run_kuka_sequence.py"
RESULT_ROOT = Path("/content/drive/MyDrive/<YOUR_RESULT_DIRECTORY>")
for required in (EPISODE, YOLO_WEIGHTS, CUBE_MESH, FP_WEIGHTS, LEGACY_RUNNER):
    assert required.exists(), required
for name, value in {
    "EPISODE": EPISODE, "YOLO_WEIGHTS": YOLO_WEIGHTS, "CUBE_MESH": CUBE_MESH,
    "FP_WEIGHTS": FP_WEIGHTS, "LEGACY_RUNNER": LEGACY_RUNNER,
    "RESULT_ROOT": RESULT_ROOT,
}.items():
    os.environ[name] = str(value)
```

The episode needs matching `rgb/<id>.png`, `depth/<id>.png`, `camera.json`, and—for
the v7 profile—`masks/<first-id>.png`.

## 4. Prepare pinned FoundationPose

Follow the legacy notebook cells for Miniconda, build packages, and `mycpp`. The
equivalent core commands are shown here for auditability; run them in separate Colab
cells so failures are visible:

```bash
!git clone https://github.com/NVlabs/FoundationPose.git /content/FoundationPose
!git -C /content/FoundationPose checkout --detach a1b694b83e633c2cb6115b9063d940a687759392
!git -C /content/FoundationPose submodule update --init --recursive
```

Install Miniconda at `/opt/conda`, create `foundationpose` with Python 3.11, and use
the notebook's exact system packages (`build-essential`, Boost system/program
options, Eigen3). Then install:

```bash
!/opt/conda/bin/conda install -y -n foundationpose \
  -c pytorch3d -c pytorch -c nvidia -c iopath -c fvcore -c conda-forge \
  pytorch3d=0.7.8=py311_cu121_pyt241 iopath fvcore ninja cmake pybind11 mkl=2024.0.0
!/opt/conda/envs/foundationpose/bin/python -m pip install --force-reinstall --no-cache-dir \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
!/opt/conda/envs/foundationpose/bin/python -m pip install --no-build-isolation \
  git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3
!/opt/conda/envs/foundationpose/bin/python -m pip install \
  -r /content/FoundationPose/requirements.txt
!/opt/conda/envs/foundationpose/bin/python -m pip install ultralytics==8.4.138
```

Build `/content/FoundationPose/mycpp` using the environment variables and CMake
arguments in the reference notebook. Copy or link the two pretrained FoundationPose
weight directories into the location expected by that pinned checkout. Do not let
the application download weights implicitly.

Install this package into the same environment without replacing the pinned GPU
stack:

```bash
!/opt/conda/envs/foundationpose/bin/python -m pip install -e /content/kuka-deployment --no-deps
```

Before inference, record versions and asset hashes:

```bash
!/opt/conda/envs/foundationpose/bin/python -c \
  "import torch,ultralytics; print(torch.__version__, torch.version.cuda, ultralytics.__version__)"
!sha256sum "${YOLO_WEIGHTS}" "${CUBE_MESH}"
```

## 5. Prepare the modular configuration

```python
import yaml

source = Path("/content/kuka-deployment/configs/perception_v7_red_cube.yaml")
config = yaml.safe_load(source.read_text())
config["models"]["yolo_weights"] = str(YOLO_WEIGHTS)
config["models"]["mesh"] = str(CUBE_MESH)
config["models"]["foundationpose_repository"] = "/content/FoundationPose"
config["models"]["foundationpose_weights"] = str(FP_WEIGHTS)
effective = Path("/content/perception_v7_effective.yaml")
effective.write_text(yaml.safe_dump(config, sort_keys=False))
```

## 6. Run both pipelines on the same episode

Legacy:

```bash
!/opt/conda/envs/foundationpose/bin/python "${LEGACY_RUNNER}" \
  --data_dir "${EPISODE}" \
  --out_dir /content/legacy_v7_result \
  --mesh_path "${CUBE_MESH}" \
  --fp_root /content/FoundationPose \
  --yolo_model "${YOLO_WEIGHTS}" \
  --allow_initial_mask_fallback --no_vis
```

Modular:

```bash
!/opt/conda/envs/foundationpose/bin/python \
  /content/kuka-deployment/scripts/replay_perception.py \
  --input "${EPISODE}" \
  --config /content/perception_v7_effective.yaml \
  --output /content/modular_v7_result
```

Use a fresh output directory for each run. Do not use different frames, masks,
intrinsics, meshes, weights, or iteration counts.

## 7. Compare on CPU and save reports

```bash
!python3 /content/kuka-deployment/scripts/compare_perception_runs.py \
  --legacy /content/legacy_v7_result \
  --modular /content/modular_v7_result \
  --output /content/comparison
```

Optional tolerances annotate counts but never create an automatic overall verdict:

```text
--translation-tolerance-m 0.005 --rotation-tolerance-deg 2.0
```

Copy `legacy_v7_result`, `modular_v7_result`, `comparison`, the effective YAML,
commit hashes, version output, and asset SHA-256 manifest to `RESULT_ROOT`.

## Comparison interpretation

Numerical pose errors use frames valid in both runs. Validity and recovery statistics
retain invalid frames. Frame-set mismatches are reported. Rotation errors minimize
over the 24 proper signed-permutation rotations of an ideal cube whose symmetry axes
align with the CAD object axes. If the CAD origin, axes, or geometry differ from an
ideal centered cube, that metric must be revised before drawing conclusions.

## Troubleshooting

- **No GPU / `torch.cuda.is_available()` false:** restart with an NVIDIA runtime;
  do not fall back to CPU inference.
- **CUDA extension build failure:** compare `nvidia-smi`, `nvcc`, PyTorch CUDA, GPU
  capability, compiler paths, and `TORCH_CUDA_ARCH_LIST` with the recorded notebook.
- **`mycpp` import failure:** confirm the build completed and its build directory is
  on `PYTHONPATH`, as in the reference notebook wrapper.
- **Missing weights:** supply them explicitly and verify hashes; do not download an
  arbitrary replacement.
- **Different frames:** stop and repair the episode/output pairing before comparing
  trajectory metrics.
- **Codec failure:** rerun without visualization; video is not required for parity.
