# First offline deployment on `yxt-X99`

This procedure prepares the existing offline YOLO-Seg + FoundationPose replay on
the RTX 3060 workstation. It does not install ROS, start camera nodes, or control a
robot. The real GPU path remains unverified until the final opt-in test succeeds.

## Fixed compatibility baseline

- Host: Ubuntu 20.04, NVIDIA RTX 3060 12 GB (SM 8.6)
- Conda environment: `metaworld`, Python 3.11
- PyTorch `2.4.1+cu121`, NumPy `1.26.4`, PyTorch3D `0.7.8`
- NVDiffRast commit `729261dc64c4241ea36efda84fbf532cc8b425b8`
  (the commit recorded for v0.3.3)
- FoundationPose commit `a1b694b83e633c2cb6115b9063d940a687759392`
- Application GPU pins: `docker/requirements-gpu.txt`

The Docker files are the authoritative package/build record even when the existing
`metaworld` environment is used natively. Do not install or upgrade packages until
the environment inventory below has been saved and compared with those pins.

## 1. Clone and record the starting state

```bash
ssh yxt-X99
git clone git@github.com:yikai-he/kuka-deployment.git
cd kuka-deployment
git rev-parse HEAD
git status --short

conda activate metaworld
python --version
python -m pip freeze > /tmp/metaworld-before-kuka.txt
nvidia-smi
```

The repository should be clean. Preserve the commit hash and package inventory with
the test results.

## 2. Verify the required FoundationPose dtype patch

The Colab session first failed in FoundationPose crop projection with a
`float != double` matrix multiplication. Its successful path added this conversion
in upstream `Utils.py` before projecting points:

```python
pts = pts.to(dtype=K.dtype, device=K.device)
```

The same change is preserved as
`docker/patches/foundationpose-utils-dtype.patch`. The application adapter also
passes separate contiguous float32 model points/normals and a float32 diameter;
these are complementary safeguards. The clean upstream-only path has not been
proven on the target GPU, so apply the recorded patch for the first test.

Assuming FoundationPose is cloned at `/opt/FoundationPose`:

```bash
git -C /opt/FoundationPose fetch --all --tags
git -C /opt/FoundationPose checkout --detach a1b694b83e633c2cb6115b9063d940a687759392
git -C /opt/FoundationPose submodule update --init --recursive
git -C /opt/FoundationPose status --short

git -C /opt/FoundationPose apply --check \
  "$PWD/docker/patches/foundationpose-utils-dtype.patch"
git -C /opt/FoundationPose apply \
  "$PWD/docker/patches/foundationpose-utils-dtype.patch"
git -C /opt/FoundationPose diff --check
git -C /opt/FoundationPose diff -- Utils.py
python -m py_compile /opt/FoundationPose/Utils.py
```

The first `status --short` must be empty. After application, the only expected
tracked modification is `Utils.py`. If `git apply --check` reports that the patch is
already applied, verify it instead with:

```bash
git -C /opt/FoundationPose apply --reverse --check \
  "$PWD/docker/patches/foundationpose-utils-dtype.patch"
git -C /opt/FoundationPose diff -- Utils.py
```

Do not force-apply the patch or accept unrelated FoundationPose modifications.

## 3. Inspect the existing Conda environment

These commands are read-only and should succeed before running inference:

```bash
conda activate metaworld
python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__, 'CUDA', torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"
python -c "import nvdiffrast.torch as dr; print('nvdiffrast import OK'); dr.RasterizeCudaContext(); print('raster context OK')"
python -c "import warp; print('warp', warp.__version__)"
PYTHONPATH=/opt/FoundationPose:/opt/FoundationPose/mycpp/build \
  python -c "import mycpp; from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor; print('FoundationPose imports OK')"
```

Import success is not inference validation. If an import fails, compare the saved
environment with `Dockerfile` and `docker/requirements-gpu.txt`; do not repair it by
unbounded upgrades.

Install only this repository without resolving or changing GPU dependencies:

```bash
python -m pip install -e . --no-deps
python -m pip check
```

## 4. Keep assets and configuration local

Recommended private layout:

```text
/data/kuka/episode_000/
  camera.json
  rgb/<numeric-id>.png
  depth/<numeric-id>.png
  masks/<numeric-id>.png
/data/kuka/models/red_cube_yoloseg_best.pt
/data/kuka/models/red_cube.obj
/data/kuka/models/foundationpose_weights/
/data/kuka/baseline/
```

Create an ignored machine-specific configuration:

```bash
mkdir -p configs/local
cp configs/perception_v7_red_cube.yaml \
  configs/local/perception_v7_effective.yaml
```

Edit only these fields in the copy:

```yaml
models:
  yolo_weights: /data/kuka/models/red_cube_yoloseg_best.pt
  foundationpose_repository: /opt/FoundationPose
  foundationpose_weights: /data/kuka/models/foundationpose_weights
  mesh: /data/kuka/models/red_cube.obj
```

`configs/local/` is ignored. Never put workstation paths or private asset names in
the tracked templates.

## 5. Verify CAD and input conventions

FoundationPose does not rescale or recenter the mesh. Mesh vertices must be metres;
the mesh's native origin and axes become the returned object frame. For the reference
cube, all three extents should be approximately 0.040 m:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import trimesh

path = Path('/data/kuka/models/red_cube.obj')
mesh = trimesh.load(path, force='mesh')
print('mesh:', path.resolve())
print('vertices dtype:', mesh.vertices.dtype)
print('bounds [m]:\n', mesh.bounds)
print('extents [m]:', mesh.extents)
print('centroid [m]:', mesh.centroid)
assert np.allclose(mesh.extents, [0.04, 0.04, 0.04], atol=0.001, rtol=0)
PY

sha256sum \
  /data/kuka/models/red_cube.obj \
  /data/kuka/models/red_cube_yoloseg_best.pt \
  /data/kuka/models/foundationpose_weights/*/model_best.pth \
  > /data/kuka/experiment-assets.sha256
```

Document what +X, +Y, and +Z mean physically and where the CAD origin lies. The
offline source consumes only `K`, `depth_scale`, `frame_id`, and optional timestamps
from `camera.json`; it does not undistort images. RGB and depth must already be
aligned and rectified for that `K`.

## 6. CPU qualification

```bash
python -m pytest -q -m "not gpu_integration"
python -m compileall -q src scripts tests
python scripts/replay_perception.py --help
```

## 7. Stage the GPU test from smallest to largest

Use a new empty output directory for each run. A one-frame directory exercises
detection and registration; a short copied sequence adds `track_one()`; the original
episode exercises full tracking and recovery. Do not modify the source episode.

```bash
python scripts/replay_perception.py \
  --input /data/kuka/episode_one_frame \
  --config configs/local/perception_v7_effective.yaml \
  --output /data/kuka/results/rtx3060_one_frame \
  --save-reproducibility-manifest \
  --save-visualization

python scripts/replay_perception.py \
  --input /data/kuka/episode_short \
  --config configs/local/perception_v7_effective.yaml \
  --output /data/kuka/results/rtx3060_short \
  --save-reproducibility-manifest \
  --save-visualization

python scripts/replay_perception.py \
  --input /data/kuka/episode_000 \
  --config configs/local/perception_v7_effective.yaml \
  --output /data/kuka/results/rtx3060_full \
  --save-reproducibility-manifest \
  --save-visualization
```

The video is a status diagnostic, not a CAD pose overlay. Inspect `poses.csv`,
`metrics.json`, invalid frames, recovery events, and independent overlays before
treating a high valid ratio as pose accuracy.

## 8. Existing full regression acceptance test

The integration test expects the preserved 429-frame episode and baseline:

```bash
export KUKA_GPU_REGRESSION=1
export KUKA_GPU_DATASET=/data/kuka/episode_000
export KUKA_GPU_CONFIG="$PWD/configs/local/perception_v7_effective.yaml"
export KUKA_GPU_BASELINE=/data/kuka/baseline
export KUKA_GPU_BASELINE_KIND=modular
export KUKA_GPU_OUTPUT=/data/kuka/results/rtx3060_regression
export KUKA_GPU_TRANSLATION_TOLERANCE_M=0.005
export KUKA_GPU_ROTATION_TOLERANCE_DEG=2.0
export KUKA_GPU_FAIL_ON_DEVIATION=1

python -m pytest -q -s tests/test_gpu_integration.py -m gpu_integration
```

This test checks CUDA/GPU visibility, asset paths, the FoundationPose commit, mesh
extent, 429-frame correspondence, dtype diagnostics, validity, recovery events, and
pose tolerances. Its default expectation records the historical T4 as a hardware
deviation; the RTX 3060 therefore needs an explicitly reviewed result rather than an
assumption of numerical equivalence.

Preserve the complete output, `/tmp/metaworld-before-kuka.txt`, repository commit,
both Git status outputs, asset hashes, and any deviations. Only a successful reviewed
run qualifies the RTX 3060 deployment.
