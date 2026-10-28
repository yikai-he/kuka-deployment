# KUKA Deployment Project Status

**Review date:** 2026-10-23
**Current repository commit:** `bed0169f938cf79a9c1dbfae0b3ab0c526b61521` (`main`)
**GPU-regression infrastructure commit:** `c641f205eae68dd870d449ad4d7a3341954509c3`
**Commits used in the Colab notebook:** exploratory/legacy work began at `1ceaa5ce888fc18a2b1530b6bfeeb8f086441248`; the first modular attempt used `c641f205eae68dd870d449ad4d7a3341954509c3`; the reported recovery-disabled and recovery-enabled runs used `bed0169f938cf79a9c1dbfae0b3ab0c526b61521`.
**Working-tree qualification:** the last observed repository HEAD during the
laboratory-preparation review was `133ae095b7cc23ed82b2ab00dc42ad783813a4a5`, with
additional uncommitted implementation and documentation changes. The successful
historical modular experiments used `bed0169f938cf79a9c1dbfae0b3ab0c526b61521`.
This document describes the reviewed working tree and does not claim that it is
byte-identical to either saved experiment. Always record `git rev-parse HEAD` and
`git status --short` with a new run. The notebook also modified the external
FoundationPose checkout after checking out its pinned commit; see Sections 5, 6,
and 8.

Evidence labels used below:

- **Repository-verified:** present in code, tests, configuration, or committed documentation.
- **Notebook-verified:** visible in saved notebook code/output; this is evidence of a historical Colab session, not an independently reproduced run or a substitute for the external result bundle.
- **Context only:** supplied with the review request but absent from repository artifacts, so it could not be independently checked.

## 1. Project Overview

The repository implements Phase 1 of a KUKA-oriented perception stack: offline estimation of a red cube's 6D pose from aligned RGB-D recordings. YOLO instance segmentation supplies an object mask, FoundationPose registers the CAD model and tracks it between frames, and validation plus detector-guided re-registration attempts to recover from drift or tracking failure.

The package is intentionally ROS-independent. It does **not** contain live-camera acquisition, ROS nodes/messages, hand-eye calibration, camera-to-robot transforms, motion planning, gripper or robot control, or continual-learning policy integration. It therefore demonstrates an offline perception subsystem, not full robot deployment.

## 2. Repository Structure

| Path | Role |
|---|---|
| `README.md` | Phase 1.1 design, contracts, legacy-v7 parity notes, replay usage, and stated limits |
| `src/kuka_deployment/perception/` | RGB-D/pose contracts, sources, segmentation, FoundationPose adapter, validation/state machine, transforms, and regression metadata |
| `scripts/replay_perception.py` | Main offline CLI; runs inference and writes poses, diagnostics, metrics, effective configuration, and optional visualization/manifest |
| `scripts/compare_perception_runs.py` | Loads legacy/modular result schemas and compares frame sets, validity, events, latency, translation, and cube-symmetry-aware rotation |
| `configs/perception.yaml` | General safety-oriented profile (`depth_unavailable_policy: fail`) |
| `configs/perception_v7_red_cube.yaml` | Legacy-v7-compatible red-cube profile, including 2 cm jump threshold, cooldown, detection hinting, and permissive missing-depth behavior |
| `tests/` | CPU unit/regression tests plus one opt-in real GPU integration test |
| `docs/COLAB_GPU_REGRESSION.md` | Pinned Colab setup and formal legacy-versus-modular GPU regression procedure |
| `models/README.md` | External asset layout and provenance/hash requirements; no model, mesh, or dataset is stored in Git |
| `kuka_deployment_refactored(1).ipynb` | Untracked exploratory Colab notebook with environment setup, RGB-D/CAD inspection, dtype debugging, FoundationPose inference, and saved outputs |
| `kuka_deployment_refactored(2).ipynb` | Untracked continuation containing the modular GPU regressions, recovery evaluation, result-backup paths, frame-328 diagnostics, pose projections, and CAD overlays |
| `pyproject.toml` | Python package metadata; minimal runtime dependencies and optional YOLO/test groups |

There are no checked-in experiment result directories (`poses.csv`, `metrics.json`, comparison reports, manifests, datasets, weights, meshes, or overlay images). The second notebook records Colab-local and Google Drive artifact paths, but those artifacts are not present in this checkout. Experiment verification is therefore verification of saved notebook outputs and code, not an independent inspection or rerun of the referenced result bundles.

## 3. System Architecture and Data Flow

1. **Input:** `RecordedSequenceSource` requires exactly matched numeric filenames under `rgb/` and `depth/`, reads `camera.json`, converts BGR to RGB, scales integer depth to float32 metres, rejects invalid ranges as zero, and preserves capture timestamps when supplied (`sources.py`, `rgbd.py`). `VideoDepthSource` additionally supports an explicit video-to-depth mapping; RGB-only video is rejected.
2. **Segmentation:** `YoloSegmenter` runs Ultralytics YOLO-Seg. Each mask is probability-resized, thresholded at 0.5, and reduced to its largest 8-connected component (`segmenter.py`).
3. **Candidate selection:** candidates are filtered by class, component area, valid-depth fraction, physical equivalent side length, aspect ratio, and (when available) distance from the projected last accepted pose. The v7 score combines negative log confidence, size error, aspect error, and hint distance. After a configurable LOST streak, selection may fall back to geometry-valid candidates without the hint.
4. **Registration/tracking:** `FoundationPoseEstimator` lazily loads an external pinned FoundationPose checkout, CAD mesh, weights, and CUDA renderer once. Registration receives RGB, metric depth, intrinsics, and a mask; later frames call `track_one()` (`foundationpose.py`). Returned poses are float32 `T_camera_object` matrices.
5. **Validation:** registration checks finite rigid SE(3), configured depth range, and optionally the distance between pose translation and the mask's RGB-D centroid. Tracking additionally checks frame-to-frame translation, optional rotation change, and the residual between projected pose-center depth and observed depth (`tracker.py`). Missing depth is explicitly `UNAVAILABLE`; policy can reject or allow it.
6. **State and recovery:** the state machine is `UNINITIALIZED -> TRACKING -> LOST`. Initial registration uses YOLO and may use an explicitly enabled recorded-mask fallback. A suspicious tracking result triggers detection and registration on the same frame. In `LOST`, tracking is not called: recovery-enabled operation retries registration, while recovery-disabled operation emits invalid results. The last accepted pose is retained only as a detection hint and is never emitted as a current valid pose.
7. **Outputs/evaluation:** replay preserves every input frame, serializes invalid poses as NaN matrices, and writes `poses.npy`, `poses.csv`, `metrics.json`, and `config.yaml`. Optional outputs include per-frame pose files, a status video, and a reproducibility manifest. The comparison CLI normalizes legacy/modular schemas and evaluates rotations modulo the 24 proper symmetries of an ideal axis-aligned cube.

The pose remains in the declared camera frame. No implemented path transforms it into a robot base/tool frame.

## 4. Implemented Components

| Component | Status | Evidence and qualification |
|---|---|---|
| Offline RGB-D replay | **Implemented and tested** | `replay_perception.py`, `RecordedSequenceSource`, source/RGB-D tests |
| YOLO-Seg adapter and v7 red-cube selection | **Implemented and tested** | `segmenter.py`; candidate scoring, class/geometry/depth/hint behavior exercised in `test_v7_recovery.py` |
| FoundationPose registration/tracking adapter | **Implemented; notebook-validated on GPU with qualification** | The opt-in test passed for both final modular runs and recorded float32 boundary diagnostics, but the external FoundationPose checkout still contained the notebook's local dtype patch |
| Pose/data contracts and SE(3) validation | **Implemented and tested** | `interfaces.py`, `transforms.py`, `tracker.py`; invalid/stale poses rejected by tests |
| Translation/depth validation | **Implemented and tested** | configurable jump, pose depth, centroid, and projected-center checks; rotation jump exists but is disabled for the symmetric cube profiles |
| Tracking state machine and recovery | **Implemented and tested with mocks** | same-frame recovery, LOST retry, cooldown, hint expansion/fallback, and recovery-disabled behavior in `test_tracker.py` and `test_v7_recovery.py` |
| Replay metrics and legacy comparison | **Implemented and tested** | `replay_perception.py`, `compare_perception_runs.py`, `test_compare_perception_runs.py` |
| GPU preflight/reproducibility manifest | **Implemented and notebook-exercised; artifacts external** | `regression.py`, `test_gpu_integration.py`, and `COLAB_GPU_REGRESSION.md`; notebook output shows T4 preflight and saved manifests, but the files themselves are not checked in |
| Detection-guided recovery | **Implemented, mock-tested, and demonstrated once in saved notebook output** | One frame-328 re-registration is recorded; this establishes a recovery event and accepted pose, not pose accuracy or general robustness |
| RGB-D/CAD pose visualization | **Partially implemented; notebook-verified qualitatively** | replay status video does not render the CAD; notebook outputs include projected pose centers and a legacy-versus-recovery CAD-box overlay saved to an external Drive result directory |
| ROS interface | **Not implemented** | no ROS packages, nodes, message definitions, launch files, or ROS dependencies |
| Camera-to-robot transformation | **Not implemented** | generic SE(3) utilities exist, but no calibration or camera/base transform is loaded or applied |
| Real robot manipulation | **Not implemented** | no KUKA/gripper/control/motion code |
| Continual-learning policy integration | **Not implemented** | no policy API or dependency in this repository |

## 5. Development and Engineering Challenges

### FoundationPose and Trimesh dtype compatibility

Notebook cells 77–84 record a real `float != double` failure inside FoundationPose crop projection. Assigning or reconstructing a Trimesh mesh with float32 vertices did not solve it: saved output still reported `mesh.vertices` as float64. The notebook ultimately patched upstream `FoundationPose/Utils.py` to cast points to the intrinsics dtype, after which registration succeeded.

The repository adapter uses a less invasive boundary fix in `foundationpose.py`: it leaves Trimesh's native float64 storage intact, passes separate contiguous float32 `model_pts` and `model_normals` to the FoundationPose constructor, coerces the estimator diameter to float32, and records actual input/output dtypes. `test_foundationpose.py` checks this separation. The final recovery-disabled manifest output records float64 mesh vertices/normals, float32 model points/normals and diameter, 1 registration call, and 328 tracking calls; the integration test also checks float32 depth inputs and float32 outputs.

This is **Notebook-verified dtype-boundary evidence**, but not a clean isolation test of the adapter fix: earlier notebook cells patched `/content/FoundationPose/Utils.py` to cast `pts` to `K.dtype`, and no later cell resets or proves a clean FoundationPose working tree before the modular runs. `git rev-parse` continued to report the pinned FoundationPose commit because local modifications do not change the commit hash.

### GPU environment and dependency setup

The notebook records a Tesla T4 (compute capability 7.5), a successful CUDA computation, PyTorch `2.4.1+cu121`, NumPy `1.26.4`, Pandas `2.2.3`, FoundationPose imports, nvdiffrast rasterization, and successful weight discovery. It also records failed import/build attempts, a missing Ninja executable on `PATH`, and repeated registration failures before the environment and dtype fixes. Later cells add the Conda environment to `PATH` and complete registration/tracking.

The documented reproducible route pins FoundationPose commit `a1b694b83e633c2cb6115b9063d940a687759392`, PyTorch/PyTorch3D/nvdiffrast/Ultralytics versions, GPU expectations, assets, hashes, dataset shape, mesh scale, and iteration counts. Some Colab/system packages remain runtime-dependent, and the assets are private/external.

### Failure detection and recovery

A finite 4x4 output alone is not evidence that tracking is correct. The modular pipeline rejects non-rigid transforms, out-of-range depth, excessive translation changes, centroid disagreement, and projected-depth disagreement. On failure, it never publishes the stale pose. Recovery uses current YOLO evidence and the prior pose only as a candidate-selection hint. The recovery-disabled path enters `LOST` and remains invalid without calling tracking again; recovery-enabled operation re-registers or stays invalid.

### Symmetry and absence of ground truth

The red cube's rotational symmetry makes raw rotation-matrix differences misleading; the comparison tool minimizes error over 24 cube symmetries. More fundamentally, the episode provides no independently measured 6D ground truth. Valid-pose ratio measures pipeline acceptance/availability, **not pose accuracy**. RGB projection/CAD overlay is useful qualitative support but is not quantitative ground truth, especially while image rectification/distortion assumptions remain unverified.

## 6. Testing and Experimental Validation

### Local CPU tests

On 2026-10-28, `python3 -m pytest -q -m 'not gpu_integration'` completed with
**59 passed, 1 deselected**. Tests use mocks and synthetic arrays; they do not require
CUDA, weights, ROS, or robot hardware. They cover contracts, RGB/depth conversion,
paired and live sources, transforms, candidate selection, state transitions,
recovery and failure semantics, FoundationPose dtype handoff, shared runtime logic,
comparison logic, and regression helpers.

`tests/test_gpu_integration.py` is intentionally opt-in. It invokes the real replay CLI, checks 429 outputs and dtype observations, writes preflight/manifest/comparison files, and can gate deviations from a saved baseline. It was not run during this review, and no generated outputs are checked into the repository.

### Colab notebook evidence

The first untracked notebook contains 95 cells. The second contains 119 cells (99 code cells), saved outputs, and nine saved error outputs; all top-level `execution_count` fields are null, although cell metadata retains original indices. Consequently, cell order and stored outputs show a historical interactive session, but do not prove a clean top-to-bottom execution. Later saved outputs nevertheless verify:

- T4 GPU detection and basic CUDA/nvdiffrast computation;
- 429 matched 640×480 RGB/depth frames, a 1 mm depth scale, one initial mask, and a 4 cm cube mesh;
- successful FoundationPose import, weight loading, initial registration, 20-frame tracking, and full 429-frame legacy-style tracking;
- initial CAD projection/pose overlay and generation of a 429-frame tracking overlay video;
- a full-run legacy-style tracker result of mean/max translation jump 2.079/14.825 mm, mean/max rotation jump 2.817/19.647°, mean frame time 56.13 ms, and no jump above its 20 mm threshold.
- execution of the real modular replay through `tests/test_gpu_integration.py` in recovery-disabled and recovery-enabled configurations, including saved metrics, per-frame diagnostics, comparison reports, and external backup paths.

That final run directly calls FoundationPose with the recorded initial mask and then calls `track_one()` for every remaining frame. It marks all finite matrices as usable and has no modular validation, LOST state, YOLO recovery, or accuracy ground truth. It is therefore a useful **legacy FoundationPose comparison**, but not evidence that all 429 poses followed the cube. The notebook's projected CAD overlay is qualitative evidence only; the code explicitly assumes rectified imagery/zero distortion for some visualizations despite `camera.json` containing non-zero distortion coefficients.

### Phase 1 GPU Regression and Recovery Evaluation

The second notebook's saved Cell 43 and Cell 46 outputs directly report the following values:

| Metric | Recovery Disabled | Recovery Enabled | Verification |
|---|---:|---:|---|
| Total frames | 429 | 429 | **Notebook-verified** |
| Valid frames | 328 | 429 | **Notebook-verified** |
| Valid pose ratio | 76.4569% | 100% | **Notebook-verified** |
| Invalid frames | 101 | 0 | **Notebook-verified** |
| Recovery attempts | 0 | 1 | **Notebook-verified** |
| Recovery successes | 0 | 1 | **Notebook-verified** |
| Mean inference latency | 53.8053 ms | 75.8612 ms | **Notebook-verified** |

The saved CSV diagnostics identify frame 328 as the first tracking failure in the disabled run, caused by a translation jump exceeding the configured threshold; frames 328–428 are invalid. In the enabled run, frame 328 is `reregistered`, with `recovery_attempted=True`, `recovery_success=True`, detector confidence 0.881128, and inference latency 3874.788 ms. The notebook reports one attempt and one success. This is **Notebook-verified** evidence for the event and its recorded latency. It remains one episode, and the test ran with `KUKA_GPU_FAIL_ON_DEVIATION=0`, so “passed” means the integration assertions completed while baseline deviations were recorded; it is not a numerical parity verdict.

The reported final runs used KUKA commit `bed0169f938cf79a9c1dbfae0b3ab0c526b61521`. The enabled cell asserts this hash directly. The disabled cell follows the successful checkout/reinstall of that hash but does not repeat the hash assertion inside the run cell, so its commit attribution depends on notebook ordering. An earlier modular run at `c641f205eae68dd870d449ad4d7a3341954509c3` produced 328 valid frames but failed the then-current integration assertion (`mesh_vertices_dtype` was float64 rather than the expected float32); its 87.94 ms mean is not the reported final disabled result. The direct legacy FoundationPose run occurred before the modular checkout sequence and used a standalone script rather than repository replay code; the repository was then at `1ceaa5ce888fc18a2b1530b6bfeeb8f086441248`.

Pose-center projections and the saved CAD-box comparison provide **qualitative visual evidence** for the interpretation that, from frame 328 onward, the recovered pose returns to the red cube on the table while the legacy tracker continues near the gripper. The notebook reports projected centers at frame 328 of approximately `[257.0, 192.1]` for legacy and `[251.9, 298.6]` for recovery, a 255.2 mm translation difference, a YOLO detection around the table cube, and an overlay generated from the actual 4 cm CAD bounds with camera distortion applied. This is not independent 6D ground truth and does not prove orientation or full pose accuracy.

Accordingly, 100% valid poses must be interpreted only as 429/429 accepted outputs in the reported run. It does not mean 100% 6D pose accuracy.

## 7. Current Capabilities

| Area | Current assessment |
|---|---|
| 1. Offline perception replay | **Implemented and tested** on CPU orchestration; real episode operation is supported by the CLI and partly evidenced by notebook experiments |
| 2. GPU-compatible perception inference | **Implemented and notebook-validated on a T4, with reproducibility qualifications**; both final modular runs completed, but their external artifacts are absent and FoundationPose was locally patched |
| 3. Perception robustness and recovery | **Implemented, mock-tested, and notebook-demonstrated once; experimentally under-validated**; one recovery event restored accepted output, without independent pose ground truth |
| 4. ROS integration | **Not implemented** |
| 5. Camera-to-robot coordinate transformations | **Not implemented**; camera-frame pose only |
| 6. Real robot manipulation | **Not implemented** |
| 7. Continual-learning policy integration | **Not implemented** |

## 8. Known Limitations

- No independent 6D ground truth, so accepted poses, smooth trajectories, and overlays cannot establish accuracy.
- The recovery comparison and frame-328 visual conclusion are present in saved notebook outputs, but the referenced CSV/NPY/metrics/video/manifest/overlay artifacts live only at recorded Colab/Drive paths and are not independently inspectable from this checkout.
- Only one 429-frame episode and one formal recovery event were evaluated; robustness across tasks, episodes, lighting, occlusion, distractors, camera motion, repeated losses, or failed recovery is unknown.
- Detection quality, recovery false positives, erroneous but validation-passing re-registrations, repeated recovery, and unrecoverable failures are not quantitatively characterized.
- Mean latency mixes cheap tracking and expensive registration. The recovery run averaged 75.86 ms versus 53.81 ms disabled, and the single recovery event took 3.875 s; tail latency, per-stage latency, throughput, warm-up, and additional recovery samples remain uncharacterized.
- The notebooks are exploratory and mutable: the second has null execution counts and nine saved errors; it moves through KUKA commits `1ceaa5c`, `c641f20`, and `bed0169`; patches the external FoundationPose checkout without recording a clean-tree status afterward; and uses more than one dataset/result path convention.
- The current checkout contains substantial uncommitted source changes relative to `bed0169` and both notebooks are untracked. Those changes were not part of the recorded Colab runs unless they are merely later formatting of identical logic; no clean diff/hash artifact proves equivalence.
- Model weights, CAD, dataset, legacy runner, baselines, and result artifacts are external. The repository documents hashes but provides none for the reviewed experiment.
- The generic and v7 configurations differ intentionally (notably depth-unavailable policy and translation threshold), so comparisons require saving the effective configuration.
- Camera distortion/rectification assumptions are unresolved for quantitative overlays.
- No live source, synchronization validation, ROS transport, calibration, safety layer, robot command path, or closed-loop manipulation test exists.

## 9. Remaining Work and Prioritized Roadmap

### Essential next steps

1. **Multi-episode robustness evaluation.** Run the pinned modular pipeline over multiple predeclared episodes/tasks and conditions. Save immutable per-run outputs, asset hashes, commit/diff identity, environment manifest, and aggregate success/failure distributions.
2. **Verify recovery pose quality and failure handling.** Add independent pose/reference measurements where possible; otherwise use annotated object location, segmentation/centroid residuals, blinded overlay review, and explicit false-recovery criteria. Test missing detections, wrong-instance detections, repeated loss, occlusion, and registration exceptions.
3. **Evaluate latency properly.** Separate YOLO, tracking, validation, registration, serialization, and visualization; report median/p95/p99, warm versus cold behavior, recovery latency, and hardware/runtime identity. Reproduce the notebook-verified 53.81/75.86 ms means and 3.87 s recovery event from preserved raw artifacts across more episodes.
4. **Create a ROS-compatible perception boundary.** Define a timestamped camera-frame pose plus validity, state, failure reason, recovery event, and covariance/quality fields. Then implement and test a ROS node or bridge without coupling the core offline package to robot control.
5. **Implement camera-to-robot calibration and transforms.** Establish image rectification and RGB-depth alignment, measure and version `T_base_camera` (or the appropriate frame chain), propagate timestamps, and test transforms against physical fiducials/measurements before commanding motion.

### Optional staged deployment

6. **Single-task real-robot execution.** Only after perception/calibration gates pass, add a bounded, low-speed task with workspace checks, collision/motion planning, gripper integration, stop behavior, and operator supervision. This is optional for proving the perception phase.
7. **Continual-learning integration.** Define a stable observation/action interface, confidence-aware fallback behavior, dataset/version provenance, and offline/simulation safety evaluation before allowing a learned policy to consume real perception outputs. This is future work, not a current capability.

## 10. Reproducibility and Key References

Primary repository references:

- `README.md` — contracts, v7 selection/state logic, CLI use, and limits.
- `docs/COLAB_GPU_REGRESSION.md` — pinned environment and formal regression recipe.
- `models/README.md` — external asset provenance and integrity requirements.
- `configs/perception.yaml` and `configs/perception_v7_red_cube.yaml` — effective behavior profiles.
- `scripts/replay_perception.py` and `scripts/compare_perception_runs.py` — execution and evaluation entry points.
- `src/kuka_deployment/perception/foundationpose.py`, `segmenter.py`, `tracker.py`, and `regression.py` — core inference, recovery, and provenance implementation.
- `tests/test_v7_recovery.py`, `test_foundationpose.py`, and `test_gpu_integration.py` — recovery semantics, dtype boundary, and opt-in real regression contract.
- `kuka_deployment_refactored(1).ipynb` — initial historical Colab setup, dtype debugging, inference, and visualization evidence.
- `kuka_deployment_refactored(2).ipynb` — continuation containing the final recovery-disabled and recovery-enabled modular runs, metrics and frame diagnostics, legacy comparison, projection/CAD-overlay outputs, and external artifact backup paths.

For a reproducible GPU result, use a reviewed KUKA commit, the pinned FoundationPose
checkout at `a1b694b83e633c2cb6115b9063d940a687759392` with the explicitly recorded dtype
patch, a fresh output directory, an unchanged 429-frame episode, recorded effective
YAML, and SHA-256 hashes for the mesh, YOLO weights, and FoundationPose weights.
Preserve `preflight.json`, `reproducibility_manifest.json`, `poses.npy`, `poses.csv`,
`metrics.json`, comparison JSON/Markdown, raw timing data, and visualizations
together. Record `git status` and a diff/hash for both repositories. The reported
final experiments used KUKA `bed0169`; new RTX 3060 results must record the actual
reviewed commit and must not be described as validated until the opt-in GPU test
passes.
