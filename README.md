# Phase 1.1: Modular Offline RGB-D Perception

This ROS-free package processes aligned RGB and metric depth with YOLO-Seg and an
unmodified FoundationPose checkout. It contains no live camera, robot, gripper, or
policy code. The Phase 1 contracts remain unchanged: `process_frame(frame)` accepts
an `RGBDFrame` and returns an `ObjectPoseEstimate`.

## Provenance

The recovery behavior was ported from these preserved workspace references:

- `../FoundationPose_KUKA_v7_YOLOSeg_drive_batch(4).ipynb`
- `../run_kuka_sequence(2).py`
- `../FoundationPose_KUKA_clean(2).ipynb`

The v7 notebook identifies FoundationPose revision
`a1b694b83e633c2cb6115b9063d940a687759392`. No reference file is imported or
modified by this package.

## Contracts and pose convention

RGB is `(H,W,3)` uint8 in RGB order. Depth is `(H,W)` float32 in metres and zero
means invalid. Capture timestamps may be unavailable; derived video presentation
time is stored separately. RGB/depth filename sets must match exactly.

`T_frame_object` maps object-frame points into the declared camera `frame_id`, with
translation in metres. Invalid estimates contain `None`, and replay serializes them
as NaN matrices. Detector confidence is YOLO confidence, never tracking confidence.

## v7 candidate selection

Each returned YOLO mask is resized as a probability mask with linear interpolation,
thresholded at 0.5, and reduced to its largest 8-connected component. Candidates
then pass class, area, physical-size, physical-aspect, and valid-depth-ratio gates.
The physical equivalent side is:

```text
equiv_side_m = sqrt(area_px) * median_depth_m / sqrt(fx * fy)
```

The exact v7 score is minimized:

```text
-log(confidence)
+ 0.70 * abs(log(equiv_side_m / target_size_m))
+ 0.15 * abs(log(aspect_ratio))
+ recovery_hint_weight * hint_distance_px / 100
```

The final term is present only when a previous accepted pose projects to a usable
hint. Its gate expands by `lost_hint_expand_px * lost_streak`, capped at the image
diagonal. When no candidate passes the hint and the LOST streak reaches three, v7
falls back to geometry-valid candidates without the hint; its original fallback
score omits the aspect term as well as the hint term.

## Validation and state machine

The state machine is `UNINITIALIZED -> TRACKING -> LOST`. Initial registration uses
YOLO and can optionally use the recorded initial mask. Only a successful registration
enables tracking. Suspicious tracking triggers same-frame detection and registration;
LOST frames retry registration directly and never call `track_one()`. The last
accepted pose is retained only as a detection hint and is never emitted as valid.

Registration validates full SE(3), translation depth, and pose-center agreement with
the mask RGB-D centroid. Tracking validates full SE(3), translation depth, translation
jump outside cooldown, and projected-center depth residual. Rotation-matrix structure
is always checked, but generic rotation-jump rejection is disabled for the symmetric
red cube.

Depth consistency is always recorded as `PASSED`, `FAILED`, or `UNAVAILABLE`:

- `fail` treats unavailable depth as suspicious and is the general deployment default.
- `allow` records it as unavailable but accepts it, reproducing v7 occlusion behavior.

## Configuration profiles

- `configs/perception.yaml` is the general, safety-oriented profile. It uses the
  `fail` depth policy and does not assume a recorded fallback mask.
- `configs/perception_v7_red_cube.yaml` contains the original v7 red-cube defaults,
  including 0.30 YOLO confidence, 2 cm jump, two-frame cooldown, physical size
  range 1.5–8 cm, 0.35 depth ratio, and permissive `allow` depth policy.

Important legacy mapping:

| Legacy CLI | Modular configuration |
|---|---|
| `--yolo_conf/iou/imgsz/max_det/class_id` | `segmentation.*` |
| `--target/min/max_object_size_m` | `red_cube.*_size_m` |
| `--min_component_area/max_aspect_ratio/min_candidate_depth_ratio` | `red_cube.*` |
| `--jump_threshold_m` | `validation.pose.max_translation_step_m` |
| `--center_depth_residual_m/window_px` | `validation.pose.center_depth_*` |
| `--register_mask_center_max_m` | `validation.pose.register_mask_center_max_m` |
| `--recovery_cooldown_frames` | `recovery.cooldown_frames` |
| `--recovery_hint_weight/max_px/lost_hint_expand_px` | `recovery.*` |
| `--disable_recovery` | `recovery.enabled: false` |
| `--allow_initial_mask_fallback` | `recovery.allow_initial_mask_fallback` |

The legacy HSV arguments were not active in `detect_red_cube_mask()` and therefore
are documented rather than introduced as misleading active settings.

## Offline replay and outputs

```bash
python3 scripts/replay_perception.py \
  --input /path/to/recorded_sequence \
  --config configs/perception.yaml \
  --output /path/to/results
```

Replay writes `poses.npy`, `poses.csv`, `metrics.json`, and the effective
`config.yaml`. Every input frame is retained. CSV diagnostics include state, LOST
streak, cooldown, recovery attempt/success, depth status/residual, registration
centroid error, and selected-candidate information. Per-frame pose text files are
profile-controlled. `--save-visualization` enables the optional status video; debug
images and video are never required.

## CPU tests

```bash
python3 -m pytest -q
```

Tests use mock YOLO/FoundationPose/video objects. They need no CUDA, model weights,
ROS, RealSense, or KUKA hardware.

## Side-by-side GPU regression (pending)

The complete pinned Colab procedure is in
[`docs/COLAB_GPU_REGRESSION.md`](docs/COLAB_GPU_REGRESSION.md). After both runs,
generate CPU-side JSON and Markdown reports with:

```bash
python3 scripts/compare_perception_runs.py \
  --legacy /path/to/legacy_result \
  --modular /path/to/modular_result \
  --output /path/to/comparison
```

Required inputs are one unchanged recorded RGB-D episode with `camera.json` and
initial mask, `red_cube.obj` in metres, `red_cube_yoloseg_best.pt`, FoundationPose
weights, the pinned FoundationPose checkout, and a compatible NVIDIA environment.

Run the legacy reference:

```bash
python3 '../run_kuka_sequence(2).py' \
  --data_dir /data/episode \
  --out_dir /results/legacy_v7 \
  --mesh_path /models/red_cube.obj \
  --fp_root /opt/FoundationPose \
  --yolo_model /models/red_cube_yoloseg_best.pt \
  --allow_initial_mask_fallback --no_vis
```

After replacing paths in the v7 YAML, run the modular pipeline:

```bash
python3 scripts/replay_perception.py \
  --input /data/episode \
  --config configs/perception_v7_red_cube.yaml \
  --output /results/modular_v7
```

Compare ordered frame identities, first registration, valid ratio, LOST frames,
recovery attempts/successes, status sequence, latency, and translations. Compare
rotations modulo the cube's 24 proper rotational symmetries rather than raw matrix
distance. Exact event parity is expected where deliberate fixes do not apply;
floating-point trajectories require tolerances established on the actual GPU.

The same document includes an opt-in `gpu_integration` pytest invocation for the
429-frame task1/episode_000 regression. It is skipped unless
`KUKA_GPU_REGRESSION=1` and explicit dataset, config, baseline, and output paths are
provided. The test uses this replay CLI rather than implementing a second pipeline.

## Deliberate deviations and remaining limits

- Invalid SE(3), premature tracking, LOST-state tracking, and stale-pose output are
  rejected rather than reproducing legacy bugs.
- Missing depth is explicit even under legacy-compatible `allow` policy.
- The modular source rejects unmatched frames; v7 silently intersected ID sets.
- GPU parity is unverified because the dataset, cube mesh, perception weights,
  FoundationPose checkout, and CUDA environment are not available here.
- RealSense `.bag`/live capture, rectification, ROS, policy inference, calibration,
  and robot control remain outside Phase 1.1.
