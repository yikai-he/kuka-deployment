# KUKA Deployment Architecture

These diagrams document two different scopes:

1. [Current offline perception](current_perception_architecture.md) describes the
   code that exists in this repository.
2. [Planned ROS1 deployment](planned_ros1_deployment.md) shows how that perception
   core could be hosted on an Alibaba Cloud GPU and connected to a ROS1 Noetic
   laboratory system. The deployment interfaces are proposals, not implementations.

## Implemented perception modules

| Module | Responsibility |
|---|---|
| `scripts/replay_perception.py` | Loads YAML, constructs the pipeline, replays every frame, records timing, and serializes poses, metrics, diagnostics, effective configuration, and optional reproducibility data. |
| `perception.sources.RecordedSequenceSource` | Requires exactly paired numeric RGB/depth PNGs, loads `camera.json`, and yields ordered frames. `VideoDepthSource` also exists but requires an explicit RGB-to-depth correspondence. RGB-only video is rejected. |
| `perception.rgbd` | Converts OpenCV BGR to RGB and integer depth to float32 metres once; invalid or out-of-range depth becomes zero. |
| `perception.interfaces.RGBDFrame` | Typed input contract: aligned RGB, metric depth, intrinsics, frame number, optional capture timestamp, and camera frame ID. |
| `perception.segmenter.YoloSegmenter` | Lazily runs external Ultralytics YOLO-Seg weights and selects a red-cube mask using class, confidence, physical geometry, valid depth, and optional projected-pose hint. It does not download weights. |
| `perception.foundationpose.FoundationPoseEstimator` | Lazily loads an external FoundationPose checkout, mesh, weights, and CUDA renderer; exposes `register()` and `track()` and returns float32 camera-frame transforms. |
| `perception.tracker.PerceptionPipeline` | Owns the `UNINITIALIZED`, `TRACKING`, and `LOST` state machine, validates results, prevents stale-pose output, and coordinates recovery. |
| `validate_registered_pose()` / `detect_tracking_failure()` | Check rigid SE(3), configured depth, translation change, optional rotation change, mask-centroid agreement, and projected-center depth consistency. |
| `ObjectPoseEstimate` | Typed output contract containing `T_frame_object`, validity, `TrackingStatus`, source frame, detector confidence, and failure reason. The transform remains in the declared camera frame. |

## Recovery behavior

Initial registration and recovery use YOLO-Seg. While `TRACKING`, FoundationPose
produces a candidate pose. If validation rejects it, recovery-enabled operation
performs detection and re-registration on the same frame. A successful registration
returns to `TRACKING`; otherwise the pipeline remains `LOST` and emits an invalid
estimate. With recovery disabled, no stale pose is published. The last accepted
pose is retained only as a detection hint.

## Configuration, outputs, and evaluation

- `configs/perception.yaml` is the general safety-oriented profile.
- `configs/perception_v7_red_cube.yaml` preserves the legacy red-cube settings,
  including the 2 cm translation threshold and permissive unavailable-depth policy.
- Replay writes `poses.npy`, `poses.csv`, `metrics.json`, and `config.yaml`; optional
  products include pose text files, a status video, and a reproducibility manifest.
- `perception.regression.run_gpu_preflight()` verifies the CUDA/GPU environment,
  dataset contract, assets, FoundationPose revision, iteration counts, intrinsics,
  resolution, and CAD scale.
- `tests/test_gpu_integration.py` is opt-in and invokes the public replay CLI on the
  real GPU path. `scripts/compare_perception_runs.py` compares legacy and modular
  outputs, including validity, recovery events, latency, translation, and
  cube-symmetry-aware rotation.

## Deployment boundaries and status

### Current boundary

The implemented system is an offline Python package. It consumes recorded RGB-D
sequences and emits camera-frame pose estimates plus diagnostics. It contains no
ROS dependency, live-camera driver, network service, calibration loader, robot
interface, or policy runtime.

### Planned laboratory boundary

ROS1 Noetic would own live camera acquisition, RGB-D synchronization, the bridge to
remote inference, pose publication, TF/calibration, and future robot integration.
These components are dashed in the deployment diagram because none is implemented.

### Planned remote-GPU boundary

The existing segmentation, FoundationPose, validation, state, and recovery logic
could be hosted behind a new remote perception service on an Alibaba Cloud GPU.
Only the perception core is implemented; service lifecycle, serialization,
concurrency, session/state ownership, authentication, retries, and deployment are
still design work.

### Planned network boundary

The request would carry aligned RGB, metric depth, camera intrinsics, timestamps,
and frame identity. The response would carry a validated camera-frame 6D pose,
validity, tracking state/status, recovery information, and failure diagnostics.
Security, bandwidth, latency, ordering, timeout behavior, and the IPC/RPC protocol
remain unspecified. The diagrams intentionally do not select ZeroMQ, gRPC, or any
other transport.

## Important limitations

- Existing GPU experiments used recorded RGB-D data; they do not validate live or
  remote operation.
- Generic SE(3) utilities exist, but camera-to-robot calibration is not loaded or
  applied.
- ROS1 integration, cloud inference, robot motion planning/control, and
  continual-learning policy deployment are not implemented.
- A valid pose is a pipeline-accepted estimate, not independent 6D ground truth.
