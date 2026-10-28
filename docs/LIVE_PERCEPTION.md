# Live RGB-D Perception

## Architecture and implementation status

The implemented core has one input and output contract for both execution modes:

```text
RecordedSequenceSource ─┐
                       ├─> RGBDFrame ─> PerceptionPipeline.process_frame()
LiveFrameSource ────────┘                  │
                                           └─> ObjectPoseEstimate
```

`OfflineFrameSource` (the backward-compatible `RecordedSequenceSource`), YOLO-Seg, FoundationPose registration/tracking, pose
validation, detection-guided recovery, and the offline output format remain the
existing implementations. `LiveFrameSource`, the camera plugin contract, the live
CLI, JSONL output, and live timing/statistics are implemented in this repository.

No hardware-specific camera backend is bundled because the laboratory camera has
not been selected. Camera SDK installation, physical camera validation, CUDA
inference validation, ROS, calibration, and robot control remain future work.

## RGB-D and pose contracts

Every source yields `kuka_deployment.perception.interfaces.RGBDFrame`:

- `rgb`: aligned `(H, W, 3)` `uint8`, RGB channel order;
- `depth_m`: aligned `(H, W)` `float32`, metres, with `0.0` for invalid depth;
- `K`: finite floating-point `3 x 3` camera intrinsic matrix;
- `timestamp_ns`: camera capture timestamp when the backend can provide one;
- `frame_number`: non-negative source sequence number;
- `frame_id`: the optical camera coordinate frame;
- `acquisition_monotonic_ns`: host monotonic arrival time, populated by the live
  source when the backend does not set it.

Camera timestamps are preserved but are not assumed to share the host clock.
End-to-end latency is therefore measured from host monotonic arrival, while CUDA
is synchronized immediately before and after inference for live inference timing.

`ObjectPoseEstimate.T_frame_object` maps object-frame coordinates into `frame_id`.
For a camera backend this is `T_camera_object`; translation is in metres. Invalid
results contain no matrix, so a failed recovery cannot publish a stale valid pose.

## Frame sources and camera backend

Offline replay reads the existing `rgb/<id>.png`, `depth/<id>.png`, and
`camera.json` layout. Ordering, intrinsics, timestamps, and strict RGB/depth pairing
are unchanged.

Live acquisition is split into two small interfaces:

- `CameraBackend.start()` initializes the device.
- `CameraBackend.read(timeout_s)` returns the latest aligned `RGBDFrame`, `None`
  for a temporary interruption, or raises `StopIteration` for a clean end.
- `CameraBackend.stop()` is idempotent and unblocks a pending read.
- `LiveFrameSource` acquires in one background thread and holds exactly one pending
  frame. A newer frame replaces an unconsumed one and increments `dropped_frames`.

An optional backend package should expose a factory accepting a configuration
mapping, for example:

```python
def create_camera(options):
    return MyCameraBackend(options)
```

Hardware SDK imports should occur inside that plugin, not in the core package.
The factory is selected as `package.module:create_camera`. This keeps `--help`,
offline replay, and CPU tests independent of camera drivers.

Optional camera settings can be added without changing the general perception
profile:

```yaml
live:
  source:
    read_timeout_s: 0.25
    retry_delay_s: 0.01
    startup_timeout_s: 5.0
    shutdown_timeout_s: 2.0
  camera:
    serial: optional-device-specific-value
```

`--backend-config camera.yaml` may override the `live.camera` mapping. Its keys are
backend-specific.

## Running

Offline behavior and arguments are unchanged:

```bash
python3 scripts/replay_perception.py \
  --input /data/episode \
  --config configs/perception.yaml \
  --output /results/offline
```

Run a separately installed camera plugin with:

```bash
python3 scripts/live_perception.py \
  --config configs/perception.yaml \
  --backend my_camera_plugin:create_camera \
  --backend-config /config/camera.yaml \
  --output /results/live \
  --max-frames 500
```

`--save-visualization` writes `live_tracking.mp4`; it does not open a GUI. Ctrl+C
stops acquisition and closes the backend. The live run writes:

- `live_results.jsonl`: one pose/status record per processed frame;
- `live_metrics.json`: counts, recovery events, dropped frames, and mean latencies;
- `config.yaml`: the effective perception configuration.

Each JSONL record includes capture timestamp, frame number and ID, object ID,
validity, tracker status, recovery state, processing latency, end-to-end latency
when measurable, and the `T_camera_object` matrix or `null`.

## Tracking and recovery

Both modes use the same stateful `PerceptionPipeline` and model instances. Initial
detection registers FoundationPose once. Valid frames track without YOLO inference.
When existing pose checks reject tracking, the pipeline invokes YOLO re-detection
and FoundationPose re-registration. Failed detection or registration produces an
invalid pose and a failure status. The live adapter does not introduce new
thresholds or change this state machine.

## Testing without a camera

```bash
python3 scripts/live_perception.py --help
python3 -m pytest -q tests/test_live_sources.py tests/test_runtime.py
python3 -m pytest -q
```

The tests use synthetic RGB-D frames and fake camera/detector/estimator objects.
They cover validation, temporary interruptions, bounded latest-frame behavior,
shutdown, timestamps, failure propagation, timing, state transitions, recovery,
and the pre-existing offline sources. They do not validate a camera SDK, CUDA
kernels, model weights, or physical pose accuracy.

## Current limitations

- No camera-specific backend or driver is included or installed in Docker.
- RGB/depth alignment and intrinsic calibration are a backend responsibility.
- Camera capture timestamps may use a device-specific clock domain.
- There is no independent 6D ground truth or live-camera GPU validation.
- The live JSONL interface is ROS-independent; no ROS publisher exists.
- Camera-to-robot calibration and robot control are not implemented.
