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

The laboratory ROS1 integration uses two local processes on the GPU host. The
system Python 3.8 ROS bridge subscribes read-only to synchronized D455 topics and
serves only its latest validated frame over a user-only Unix socket. The existing
Python 3.11 inference process consumes that socket through the normal camera
backend contract. ROS is never imported into the `metaworld` environment.

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

An optional backend package exposes a factory accepting a configuration
mapping, for example:

```python
def create_camera(options):
    return MyCameraBackend(options)
```

Hardware/ROS imports occur inside the bridge, not in the core inference package.
The factory is selected as `package.module:create_camera`. This keeps `--help`,
offline replay, and CPU tests independent of camera drivers.

### Local IPC contract

The bridge listens on an `AF_UNIX` stream socket with mode `0600`. The inference
client requests a frame using its last received sequence number and a timeout.
The bridge returns either zero length when no newer frame exists or one
length-prefixed packet containing a JSON metadata header followed by contiguous
`uint8` RGB and little-endian `float32` metre-depth bytes. The header declares the
schema version, dimensions, byte counts, `K`, optical frame ID, ROS capture stamp,
source frame number, and host monotonic acquisition time. Packets and dimensions
are validated before constructing `RGBDFrame`; pickle is not used.

The bridge converts ROS optical coordinates without an axis transform: +X right,
+Y down, and +Z forward. Consequently FoundationPose continues to return
`T_camera_object` in the color optical frame. Aligned depth and CameraInfo must
have the same dimensions and optical `frame_id` as the RGB image.

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

### ROS1 D455 bridge on `yxt-X99`

The bridge and inference worker must run as the same user because the socket is
created with mode `0600`. Start the read-only bridge with the system ROS Python:

```bash
cd /home/yxt/kuka-deployment
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.1.100:11311
export ROS_IP=192.168.1.101
/usr/bin/python3 scripts/ros_rgbd_bridge.py \
  --rgb-topic /cam_1/color/image_raw \
  --depth-topic /cam_1/aligned_depth_to_color/image_raw \
  --camera-info-topic /cam_1/color/camera_info \
  --socket /tmp/kuka_rgbd.sock \
  --depth-scale 0.001 \
  --min-depth-m 0.1 \
  --max-depth-m 3.0
```

In a second terminal, run the unchanged modular pipeline in Python 3.11. Replace
the local perception profile path with the already validated RTX 3060 profile:

```bash
cd /home/yxt/kuka-deployment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate metaworld
python scripts/live_perception.py \
  --config configs/local/perception_v7_drift_effective.yaml \
  --backend kuka_deployment.perception.ros_ipc:create_camera \
  --backend-config configs/ros_ipc_backend.yaml \
  --output /home/yxt/kuka-results/live_$(date +%Y%m%d_%H%M%S)
```

Start the bridge first. Ctrl+C stops either process cleanly; stopping inference
closes its socket client, while stopping the bridge removes only its Unix socket.
Neither process publishes ROS messages or contacts robot-control interfaces.

`--save-visualization` writes `live_tracking.mp4`; it does not open a GUI. Ctrl+C
stops acquisition and closes the backend. The live run writes:

- `live_results.jsonl`: one pose/status record per processed frame;
- `live_metrics.json`: counts, recovery events, dropped frames, and mean latencies;
- `config.yaml`: the effective perception configuration.

Each JSONL record includes capture timestamp, frame number and ID, object ID,
validity, tracker status, recovery state, processing latency, end-to-end latency
when measurable, and the `T_camera_object` matrix or `null`.

`frame_age_at_inference_start_ms` measures bridge-to-inference age using the host
monotonic clock. `end_to_end_ms` includes inference through the completed pose.
`inference_ms` is CUDA-synchronized model/pipeline time. The bridge prints counts
for synchronized frames published, frames overwritten before IPC delivery, and
frames rejected during validation when it shuts down.

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

- Hardware validation of the ROS bridge and D455 stream remains pending.
- Approximate synchronization defaults to 30 ms and should be checked against the
  observed topic timestamps before reducing or increasing it.
- The bridge requires matching optical `frame_id` values for RGB, aligned depth,
  and CameraInfo; it does not transform between coordinate frames.
- Camera capture timestamps may use a device-specific clock domain.
- There is no independent 6D ground truth or live-camera GPU validation.
- The live JSONL interface remains ROS-independent; no ROS publisher exists.
- Camera-to-robot calibration and robot control are not implemented.
