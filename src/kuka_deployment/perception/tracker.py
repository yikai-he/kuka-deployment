"""Explicit detection, registration, tracking, and legacy-v7 recovery state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from .foundationpose import PoseEstimator
from .interfaces import ObjectPoseEstimate, RGBDFrame, TrackingStatus
from .segmenter import MaskValidationConfig, Segmentation, Segmenter, mask_rgbd_centroid, validate_mask
from .transforms import validate_transform


class TrackerState(str, Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    LOST = "lost"


class DepthCheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class PoseValidationConfig:
    min_z_m: float = 0.05
    max_z_m: float = 3.0
    max_translation_step_m: float = 0.25
    max_rotation_step_deg: float | None = None
    center_depth_residual_m: float = 0.060
    center_depth_window_px: int = 7
    register_mask_center_max_m: float = 0.0
    depth_unavailable_policy: str = "fail"
    se3_atol: float = 1e-5

    def __post_init__(self) -> None:
        if self.depth_unavailable_policy not in {"fail", "allow"}:
            raise ValueError("depth_unavailable_policy must be 'fail' or 'allow'")


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    cooldown_frames: int = 0

    def __post_init__(self) -> None:
        if self.cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")


@dataclass(frozen=True)
class PipelineConfig:
    mask: MaskValidationConfig = field(default_factory=MaskValidationConfig)
    pose: PoseValidationConfig = field(default_factory=PoseValidationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)


def project_translation_to_pixel(translation: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    x, y, z = map(float, translation)
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    return float(K[0, 0] * x / z + K[0, 2]), float(K[1, 1] * y / z + K[1, 2])


def depth_patch_median(depth_m: np.ndarray, uv: tuple[float, float] | None,
                       window_px: int) -> float | None:
    if uv is None:
        return None
    height, width = depth_m.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < width and 0 <= v < height):
        return None
    window_px = max(int(window_px), 1)
    if window_px % 2 == 0:
        window_px += 1
    radius = window_px // 2
    patch = depth_m[max(0, v-radius):min(height, v+radius+1),
                    max(0, u-radius):min(width, u+radius+1)]
    valid = patch[patch > 0]
    return None if valid.size == 0 else float(np.median(valid))


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    return math.degrees(math.acos(float(cosine)))


def validate_registered_pose(
    pose: np.ndarray, config: PoseValidationConfig, *, frame: RGBDFrame | None = None,
    mask: np.ndarray | None = None, diagnostics: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    details = diagnostics if diagnostics is not None else {}
    details.update({"depth_check_status": DepthCheckStatus.UNAVAILABLE.value,
                    "register_mask_center_error_m": None})
    try:
        transform = validate_transform(np.asarray(pose), atol=config.se3_atol)
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    if not config.min_z_m <= transform[2, 3] <= config.max_z_m:
        return False, "pose depth outside configured range"
    if frame is None or mask is None or config.register_mask_center_max_m <= 0:
        return True, None
    centroid = mask_rgbd_centroid(frame, mask)
    if centroid is None:
        if config.depth_unavailable_policy == "fail":
            return False, "registration mask depth unavailable"
        return True, None
    error = float(np.linalg.norm(transform[:3, 3] - centroid))
    details["register_mask_center_error_m"] = error
    if error > config.register_mask_center_max_m:
        details["depth_check_status"] = DepthCheckStatus.FAILED.value
        return False, "registered pose disagrees with mask RGB-D centroid"
    details["depth_check_status"] = DepthCheckStatus.PASSED.value
    return True, None


def detect_tracking_failure(
    current: np.ndarray, previous: np.ndarray | None, config: PoseValidationConfig,
    *, frame: RGBDFrame | None = None, cooldown_remaining: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    details = diagnostics if diagnostics is not None else {}
    details.update({"step_translation_m": 0.0, "center_depth_m": None,
                    "center_depth_residual_m": None,
                    "depth_check_status": DepthCheckStatus.UNAVAILABLE.value})
    valid, reason = validate_registered_pose(current, config)
    if not valid:
        return True, reason
    reasons: list[str] = []
    if previous is not None:
        step = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
        details["step_translation_m"] = step
        if cooldown_remaining <= 0 and config.max_translation_step_m > 0 and step > config.max_translation_step_m:
            reasons.append("translation jump exceeded configured threshold")
        if config.max_rotation_step_deg is not None and config.max_rotation_step_deg > 0:
            if _rotation_distance_deg(previous, current) > config.max_rotation_step_deg:
                reasons.append("rotation jump exceeded configured threshold")
    if config.center_depth_residual_m > 0 and frame is not None:
        uv = project_translation_to_pixel(current[:3, 3], frame.K)
        observed = depth_patch_median(frame.depth_m, uv, config.center_depth_window_px)
        if observed is None:
            if config.depth_unavailable_policy == "fail":
                reasons.append("projected pose-center depth unavailable")
        else:
            residual = abs(observed - float(current[2, 3]))
            details["center_depth_m"] = observed
            details["center_depth_residual_m"] = residual
            if residual > config.center_depth_residual_m:
                details["depth_check_status"] = DepthCheckStatus.FAILED.value
                reasons.append("projected pose-center depth inconsistent")
            else:
                details["depth_check_status"] = DepthCheckStatus.PASSED.value
    return bool(reasons), "; ".join(reasons) if reasons else None


class PerceptionPipeline:
    def __init__(self, object_id: str, segmenter: Segmenter, estimator: PoseEstimator,
                 config: PipelineConfig | None = None, *, initial_mask: np.ndarray | None = None) -> None:
        self.object_id, self.segmenter, self.estimator = object_id, segmenter, estimator
        self.config = config or PipelineConfig()
        self.initial_mask = initial_mask
        self.state = TrackerState.UNINITIALIZED
        self._tracking_pose: np.ndarray | None = None
        self._last_accepted_pose: np.ndarray | None = None
        self.lost_streak = 0
        self.cooldown_remaining = 0
        self.last_diagnostics: dict[str, Any] = {}

    def _result(self, frame: RGBDFrame, status: TrackingStatus, pose: np.ndarray | None,
                confidence: float | None = None, reason: str | None = None) -> ObjectPoseEstimate:
        self.last_diagnostics.update({"state": self.state.value, "lost_streak": self.lost_streak,
                                      "cooldown_remaining": self.cooldown_remaining})
        return ObjectPoseEstimate(
            object_id=self.object_id, timestamp_ns=frame.timestamp_ns, frame_id=frame.frame_id,
            T_frame_object=None if pose is None else pose.copy(), valid=pose is not None,
            status=status, source_frame_number=frame.frame_number,
            detector_confidence=confidence, failure_reason=reason,
        )

    def _hint_uv(self, frame: RGBDFrame) -> tuple[float, float] | None:
        if self._last_accepted_pose is None:
            return None
        return project_translation_to_pixel(self._last_accepted_pose[:3, 3], frame.K)

    def _detect(self, frame: RGBDFrame) -> Segmentation | None:
        contextual = getattr(self.segmenter, "detect_with_context", None)
        detection = (contextual(frame, hint_uv=self._hint_uv(frame), lost_streak=self.lost_streak)
                     if contextual is not None else self.segmenter.detect(frame))
        if detection is not None:
            self.last_diagnostics.update(detection.metadata)
        self.last_diagnostics["candidate_count"] = len(getattr(self.segmenter, "last_candidates", []))
        return detection

    def _accept(self, pose: np.ndarray) -> None:
        accepted = np.asarray(pose).copy()
        self.state = TrackerState.TRACKING
        self._tracking_pose = accepted
        self._last_accepted_pose = accepted.copy()
        self.lost_streak = 0
        self.cooldown_remaining = self.config.recovery.cooldown_frames

    def _enter_lost(self) -> None:
        self.state = TrackerState.LOST
        self._tracking_pose = None
        self.lost_streak = max(1, self.lost_streak + 1)

    def _register_mask(self, frame: RGBDFrame, mask: np.ndarray,
                       confidence: float | None, recovery: bool) -> ObjectPoseEstimate:
        try:
            pose = self.estimator.register(frame, mask)
        except Exception as exc:
            self._enter_lost()
            return self._result(frame, TrackingStatus.REGISTRATION_FAILED, None, confidence, str(exc))
        valid, reason = validate_registered_pose(
            pose, self.config.pose, frame=frame, mask=mask, diagnostics=self.last_diagnostics)
        if not valid:
            self._enter_lost()
            return self._result(frame, TrackingStatus.REGISTRATION_FAILED, None, confidence, reason)
        self._accept(pose)
        self.last_diagnostics["recovery_success"] = recovery
        return self._result(frame, TrackingStatus.REREGISTERED if recovery else TrackingStatus.REGISTERED,
                            self._tracking_pose, confidence)

    def _register(self, frame: RGBDFrame, *, recovery: bool) -> ObjectPoseEstimate:
        self.last_diagnostics["recovery_attempted"] = recovery
        detection = self._detect(frame)
        detection_reason = "no segmentation candidate"
        if detection is not None:
            # The v7 selector already applied its exact physical geometry gates;
            # do not add generic pixel-geometry gates that would change parity.
            if detection.metadata.get("v7_candidate_validated", False):
                valid, detection_reason = True, None
            else:
                valid, detection_reason = validate_mask(frame, detection.mask, self.config.mask)
            if valid:
                result = self._register_mask(frame, detection.mask, detection.confidence, recovery)
                if result.valid or recovery or self.initial_mask is None:
                    return result
            elif recovery or self.initial_mask is None:
                self._enter_lost()
                return self._result(frame, TrackingStatus.DETECTION_FAILED, None,
                                    detection.confidence, detection_reason)
        if not recovery and self.initial_mask is not None:
            self.last_diagnostics["initial_mask_fallback"] = True
            fallback = np.asarray(self.initial_mask, dtype=bool)
            if fallback.shape != frame.depth_m.shape:
                self._enter_lost()
                return self._result(frame, TrackingStatus.DETECTION_FAILED, None,
                                    reason="initial fallback mask shape mismatch")
            return self._register_mask(frame, fallback, None, False)
        self._enter_lost()
        return self._result(frame, TrackingStatus.DETECTION_FAILED, None,
                            None if detection is None else detection.confidence, detection_reason)

    def try_reregister(self, frame: RGBDFrame) -> ObjectPoseEstimate:
        return self._register(frame, recovery=True)

    def process_frame(self, frame: RGBDFrame) -> ObjectPoseEstimate:
        self.last_diagnostics = {"recovery_attempted": False, "recovery_success": False,
                                 "depth_check_status": DepthCheckStatus.UNAVAILABLE.value,
                                 "failure_stage": None}
        if self.state is TrackerState.UNINITIALIZED:
            return self._register(frame, recovery=False)
        if self.state is TrackerState.LOST:
            if not self.config.recovery.enabled:
                self.lost_streak += 1
                self.last_diagnostics["failure_stage"] = "lost_recovery_disabled"
                return self._result(frame, TrackingStatus.TRACKING_FAILED, None,
                                    reason="recovery disabled while tracker is LOST")
            return self.try_reregister(frame)
        # TRACKING is the only state from which estimator.track may be called.
        assert self._tracking_pose is not None
        try:
            candidate = np.asarray(self.estimator.track(frame))
            failed, reason = detect_tracking_failure(
                candidate, self._last_accepted_pose, self.config.pose, frame=frame,
                cooldown_remaining=self.cooldown_remaining, diagnostics=self.last_diagnostics)
            failure_stage = "tracking_validation"
        except Exception as exc:
            failed, reason = True, str(exc)
            failure_stage = "tracking_exception"
        if not failed:
            self._tracking_pose = candidate.copy()
            self._last_accepted_pose = candidate.copy()
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            return self._result(frame, TrackingStatus.TRACKED, self._tracking_pose)
        if not self.config.recovery.enabled:
            self._enter_lost()
            self.last_diagnostics["failure_stage"] = failure_stage
            return self._result(frame, TrackingStatus.TRACKING_FAILED, None, reason=reason)
        recovered = self.try_reregister(frame)
        if recovered.valid:
            return recovered
        combined = f"tracking failed ({reason}); recovery failed ({recovered.failure_reason})"
        return self._result(frame, TrackingStatus.TRACKING_FAILED, None,
                            recovered.detector_confidence, combined)
