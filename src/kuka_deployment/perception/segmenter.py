"""Segmentation adapters and the legacy-v7 red-cube candidate selector."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .interfaces import RGBDFrame


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_ or self.mask.ndim != 2:
            raise ValueError("segmentation mask must be a 2D boolean array")
        if not 0 <= self.confidence <= 1:
            raise ValueError("segmentation confidence must be in [0, 1]")


class Segmenter(Protocol):
    def detect(self, frame: RGBDFrame) -> Segmentation | None: ...


@dataclass(frozen=True)
class MaskValidationConfig:
    """Generic post-selection checks used for injected/mock segmenters."""
    min_pixels: int = 50
    max_pixels: int = 100_000
    min_valid_depth_fraction: float = 0.50
    min_median_depth_m: float = 0.05
    max_median_depth_m: float = 3.0
    min_fill_ratio: float = 0.30
    max_aspect_ratio: float = 3.0


@dataclass(frozen=True)
class RedCubeCandidateConfig:
    """Red-cube-specific v7 eligibility and ranking parameters."""
    class_id: int = 0
    target_size_m: float = 0.040
    min_object_size_m: float = 0.015
    max_object_size_m: float = 0.080
    min_component_area: int = 20
    max_aspect_ratio: float = 2.5
    min_candidate_depth_ratio: float = 0.35
    recovery_hint_weight: float = 0.35
    recovery_hint_max_px: float = 150.0
    lost_hint_expand_px: float = 30.0
    hint_fallback_lost_streak: int = 3


@dataclass(frozen=True)
class YoloInferenceConfig:
    confidence: float = 0.30
    iou: float = 0.50
    image_size: int = 640
    max_detections: int = 5
    device: str = "cuda:0"


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component as a boolean mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    cv2 = __import__("cv2")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def analyze_mask_geometry(mask: np.ndarray, depth_m: np.ndarray | None = None,
                          K: np.ndarray | None = None) -> dict[str, Any] | None:
    """Analyze a mask, including exact v7 physical geometry when depth/K are given."""
    binary = mask > 0
    ys, xs = np.nonzero(binary)
    area = int(xs.size)
    if area == 0:
        return None
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width, height = right - left + 1, bottom - top + 1
    result: dict[str, Any] = {
        "area": area, "pixels": float(area), "bbox": (left, top, width, height),
        "center": (float(xs.mean()), float(ys.mean())),
        "fill_ratio": float(area / (width * height)),
        "aspect_ratio": float(max(width / height, height / width)),
    }
    if depth_m is None or K is None:
        return result
    if depth_m.shape != binary.shape or np.asarray(K).shape != (3, 3):
        raise ValueError("mask, depth, and K dimensions are incompatible")
    valid_depth = binary & (depth_m > 0)
    valid_count = int(valid_depth.sum())
    if valid_count == 0:
        return None
    z = float(np.median(depth_m[valid_depth]))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    width_m, height_m = width * z / fx, height * z / fy
    result.update({
        "depth_m": z, "valid_depth_ratio": float(valid_count / area),
        "width_m": float(width_m), "height_m": float(height_m),
        "equiv_side_m": float(math.sqrt(area) * z / math.sqrt(fx * fy)),
        "aspect_ratio": float(max(width_m, height_m) / max(min(width_m, height_m), 1e-6)),
    })
    return result


def mask_rgbd_centroid(frame: RGBDFrame, mask: np.ndarray) -> np.ndarray | None:
    valid = (mask > 0) & (frame.depth_m > 0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None
    z = frame.depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - frame.K[0, 2]) * z / frame.K[0, 0]
    y = (ys.astype(np.float32) - frame.K[1, 2]) * z / frame.K[1, 1]
    return np.array([np.median(x), np.median(y), np.median(z)], dtype=np.float32)


def validate_mask(frame: RGBDFrame, mask: np.ndarray,
                  config: MaskValidationConfig) -> tuple[bool, str | None]:
    if mask.shape != frame.depth_m.shape:
        return False, "mask/image shape mismatch"
    geometry = analyze_mask_geometry(mask)
    if geometry is None or not config.min_pixels <= geometry["area"] <= config.max_pixels:
        return False, "mask pixel count outside configured range"
    if geometry["fill_ratio"] < config.min_fill_ratio or geometry["aspect_ratio"] > config.max_aspect_ratio:
        return False, "mask geometry outside configured range"
    depths = frame.depth_m[mask > 0]
    valid = depths > 0
    if depths.size == 0 or valid.mean() < config.min_valid_depth_fraction:
        return False, "insufficient valid depth in mask"
    median = float(np.median(depths[valid]))
    if not config.min_median_depth_m <= median <= config.max_median_depth_m:
        return False, "masked depth outside configured range"
    return True, None


def select_red_cube_candidate(
    masks: np.ndarray, confidences: np.ndarray, classes: np.ndarray, frame: RGBDFrame,
    config: RedCubeCandidateConfig, *, hint_uv: tuple[float, float] | None = None,
    lost_streak: int = 0,
) -> tuple[Segmentation | None, list[dict[str, Any]]]:
    """Select candidates using the exact legacy-v7 gates and scoring equation."""
    cv2 = __import__("cv2")
    height, width = frame.depth_m.shape
    allowed_hint_px = None
    if hint_uv is not None and config.recovery_hint_max_px > 0:
        allowed_hint_px = min(
            config.recovery_hint_max_px + config.lost_hint_expand_px * max(int(lost_streak), 0),
            math.hypot(width, height),
        )
    candidates: list[dict[str, Any]] = []
    for index in range(len(confidences)):
        if int(classes[index]) != config.class_id:
            continue
        probability = np.asarray(masks[index], dtype=np.float32)
        if probability.shape != (height, width):
            probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = keep_largest_component(probability >= 0.5)
        geometry = analyze_mask_geometry(mask, frame.depth_m, frame.K)
        if geometry is None:
            continue
        confidence = float(confidences[index])
        hint_distance_px = None
        valid_hint = True
        if hint_uv is not None:
            hint_distance_px = float(math.hypot(
                geometry["center"][0] - hint_uv[0], geometry["center"][1] - hint_uv[1]))
            if allowed_hint_px is not None:
                valid_hint = hint_distance_px <= allowed_hint_px
        eligible = (
            config.min_object_size_m <= geometry["equiv_side_m"] <= config.max_object_size_m
            and geometry["area"] >= config.min_component_area
            and geometry["aspect_ratio"] <= config.max_aspect_ratio
            and geometry["valid_depth_ratio"] >= config.min_candidate_depth_ratio
            and valid_hint
        )
        size_error = abs(math.log(max(geometry["equiv_side_m"], 1e-6) / config.target_size_m))
        aspect_error = abs(math.log(max(geometry["aspect_ratio"], 1e-6)))
        score = -math.log(max(confidence, 1e-6)) + 0.70 * size_error + 0.15 * aspect_error
        if hint_distance_px is not None and config.recovery_hint_weight > 0:
            score += config.recovery_hint_weight * hint_distance_px / 100.0
        candidates.append({
            "prediction_index": index, "confidence": confidence,
            "hint_distance_px": hint_distance_px, "allowed_hint_px": allowed_hint_px,
            "eligible": bool(eligible), "score": float(score), "mask": mask, **geometry,
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible and hint_uv is not None and lost_streak >= config.hint_fallback_lost_streak:
        for candidate in candidates:
            geometry_ok = (
                config.min_object_size_m <= candidate["equiv_side_m"] <= config.max_object_size_m
                and candidate["area"] >= config.min_component_area
                and candidate["aspect_ratio"] <= config.max_aspect_ratio
                and candidate["valid_depth_ratio"] >= config.min_candidate_depth_ratio
            )
            if not geometry_ok:
                continue
            fallback = dict(candidate)
            fallback["score"] = (
                -math.log(max(fallback["confidence"], 1e-6))
                + 0.70 * abs(math.log(max(fallback["equiv_side_m"], 1e-6) / config.target_size_m))
            )
            fallback["eligible"] = True
            fallback["fallback_without_hint"] = True
            eligible.append(fallback)
    if not eligible:
        return None, candidates
    best = min(eligible, key=lambda candidate: candidate["score"])
    info = {key: value for key, value in best.items() if key != "mask"}
    info["mask_pixels"] = int(best["mask"].sum())
    info["mask_median_depth_m"] = float(best["depth_m"])
    info["v7_candidate_validated"] = True
    return Segmentation(best["mask"].copy(), best["confidence"], info), candidates


class YoloSegmenter:
    """Lazy Ultralytics wrapper. It never downloads weights."""
    def __init__(self, weights: str | Path, *, confidence: float = 0.30,
                 class_id: int | None = 0, device: str = "cuda:0", iou: float = 0.50,
                 image_size: int = 640, max_detections: int = 5,
                 candidate_config: RedCubeCandidateConfig | None = None) -> None:
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        self.inference = YoloInferenceConfig(confidence, iou, image_size, max_detections, device)
        self.candidates = candidate_config or RedCubeCandidateConfig(class_id=0 if class_id is None else class_id)
        self._model = None
        self.last_candidates: list[dict[str, Any]] = []

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the optional Ultralytics dependency on the GPU host") from exc
            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, frame: RGBDFrame) -> Segmentation | None:
        return self.detect_with_context(frame)

    def detect_with_context(self, frame: RGBDFrame, *, hint_uv: tuple[float, float] | None = None,
                            lost_streak: int = 0) -> Segmentation | None:
        cv2 = __import__("cv2")
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        results = self._load().predict(
            source=bgr, conf=self.inference.confidence, iou=self.inference.iou,
            imgsz=self.inference.image_size, device=self.inference.device,
            max_det=self.inference.max_detections, retina_masks=True, verbose=False,
        )
        if not results or results[0].boxes is None or results[0].masks is None or len(results[0].boxes) == 0:
            self.last_candidates = []
            return None
        result = results[0]
        selected, self.last_candidates = select_red_cube_candidate(
            result.masks.data.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
            result.boxes.cls.detach().cpu().numpy().astype(int),
            frame, self.candidates, hint_uv=hint_uv, lost_streak=lost_streak,
        )
        return selected
