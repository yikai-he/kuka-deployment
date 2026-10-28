#!/usr/bin/env python3
"""Compare legacy-v7 and modular offline perception result directories on CPU."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameResult:
    frame_id: str
    original_frame_id: str
    valid: bool
    pose: np.ndarray
    mode: str | None
    lost: bool
    recovery_attempted: bool | None
    recovery_success: bool | None
    inference_ms: float | None


@dataclass(frozen=True)
class RunData:
    kind: str
    directory: Path
    frames: tuple[FrameResult, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]


def cube_rotation_symmetries() -> tuple[np.ndarray, ...]:
    """The 24 proper signed-permutation rotations of an axis-aligned cube."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                rotations.append(candidate)
    assert len(rotations) == 24
    return tuple(rotations)


CUBE_SYMMETRIES = cube_rotation_symmetries()


def rotation_geodesic_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def cube_symmetry_rotation_error_deg(modular: np.ndarray, legacy: np.ndarray) -> float:
    """min_S angle(R_modular, R_legacy S), assuming ideal cube object axes."""
    return min(
        rotation_geodesic_deg(legacy @ symmetry, modular)
        for symmetry in CUBE_SYMMETRIES
    )


def _canonical_frame_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty frame identifier")
    try:
        return str(int(text))
    except ValueError:
        return text


def _boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    raise ValueError(f"invalid boolean in {field}: {value!r}")


def _optional_boolean(row: dict[str, str], field: str) -> bool | None:
    return None if field not in row else _boolean(row[field], field=field)


def _optional_float(row: dict[str, str], field: str) -> float | None:
    if field not in row or str(row[field]).strip() == "":
        return None
    value = float(row[field])
    return value if math.isfinite(value) else None


def _load_metadata(directory: Path) -> dict[str, Any]:
    metadata = {}
    for name in ("summary.json", "metrics.json", "run_config.json"):
        path = directory / name
        if path.is_file():
            try:
                metadata[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                metadata[name] = {"error": str(exc)}
    return metadata


def load_run(directory: str | Path, kind: str) -> RunData:
    """Load either actual legacy or modular schema without assuming shared columns."""
    if kind not in {"legacy", "modular"}:
        raise ValueError("kind must be legacy or modular")
    root = Path(directory).resolve()
    csv_path, poses_path = root / "poses.csv", root / "poses.npy"
    if not csv_path.is_file() or not poses_path.is_file():
        raise FileNotFoundError(
            f"{kind} run requires poses.csv and poses.npy under {root}"
        )
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise ValueError(f"could not parse {csv_path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{csv_path} has no result rows")
    try:
        poses = np.load(poses_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load {poses_path}: {exc}") from exc
    if poses.shape != (len(rows), 4, 4):
        raise ValueError(
            f"pose shape {poses.shape} does not match {len(rows)} CSV rows"
        )

    frame_key = "frame" if kind == "legacy" else "frame_number"
    valid_key = "pose_valid" if kind == "legacy" else "valid"
    mode_key = "mode" if kind == "legacy" else "status"
    required = {frame_key, valid_key}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    warnings: list[str] = []
    if "recovery_attempted" not in rows[0]:
        warnings.append(f"{kind} CSV has no recovery_attempted column")
    if "recovery_success" not in rows[0]:
        warnings.append(f"{kind} CSV has no recovery_success column")

    results, seen = [], set()
    for index, row in enumerate(rows):
        original = str(row[frame_key]).strip()
        frame_id = _canonical_frame_id(original)
        if frame_id in seen:
            raise ValueError(f"duplicate canonical frame ID {frame_id!r} in {csv_path}")
        seen.add(frame_id)
        valid = _boolean(row[valid_key], field=valid_key)
        finite_pose = bool(np.all(np.isfinite(poses[index])))
        if valid and not finite_pose:
            raise ValueError(
                f"frame {original} is marked valid but has a non-finite pose"
            )
        if not valid and finite_pose:
            warnings.append(f"frame {original} is invalid but contains a finite pose")
        mode = row.get(mode_key) or None
        if kind == "legacy":
            lost = (mode == "LOST") or not valid
        else:
            lost = row.get("tracker_state") == "lost" or not valid
        results.append(
            FrameResult(
                frame_id,
                original,
                valid,
                poses[index],
                mode,
                lost,
                _optional_boolean(row, "recovery_attempted"),
                _optional_boolean(row, "recovery_success"),
                _optional_float(row, "inference_ms"),
            )
        )
    return RunData(kind, root, tuple(results), _load_metadata(root), tuple(warnings))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "p95": float(np.percentile(array, 95)),
    }


def _lost_intervals(frames: tuple[FrameResult, ...]) -> list[dict[str, str]]:
    intervals, start, previous = [], None, None
    for item in frames:
        if item.lost and start is None:
            start = item.frame_id
        if not item.lost and start is not None:
            intervals.append({"start": start, "end": previous})
            start = None
        previous = item.frame_id
    if start is not None:
        intervals.append({"start": start, "end": previous})
    return intervals


def compare_runs(
    legacy: RunData,
    modular: RunData,
    *,
    translation_tolerance_m: float | None = None,
    rotation_tolerance_deg: float | None = None,
) -> dict[str, Any]:
    legacy_by_id = {item.frame_id: item for item in legacy.frames}
    modular_by_id = {item.frame_id: item for item in modular.frames}
    legacy_ids, modular_ids = set(legacy_by_id), set(modular_by_id)
    matched = [
        item.frame_id for item in legacy.frames if item.frame_id in modular_by_id
    ]
    only_legacy = [
        item.original_frame_id
        for item in legacy.frames
        if item.frame_id not in modular_ids
    ]
    only_modular = [
        item.original_frame_id
        for item in modular.frames
        if item.frame_id not in legacy_ids
    ]

    validity_agreements = legacy_valid_modular_invalid = (
        legacy_invalid_modular_valid
    ) = 0
    translations, rotations = [], []
    attempt_comparable = attempt_agreements = success_comparable = (
        success_agreements
    ) = 0
    mode_comparable = mode_agreements = 0
    for frame_id in matched:
        old, new = legacy_by_id[frame_id], modular_by_id[frame_id]
        if old.valid == new.valid:
            validity_agreements += 1
        elif old.valid:
            legacy_valid_modular_invalid += 1
        else:
            legacy_invalid_modular_valid += 1
        if old.valid and new.valid:
            translations.append(
                float(np.linalg.norm(new.pose[:3, 3] - old.pose[:3, 3]))
            )
            rotations.append(
                cube_symmetry_rotation_error_deg(new.pose[:3, :3], old.pose[:3, :3])
            )
        if old.recovery_attempted is not None and new.recovery_attempted is not None:
            attempt_comparable += 1
            attempt_agreements += old.recovery_attempted == new.recovery_attempted
        if old.recovery_success is not None and new.recovery_success is not None:
            success_comparable += 1
            success_agreements += old.recovery_success == new.recovery_success
        if old.mode is not None and new.mode is not None:
            mode_comparable += 1
            # Names differ by design; compare normalized event semantics.
            old_event = (
                "register"
                if "REGISTER" in old.mode
                else "track"
                if old.mode == "TRACK"
                else "lost"
            )
            new_event = (
                "register"
                if "registered" in new.mode
                else "track"
                if new.mode == "tracked"
                else "lost"
            )
            mode_agreements += old_event == new_event

    def first_registration(run: RunData) -> str | None:
        for item in run.frames:
            if (
                item.valid
                and item.mode
                and ("REGISTER" in item.mode or "registered" in item.mode)
            ):
                return item.original_frame_id
        return next((item.original_frame_id for item in run.frames if item.valid), None)

    translation_stats, rotation_stats = _stats(translations), _stats(rotations)
    if translation_tolerance_m is not None:
        translation_stats["tolerance"] = translation_tolerance_m
        translation_stats["within_tolerance"] = sum(
            value <= translation_tolerance_m for value in translations
        )
    if rotation_tolerance_deg is not None:
        rotation_stats["tolerance"] = rotation_tolerance_deg
        rotation_stats["within_tolerance"] = sum(
            value <= rotation_tolerance_deg for value in rotations
        )
    warnings = list(legacy.warnings + modular.warnings)
    if only_legacy or only_modular:
        warnings.append(
            "frame sets differ; unmatched frames were retained in event statistics and reported"
        )
    warnings.append(
        "cube rotation comparison assumes an ideal centered cube aligned to the CAD object axes"
    )
    return {
        "inputs": {"legacy": str(legacy.directory), "modular": str(modular.directory)},
        "available_metadata": {"legacy": legacy.metadata, "modular": modular.metadata},
        "frame_correspondence": {
            "legacy_count": len(legacy.frames),
            "modular_count": len(modular.frames),
            "matched_count": len(matched),
            "matched_frame_ids": matched,
            "only_legacy": only_legacy,
            "only_modular": only_modular,
        },
        "validity": {
            "agreement_count": validity_agreements,
            "agreement_rate": validity_agreements / len(matched) if matched else None,
            "legacy_valid_modular_invalid": legacy_valid_modular_invalid,
            "legacy_invalid_modular_valid": legacy_invalid_modular_valid,
        },
        "tracking_and_recovery": {
            "first_successful_registration": {
                "legacy": first_registration(legacy),
                "modular": first_registration(modular),
            },
            "mode_agreement": {
                "comparable": mode_comparable,
                "agreements": mode_agreements,
            },
            "recovery_attempt_agreement": {
                "comparable": attempt_comparable,
                "agreements": attempt_agreements,
            },
            "recovery_success_agreement": {
                "comparable": success_comparable,
                "agreements": success_agreements,
            },
            "recovery_attempt_frames": {
                "legacy": [
                    item.original_frame_id
                    for item in legacy.frames
                    if item.recovery_attempted
                ],
                "modular": [
                    item.original_frame_id
                    for item in modular.frames
                    if item.recovery_attempted
                ],
            },
            "recovery_success_frames": {
                "legacy": [
                    item.original_frame_id
                    for item in legacy.frames
                    if item.recovery_success
                ],
                "modular": [
                    item.original_frame_id
                    for item in modular.frames
                    if item.recovery_success
                ],
            },
            "lost_intervals": {
                "legacy": _lost_intervals(legacy.frames),
                "modular": _lost_intervals(modular.frames),
            },
        },
        "translation_error_m": translation_stats,
        "cube_symmetry_rotation_error_deg": rotation_stats,
        "latency_ms": {
            "legacy": _stats(
                [
                    item.inference_ms
                    for item in legacy.frames
                    if item.inference_ms is not None
                ]
            ),
            "modular": _stats(
                [
                    item.inference_ms
                    for item in modular.frames
                    if item.inference_ms is not None
                ]
            ),
        },
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    frames, validity = report["frame_correspondence"], report["validity"]
    translation, rotation = (
        report["translation_error_m"],
        report["cube_symmetry_rotation_error_deg"],
    )
    recovery = report["tracking_and_recovery"]

    def value(item: Any) -> str:
        return (
            "N/A"
            if item is None
            else f"{item:.6g}"
            if isinstance(item, float)
            else str(item)
        )

    lines = [
        "# Perception Regression Comparison",
        "",
        "## Inputs",
        "",
        f"- Legacy: `{report['inputs']['legacy']}`",
        f"- Modular: `{report['inputs']['modular']}`",
        "",
        "## Frame correspondence",
        "",
        f"- Legacy / modular / matched: {frames['legacy_count']} / {frames['modular_count']} / {frames['matched_count']}",
        f"- Only legacy: {frames['only_legacy']}",
        f"- Only modular: {frames['only_modular']}",
        "",
        "## Validity and events",
        "",
        f"- Validity agreement rate: {value(validity['agreement_rate'])}",
        f"- Legacy-valid/modular-invalid: {validity['legacy_valid_modular_invalid']}",
        f"- Legacy-invalid/modular-valid: {validity['legacy_invalid_modular_valid']}",
        f"- First registration: {recovery['first_successful_registration']}",
        f"- LOST intervals: {recovery['lost_intervals']}",
        f"- Recovery attempts: {recovery['recovery_attempt_frames']}",
        f"- Recovery successes: {recovery['recovery_success_frames']}",
        f"- Mode agreement: {recovery['mode_agreement']}",
        f"- Recovery-attempt agreement: {recovery['recovery_attempt_agreement']}",
        f"- Recovery-success agreement: {recovery['recovery_success_agreement']}",
        "",
        "## Pose errors on mutually valid frames",
        "",
        f"- Translation count / mean / median / max / p95 [m]: {translation['count']} / {value(translation['mean'])} / {value(translation['median'])} / {value(translation['max'])} / {value(translation['p95'])}",
        f"- Cube-symmetry rotation count / mean / median / max / p95 [deg]: {rotation['count']} / {value(rotation['mean'])} / {value(rotation['median'])} / {value(rotation['max'])} / {value(rotation['p95'])}",
        "",
        "## Latency",
        "",
        f"- Legacy [ms]: {report['latency_ms']['legacy']}",
        f"- Modular [ms]: {report['latency_ms']['modular']}",
        "",
        "## Available metadata",
        "",
        f"- Legacy: {sorted(report['available_metadata']['legacy'])}",
        f"- Modular: {sorted(report['available_metadata']['modular'])}",
        "",
        "## Warnings and limitations",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(
        ["", "No overall PASS/FAIL is inferred from numerical tolerances.", ""]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--modular", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--translation-tolerance-m", type=float)
    parser.add_argument("--rotation-tolerance-deg", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, tolerance in (
        ("translation", args.translation_tolerance_m),
        ("rotation", args.rotation_tolerance_deg),
    ):
        if tolerance is not None and tolerance < 0:
            raise ValueError(f"{name} tolerance must be non-negative")
    report = compare_runs(
        load_run(args.legacy, "legacy"),
        load_run(args.modular, "modular"),
        translation_tolerance_m=args.translation_tolerance_m,
        rotation_tolerance_deg=args.rotation_tolerance_deg,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output / "comparison.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(args.output / "comparison.json")
    print(args.output / "comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
