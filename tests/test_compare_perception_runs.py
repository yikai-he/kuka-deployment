import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_perception_runs import (
    CUBE_SYMMETRIES, compare_runs, cube_symmetry_rotation_error_deg, load_run, main,
    rotation_geodesic_deg,
)


def transform(translation=(0, 0, 1), rotation=None):
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = np.eye(3) if rotation is None else rotation
    value[:3, 3] = translation
    return value


def write_run(root: Path, kind: str, frame_ids, poses, valid=None, *, optional=True):
    root.mkdir()
    valid = valid if valid is not None else [bool(np.isfinite(pose).all()) for pose in poses]
    if kind == "legacy":
        fields = ["frame", "pose_valid", "mode", "inference_ms"]
        if optional:
            fields += ["recovery_attempted", "recovery_success"]
        rows = []
        for index, frame_id in enumerate(frame_ids):
            row = {"frame": frame_id, "pose_valid": int(valid[index]),
                   "mode": "REGISTER_YOLO" if index == 0 else "TRACK", "inference_ms": 10 + index}
            if optional:
                row.update(recovery_attempted=0, recovery_success=0)
            rows.append(row)
        (root / "summary.json").write_text(json.dumps({"frames": len(rows)}))
    else:
        fields = ["frame_number", "valid", "status", "tracker_state", "inference_ms",
                  "recovery_attempted", "recovery_success"]
        rows = [{"frame_number": frame_id, "valid": valid[index],
                 "status": "registered" if index == 0 else "tracked",
                 "tracker_state": "tracking" if valid[index] else "lost",
                 "inference_ms": 11 + index, "recovery_attempted": False,
                 "recovery_success": False} for index, frame_id in enumerate(frame_ids)]
        (root / "metrics.json").write_text(json.dumps({"frames": len(rows)}))
    with (root / "poses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    np.save(root / "poses.npy", np.stack(poses))
    return root


def compare(tmp_path, legacy_poses, modular_poses, *, legacy_ids=None, modular_ids=None,
            legacy_valid=None, modular_valid=None, legacy_optional=True):
    legacy_ids = legacy_ids or list(range(len(legacy_poses)))
    modular_ids = modular_ids or list(range(len(modular_poses)))
    legacy = write_run(tmp_path / "legacy", "legacy", legacy_ids, legacy_poses,
                       legacy_valid, optional=legacy_optional)
    modular = write_run(tmp_path / "modular", "modular", modular_ids, modular_poses, modular_valid)
    return compare_runs(load_run(legacy, "legacy"), load_run(modular, "modular"))


def test_identical_trajectories_have_zero_error(tmp_path):
    report = compare(tmp_path, [transform(), transform((0.1, 0, 1))],
                     [transform(), transform((0.1, 0, 1))])
    assert report["translation_error_m"]["max"] == pytest.approx(0)
    assert report["cube_symmetry_rotation_error_deg"]["max"] == pytest.approx(0)
    assert report["validity"]["agreement_rate"] == 1


def test_known_translation_offset(tmp_path):
    report = compare(tmp_path, [transform()], [transform((0.01, 0, 1))])
    assert report["translation_error_m"]["mean"] == pytest.approx(0.01)


def test_cube_symmetry_rotation_has_zero_error():
    assert cube_symmetry_rotation_error_deg(CUBE_SYMMETRIES[7], np.eye(3)) == pytest.approx(0)


def test_non_symmetry_rotation_is_nonzero():
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert cube_symmetry_rotation_error_deg(rotation, np.eye(3)) == pytest.approx(30)
    assert rotation_geodesic_deg(np.eye(3), rotation) == pytest.approx(30)


def test_validity_disagreement_excludes_pose_error(tmp_path):
    nan_pose = np.full((4, 4), np.nan, np.float32)
    report = compare(tmp_path, [transform(), nan_pose], [nan_pose, transform()],
                     legacy_valid=[True, False], modular_valid=[False, True])
    assert report["validity"]["legacy_valid_modular_invalid"] == 1
    assert report["validity"]["legacy_invalid_modular_valid"] == 1
    assert report["translation_error_m"]["count"] == 0


def test_mismatched_frame_ids_are_reported_not_silently_skipped(tmp_path):
    report = compare(tmp_path, [transform(), transform()], [transform(), transform()],
                     legacy_ids=["000001", "000002"], modular_ids=[2, 3])
    frames = report["frame_correspondence"]
    assert frames["matched_frame_ids"] == ["2"]
    assert frames["only_legacy"] == ["000001"] and frames["only_modular"] == ["3"]
    assert any("frame sets differ" in warning for warning in report["warnings"])


def test_missing_optional_legacy_metrics_are_warned(tmp_path):
    report = compare(tmp_path, [transform()], [transform()], legacy_optional=False)
    assert report["tracking_and_recovery"]["recovery_attempt_agreement"]["comparable"] == 0
    assert any("recovery_attempted" in warning for warning in report["warnings"])


def test_nan_matrix_is_accepted_for_explicitly_invalid_frame(tmp_path):
    nan_pose = np.full((4, 4), np.nan, np.float32)
    report = compare(tmp_path, [nan_pose], [nan_pose], legacy_valid=[False], modular_valid=[False])
    assert report["validity"]["agreement_rate"] == 1
    assert report["translation_error_m"]["count"] == 0


def test_different_csv_schemas_and_leading_zero_ids_correspond(tmp_path):
    report = compare(tmp_path, [transform()], [transform()], legacy_ids=["000007"], modular_ids=[7])
    assert report["frame_correspondence"]["matched_frame_ids"] == ["7"]
    assert report["available_metadata"]["legacy"]["summary.json"]["frames"] == 1
    assert report["available_metadata"]["modular"]["metrics.json"]["frames"] == 1


def test_missing_and_malformed_files_raise_clear_errors(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(FileNotFoundError, match="poses.csv and poses.npy"):
        load_run(empty, "legacy")
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "poses.csv").write_text("frame,pose_valid\n0,1\n")
    (bad / "poses.npy").write_bytes(b"not a numpy file")
    with pytest.raises(ValueError, match="could not load"):
        load_run(bad, "legacy")


def test_valid_row_with_nan_pose_is_rejected(tmp_path):
    nan_pose = np.full((4, 4), np.nan, np.float32)
    root = write_run(tmp_path / "legacy", "legacy", [0], [nan_pose], valid=[True])
    with pytest.raises(ValueError, match="marked valid"):
        load_run(root, "legacy")


def test_cli_generates_json_and_markdown_reports(tmp_path, monkeypatch):
    legacy = write_run(tmp_path / "legacy", "legacy", [0], [transform()])
    modular = write_run(tmp_path / "modular", "modular", [0], [transform()])
    output = tmp_path / "comparison"
    monkeypatch.setattr("sys.argv", ["compare_perception_runs.py", "--legacy", str(legacy),
                                    "--modular", str(modular), "--output", str(output)])
    assert main() == 0
    assert (output / "comparison.json").is_file()
    assert "Perception Regression Comparison" in (output / "comparison.md").read_text()
