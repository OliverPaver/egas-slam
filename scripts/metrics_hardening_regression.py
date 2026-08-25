#!/usr/bin/env python3
"""Focused regression checks for strict, reproducible evaluation behavior."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluation_common import (  # noqa: E402
    atomic_write_json,
    collect_indexed_images,
    require_primary_coverage,
    resolve_torch_device,
)
from geo_general import sample_mesh_surface, validated_points  # noqa: E402
from pose_evaluation_common import matched_pose_ids, read_poses  # noqa: E402


class FakeDevice:
    def __init__(self, name: str):
        self.name = name
        self.type = name.split(":", 1)[0]

    def __str__(self) -> str:
        return self.name


class FakeCuda:
    def __init__(self, available: bool, count: int = 0):
        self.available = available
        self.count = count
        self.selected = None

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def set_device(self, device) -> None:
        self.selected = str(device)


class FakeTorch:
    def __init__(self, available: bool, count: int = 0):
        self.cuda = FakeCuda(available, count)

    @staticmethod
    def device(name: str):
        return FakeDevice(name)


def expect_raises(expected, operation, message: str) -> None:
    try:
        operation()
    except expected:
        return
    raise AssertionError(message)


def write_pose(path: Path, translation: tuple[float, float, float]) -> None:
    pose = np.eye(4)
    pose[:3, 3] = translation
    np.savetxt(path, pose, fmt="%.12f")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def check_image_pairing(root: Path) -> None:
    images = root / "images"
    images.mkdir()
    (images / "frame000001.color.png").touch()
    (images / "frame000002.color.jpg").touch()
    (images / "frame000003_comp.png").touch()
    indexed = collect_indexed_images(images, required_name_token="color")
    assert sorted(indexed) == [1, 2]

    (images / "duplicate000001.color.jpeg").touch()
    expect_raises(
        ValueError,
        lambda: collect_indexed_images(images, required_name_token="color"),
        "duplicate image frame IDs must fail",
    )

    expect_raises(
        RuntimeError,
        lambda: require_primary_coverage(
            [1, 2, 3], [1, 3], primary_label="render", allow_partial=False
        ),
        "strict pairing must reject missing references",
    )
    missing, coverage = require_primary_coverage(
        [1, 2, 3], [1, 3], primary_label="render", allow_partial=True
    )
    assert missing == 1 and np.isclose(coverage, 2.0 / 3.0)


def check_devices() -> None:
    cpu_torch = FakeTorch(False)
    assert str(resolve_torch_device(cpu_torch, "auto")) == "cpu"
    expect_raises(
        RuntimeError,
        lambda: resolve_torch_device(cpu_torch, "cuda"),
        "explicit unavailable CUDA must fail",
    )

    cuda_torch = FakeTorch(True, 2)
    assert str(resolve_torch_device(cuda_torch, "cuda:1")) == "cuda:1"
    assert cuda_torch.cuda.selected == "cuda:1"
    expect_raises(
        RuntimeError,
        lambda: resolve_torch_device(cuda_torch, "cuda:2"),
        "out-of-range CUDA device must fail",
    )


def check_pose_pairing(root: Path) -> None:
    est = root / "pose_pair_est"
    gt = root / "pose_pair_gt"
    est.mkdir()
    gt.mkdir()
    write_pose(est / "frame000000.txt", (0.0, 0.0, 0.0))
    write_pose(est / "frame000001.txt", (1.0, 0.0, 0.0))
    write_pose(gt / "pose000000.txt", (0.0, 0.0, 0.0))
    write_pose(gt / "pose000001.txt", (1.0, 0.0, 0.0))
    est_poses = read_poses(est, label="estimated")
    gt_poses = read_poses(gt, label="ground-truth")
    common, coverage = matched_pose_ids(est_poses, gt_poses, allow_partial=False)
    assert common == [0, 1]
    assert coverage["estimated_pose_coverage"] == 1.0

    write_pose(est / "duplicate000001.txt", (1.0, 0.0, 0.0))
    expect_raises(
        ValueError,
        lambda: read_poses(est, label="estimated"),
        "duplicate pose IDs must fail",
    )
    (est / "duplicate000001.txt").unlink()
    invalid_pose = np.eye(4)
    invalid_pose[3, 3] = 0.0
    np.savetxt(est / "frame000002.txt", invalid_pose)
    expect_raises(
        ValueError,
        lambda: read_poses(est, label="estimated"),
        "invalid homogeneous pose rows must fail",
    )


def check_ate_and_rpe_cli(root: Path) -> None:
    val = root / "run" / "val"
    est = val / "pose"
    gt = root / "gt"
    est.mkdir(parents=True)
    gt.mkdir()
    gt_x = [0.0, 1.0, 2.0, 3.0]
    est_x = [4.0, 5.1, 6.2, 7.4]
    for idx, (gt_value, est_value) in enumerate(zip(gt_x, est_x)):
        write_pose(gt / f"pose{idx:06d}.txt", (gt_value, 0.0, 0.0))
        write_pose(est / f"frame{idx:06d}.txt", (est_value, 2.0, -1.0))

    ate = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ate_general.py"),
            "--gt_path",
            str(gt),
            "--est_path",
            str(est),
        ]
    )
    assert ate.returncode == 0, ate.stderr
    ate_json = json.loads((val / "pose_eval.json").read_text())
    assert ate_json["status"] == "ok"
    assert ate_json["matched_poses"] == 4
    assert ate_json["ate_rmse_cm"] >= ate_json["ate_mean_translation_error_cm"]
    pose_text = (val / "pose_eval.txt").read_text()
    assert "ATE mean translation error:" in pose_text
    assert "ATE actual RMSE:" in pose_text
    assert "\nATE RMSE:" not in "\n" + pose_text

    rpe = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "rpe_general.py"),
            "--gt_path",
            str(gt),
            "--est_path",
            str(est),
            "--delta",
            "1",
        ]
    )
    assert rpe.returncode == 0, rpe.stderr
    rpe_json = json.loads((val / "rpe_eval.json").read_text())
    assert rpe_json["status"] == "ok"
    assert rpe_json["rpe_pairs"] == 3

    sparse_val = root / "sparse" / "val"
    sparse_est = sparse_val / "pose"
    sparse_gt = root / "sparse_gt"
    sparse_est.mkdir(parents=True)
    sparse_gt.mkdir()
    for idx in (0, 10):
        write_pose(sparse_gt / f"pose{idx:06d}.txt", (float(idx), 0.0, 0.0))
        write_pose(sparse_est / f"frame{idx:06d}.txt", (float(idx) + 3.0, 1.0, 0.0))
    sparse_rpe = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "rpe_general.py"),
            "--gt_path",
            str(sparse_gt),
            "--est_path",
            str(sparse_est),
            "--delta",
            "10",
        ]
    )
    assert sparse_rpe.returncode == 0, sparse_rpe.stderr
    assert json.loads((sparse_val / "rpe_eval.json").read_text())["rpe_pairs"] == 1

    invalid_delta = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "rpe_general.py"),
            "--gt_path",
            str(gt),
            "--est_path",
            str(est),
            "--delta",
            "0",
        ]
    )
    assert invalid_delta.returncode == 1
    error_json = json.loads((val / "rpe_eval.json").read_text())
    assert error_json["status"] == "error"
    assert not (val / "rpe_eval.txt").exists()

    write_pose(est / "frame000099.txt", (99.0, 0.0, 0.0))
    partial = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ate_general.py"),
            "--gt_path",
            str(gt),
            "--est_path",
            str(est),
        ]
    )
    assert partial.returncode == 1
    assert json.loads((val / "pose_eval.json").read_text())["status"] == "error"
    assert not (val / "pose_eval.txt").exists()

    allowed = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ate_general.py"),
            "--gt_path",
            str(gt),
            "--est_path",
            str(est),
            "--allow-partial",
        ]
    )
    assert allowed.returncode == 0, allowed.stderr
    allowed_json = json.loads((val / "pose_eval.json").read_text())
    assert allowed_json["missing_ground_truth_poses"] == 1
    assert np.isclose(allowed_json["estimated_pose_coverage"], 0.8)


def check_geometry_sampling() -> None:
    class Mesh:
        triangles = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            ]
        )

    first = sample_mesh_surface(Mesh(), 128, np.random.default_rng(17))
    second = sample_mesh_surface(Mesh(), 128, np.random.default_rng(17))
    third = sample_mesh_surface(Mesh(), 128, np.random.default_rng(18))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    expect_raises(
        ValueError,
        lambda: validated_points(np.empty((0, 3)), "empty"),
        "empty point clouds must fail",
    )


def check_metric_failure_artifacts(root: Path) -> None:
    dataset = root / "metric_dataset"
    render = root / "metric_render"
    (dataset / "camera").mkdir(parents=True)
    (dataset / "depth").mkdir()
    render.mkdir()
    for path in (
        dataset / "camera" / "frame000001.jpg",
        dataset / "depth" / "frame000001.png",
        render / "frame000001.jpg",
    ):
        path.touch()
    atomic_write_json(render / "results.json", {"stale": True})
    metric = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "metric_general.py"),
            "--gt_path",
            str(dataset),
            "--render_path",
            str(render),
            "--device",
            "cpu",
        ]
    )
    assert metric.returncode == 1
    assert not (render / "results.json").exists()
    error = json.loads((render / "metric_error.json").read_text())
    assert error["status"] == "error"
    assert error["render_path"] == str(render)


def check_geometry_failure_artifact(root: Path) -> None:
    output = root / "geometry_error.json"
    geometry = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "geo_general.py"),
            "--gt_mesh",
            str(root / "missing_gt.ply"),
            "--recon_mesh",
            str(root / "missing_reconstruction.ply"),
            "--output",
            str(output),
        ]
    )
    assert geometry.returncode == 1
    error = json.loads(output.read_text())
    assert error["status"] == "error"
    assert error["reconstruction_mesh"].endswith("missing_reconstruction.ply")


def check_atomic_json(root: Path) -> None:
    output = root / "status.json"
    atomic_write_json(output, {"status": "ok", "value": 3})
    assert json.loads(output.read_text()) == {"status": "ok", "value": 3}
    assert not list(root.glob(".status.json.*"))


def check_no_forced_cuda_entrypoints() -> None:
    for name in ("metric.py", "metric_general.py", "metric_sampled_eval.py"):
        source = (SCRIPT_DIR / name).read_text()
        main_source = source.split('if __name__ == "__main__":', 1)[-1]
        assert "torch.cuda.set_device" not in main_source, name
        assert 'default="auto"' in main_source, name


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gps_slam_metrics_regression_") as temporary:
        root = Path(temporary)
        checks = [
            ("image pairing", lambda: check_image_pairing(root)),
            ("device resolution", check_devices),
            ("pose pairing", lambda: check_pose_pairing(root)),
            ("ATE/RPE CLI", lambda: check_ate_and_rpe_cli(root)),
            ("geometry sampling", check_geometry_sampling),
            ("metric failure artifacts", lambda: check_metric_failure_artifacts(root)),
            ("geometry failure artifact", lambda: check_geometry_failure_artifact(root)),
            ("atomic JSON", lambda: check_atomic_json(root)),
            ("no forced CUDA entrypoints", check_no_forced_cuda_entrypoints),
        ]
        for label, check in checks:
            check()
            print(f"[PASS] {label}")
    print(f"[PASS] {len(checks)}/{len(checks)} metric hardening checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
