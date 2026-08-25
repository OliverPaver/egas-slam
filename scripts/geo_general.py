#!/usr/bin/env python3
"""Evaluate reconstructed geometry against a ground-truth mesh."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

try:
    from .evaluation_common import atomic_write_json, write_status_error
except ImportError:
    from evaluation_common import atomic_write_json, write_status_error


def validated_points(points, label):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} points must have shape Nx3, got {points.shape}")
    if len(points) == 0:
        raise ValueError(f"{label} point set is empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} point set contains NaN or infinity")
    return points


def kd_tree(points):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Geometry evaluation requires SciPy") from exc
    return cKDTree(points)


def completion_ratio(gt_points, rec_points, dist_th=0.03):
    gt_points = validated_points(gt_points, "ground-truth")
    rec_points = validated_points(rec_points, "reconstruction")
    distances, _ = kd_tree(rec_points).query(gt_points)
    return float(np.mean(distances < dist_th))


def accuracy_ratio(gt_points, rec_points, dist_th=0.03):
    gt_points = validated_points(gt_points, "ground-truth")
    rec_points = validated_points(rec_points, "reconstruction")
    distances, _ = kd_tree(gt_points).query(rec_points)
    return float(np.mean(distances < dist_th))


def accuracy(gt_points, rec_points):
    gt_points = validated_points(gt_points, "ground-truth")
    rec_points = validated_points(rec_points, "reconstruction")
    distances, _ = kd_tree(gt_points).query(rec_points)
    return float(np.mean(distances))


def completion(gt_points, rec_points):
    gt_points = validated_points(gt_points, "ground-truth")
    rec_points = validated_points(rec_points, "reconstruction")
    distances, _ = kd_tree(rec_points).query(gt_points)
    return float(np.mean(distances))


def sample_mesh_surface(mesh, count, rng):
    """Sample triangle area uniformly using the supplied deterministic RNG."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("Ground-truth input must contain triangular faces")
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = np.isfinite(double_areas) & (double_areas > 0)
    triangles = triangles[valid]
    double_areas = double_areas[valid]
    if len(triangles) == 0:
        raise ValueError("Ground-truth mesh has no finite, non-degenerate triangles")
    face_ids = rng.choice(len(triangles), size=count, p=double_areas / double_areas.sum())
    selected = triangles[face_ids]
    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return (
        (1.0 - root_u)[:, None] * selected[:, 0]
        + (root_u * (1.0 - v))[:, None] * selected[:, 1]
        + (root_u * v)[:, None] * selected[:, 2]
    )


def loaded_vertices(geometry, label):
    if hasattr(geometry, "vertices"):
        return validated_points(geometry.vertices, label)
    if hasattr(geometry, "geometry"):
        chunks = [
            np.asarray(item.vertices)
            for item in geometry.geometry.values()
            if hasattr(item, "vertices") and len(item.vertices)
        ]
        if chunks:
            return validated_points(np.concatenate(chunks, axis=0), label)
    raise ValueError(f"{label} file did not contain mesh or point-cloud vertices")


def transformed_points(points, transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed = (transform @ homogeneous.T).T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise ValueError("transform produced points with zero homogeneous scale")
    return validated_points(
        transformed[:, :3] / transformed[:, 3:4],
        "transformed reconstruction",
    )


def eval_pcd(
    rec_meshfile,
    gt_meshfile,
    transform=None,
    dist_thres=None,
    sample_nums=1_000_000,
    seed=0,
):
    """Compute accuracy, completion, and F1 from deterministic surface samples."""
    if sample_nums <= 0:
        raise ValueError("sample_nums must be positive")
    thresholds = [0.03] if dist_thres is None else [float(value) for value in dist_thres]
    if not thresholds or any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("all distance thresholds must be finite and positive")
    transform = np.eye(4) if transform is None else transform

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Geometry evaluation requires trimesh; install the container/runtime metric dependencies"
        ) from exc

    gt_geometry = trimesh.load(gt_meshfile, process=False)
    rec_geometry = trimesh.load(rec_meshfile, process=False)
    rec_points = transformed_points(loaded_vertices(rec_geometry, "reconstruction"), transform)
    print("recon points num:", len(rec_points))

    rng = np.random.default_rng(seed)
    rec_count = min(len(rec_points), sample_nums)
    rec_indices = rng.choice(len(rec_points), rec_count, replace=False)
    rec_sample = rec_points[rec_indices]
    if not hasattr(gt_geometry, "triangles"):
        raise ValueError("Ground-truth input must be a triangle mesh")
    gt_sample = sample_mesh_surface(gt_geometry, sample_nums, rng)

    print("compute acc")
    accuracy_cm = accuracy(gt_sample, rec_sample) * 100.0
    print("compute comp")
    completion_cm = completion(gt_sample, rec_sample) * 100.0
    results = {
        "status": "ok",
        # Preserve the historical keys while documenting their centimetre unit.
        "accuracy": accuracy_cm,
        "completion": completion_cm,
        "accuracy_cm": accuracy_cm,
        "completion_cm": completion_cm,
        "seed": int(seed),
        "requested_samples": int(sample_nums),
        "reconstruction_samples": int(rec_count),
        "ground_truth_samples": int(sample_nums),
    }
    for threshold in thresholds:
        precision = accuracy_ratio(gt_sample, rec_sample, dist_th=threshold) * 100.0
        recall = completion_ratio(gt_sample, rec_sample, dist_th=threshold) * 100.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        results[f"accuracy ratio (< {threshold})"] = precision
        results[f"completion ratio (< {threshold})"] = recall
        results[f"F1 (< {threshold})"] = f1
    print(results)
    return results


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--gt_mesh", required=True, type=Path)
    parser.add_argument("--recon_mesh", required=True, type=Path)
    parser.add_argument("--transform", default=None, type=Path)
    parser.add_argument("--dist-thres", nargs="+", type=float, default=[0.03])
    parser.add_argument("--sample-nums", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or (args.recon_mesh.parent / "geometry_results.json")
    try:
        transform = np.eye(4)
        if args.transform is not None:
            transform = np.loadtxt(args.transform, dtype=np.float64)
        results = eval_pcd(
            args.recon_mesh,
            args.gt_mesh,
            transform,
            args.dist_thres,
            args.sample_nums,
            args.seed,
        )
        atomic_write_json(output_path, results)
    except Exception as exc:
        if output_path.parent.is_dir():
            write_status_error(
                output_path,
                exc,
                reconstruction_mesh=str(args.recon_mesh),
                ground_truth_mesh=str(args.gt_mesh),
                seed=args.seed,
            )
        print(f"Geometry evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
