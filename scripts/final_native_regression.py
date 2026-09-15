#!/usr/bin/env python3
"""Exercise native estimated-pose isolation and fail-closed saved-pose loading.

Run inside the Thor container. Creates only a fresh, explicitly named output
directory and never edits the source dataset or an existing experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build-dir", default="build_thor")
    parser.add_argument("--hash-profile", choices=["medium_128k", "legacy_large", "legacy_pool_2m"], default="medium_128k")
    parser.add_argument("--template", type=Path, default=PROJECT / "configs/final/replica/office0.yaml")
    parser.add_argument("--frames", default=40, type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing validation: {output}")
    if not output.is_relative_to(Path("/workspace/runs")) or output == Path("/workspace/runs"):
        raise RuntimeError("Validation output must be a child of /workspace/runs")
    if not 30 <= args.frames <= 120:
        raise ValueError("Use 30–120 input frames for the native regression")
    output.mkdir(parents=True)
    binary = (PROJECT / args.build_dir / "slam_trainer").resolve()
    template = args.template.resolve()
    base = yaml.safe_load(template.read_text())
    reader = base["READER"]
    original_input = Path(reader["input_dir"])
    reader.update(preload_images=False, end_frame=args.frames - 1, scene_scale=1.0)
    pipe = base["PIPE"]
    for key in ("benchmark_protocol", "benchmark_role"):
        pipe.pop(key, None)
    pipe.update(eval_after_train=False, save_after_train=True, random_seed=20260908,
                export_online_end_trajectory=True)
    pipe["TSDF"].update(engine="basic", use_gt_pose=False, input_mode="on_demand",
                         input_prefetch=False, saved_mesh="")
    pipe["TSDF"]["multi"]["hash_profile"] = args.hash_profile
    # This test isolates BasicEngine/unified-Gaussian input handling regardless
    # of the proposed-method settings in the user-supplied portable template.
    pipe["gaussian_submaps"]["enable"] = False
    pipe["pose_graph"].update(candidate_detection=False, apply_mode="disabled",
                              online_updates=False, dry_run=False, solve_only=False,
                              warm_loop_observations=False)
    records = []

    def run(name: str, cfg: dict, expected_error: str | None = None) -> Path:
        configuration = output / f"{name}.yaml"
        configuration.write_text(yaml.safe_dump(cfg, sort_keys=False))
        started = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run([str(binary), str(configuration)], stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        text = (output / f"{name}.log").read_text(errors="replace")
        passed = result.returncode == 0 if expected_error is None else (
            result.returncode != 0 and expected_error in text)
        records.append(dict(name=name, returncode=result.returncode,
                            wall_seconds=time.monotonic() - started, passed=passed,
                            expected_error=expected_error))
        (output / "results.json").write_text(json.dumps({"tests": records}, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"Native regression failed: {name}; see {output / (name + '.log')}")
        print(f"PASS {name}", flush=True)
        return Path(cfg["workspace_dir"])

    poses = {}
    for variant in ("missing_gt", "poisoned_gt"):
        inputs = output / "inputs" / variant
        (inputs / "camera").mkdir(parents=True)
        (inputs / "depth").mkdir()
        for index in range(args.frames):
            frame = f"{index:06d}"
            (inputs / "camera" / f"frame{frame}.jpg").symlink_to(original_input / "camera" / f"frame{frame}.jpg")
            (inputs / "depth" / f"depth{frame}.png").symlink_to(original_input / "depth" / f"depth{frame}.png")
            if variant == "poisoned_gt":
                (inputs / "camera" / f"pose{frame}.txt").write_text("not a valid ground-truth pose\n")
        cfg = copy.deepcopy(base)
        cfg["READER"]["input_dir"] = str(inputs)
        cfg["workspace_dir"] = str(output / "outputs" / variant)
        workspace = run(variant, cfg)
        matrices = sorted((workspace / "val/pose").glob("frame*.txt"))
        if len(matrices) != args.frames:
            raise AssertionError(f"Incomplete native output in {variant}")
        poses[variant] = np.stack([np.loadtxt(path) for path in matrices])
        if not np.isfinite(poses[variant]).all():
            raise AssertionError("Non-finite native trajectory")
        if "[POSE SOURCE] estimated; configured scene scale: 1.100000" not in (output / f"{variant}.log").read_text():
            raise AssertionError("Estimated run did not use the fixed configured scale")

    maximum_pose_difference = float(np.max(np.abs(poses["missing_gt"] - poses["poisoned_gt"])))
    if maximum_pose_difference > 1e-4:
        raise AssertionError(f"Unexpected trajectory change: {maximum_pose_difference}")

    # Reload the saved map in two fresh processes. Sampling must only remove
    # views, not change the pixel values of views present in both evaluations.
    eval_base = copy.deepcopy(base)
    eval_base["work_mode"] = "eval"
    eval_base["READER"]["input_dir"] = str(output / "inputs/missing_gt")
    eval_base["workspace_dir"] = str(output / "outputs/missing_gt")
    eval_base["PIPE"].update(eval_image_extension=".png", eval_write_auxiliary_images=False,
                             eval_stride=1, eval_start_offset=0, eval_max_frames=0)
    sampled = sorted({0, 7, 29, args.frames - 1})
    for name, indices in (("full_eval", None), ("sampled_eval", sampled)):
        cfg = copy.deepcopy(eval_base)
        cfg["PIPE"]["eval_path"] = name
        if indices is not None:
            cfg["PIPE"]["eval_frame_indices"] = indices
        run(name, cfg)
        images = list((Path(cfg["workspace_dir"]) / name).rglob("*.png"))
        if len(images) != (args.frames if indices is None else len(indices)):
            raise AssertionError(f"Incorrect evaluation output count for {name}")
        if any(p.parent.name != "render" for p in images):
            raise AssertionError("Auxiliary evaluation images were not disabled")
    for index in sampled:
        pattern = f"frame{index:06d}_iter*.color.png"
        full = next((output / "outputs/missing_gt/full_eval/render").glob(pattern))
        subset = next((output / "outputs/missing_gt/sampled_eval/render").glob(pattern))
        if full.read_bytes() != subset.read_bytes():
            raise AssertionError(f"Evaluation selection changed rendered pixels for frame {index}")
    bad = copy.deepcopy(eval_base)
    bad["PIPE"]["eval_frame_indices"] = [0, 0]
    run("duplicate_eval_indices", bad, "must be sorted, unique and nonnegative")
    bad["PIPE"]["eval_frame_indices"] = [args.frames]
    run("absent_eval_index", bad, "Requested evaluation frame is absent")

    for failure, replacement, expected in [
        ("missing_saved_pose", None, "Incomplete saved SLAM trajectory"),
        ("malformed_saved_pose", "1 2\n3 4\n", "Invalid saved 4x4 SLAM pose"),
        ("nonfinite_saved_pose", "nan 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", "matrix value"),
    ]:
        cfg = copy.deepcopy(base)
        cfg["work_mode"] = "eval"
        cfg["READER"]["input_dir"] = str(output / "inputs/missing_gt")
        cfg["workspace_dir"] = str(output / "outputs" / failure)
        destination = Path(cfg["workspace_dir"]) / "val/pose"
        destination.mkdir(parents=True)
        for index, matrix in enumerate(poses["missing_gt"]):
            if index == args.frames - 1:
                if replacement is not None:
                    (destination / f"frame{index:06d}.txt").write_text(replacement)
            else:
                np.savetxt(destination / f"frame{index:06d}.txt", matrix)
        run(failure, cfg, expected)

    result = dict(passed=True, tests=records, frames=args.frames,
                  binary_sha256=hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
                  maximum_pose_difference=maximum_pose_difference)
    result["sampled_full_render_bytes_equal"] = True
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
