#!/usr/bin/env python3
"""
Small bounded training sweep for hybrid Gaussian submap quality.
"""
import argparse
import csv
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from hybrid_validation import PROJECT_ROOT, CONFIG_DIR, GT_HALF, GT_FULL, ATE_SCRIPT, METRIC_SCRIPT, load_base, run_cmd, evaluate_ate, run_quality_eval


OUTPUT_ROOT = PROJECT_ROOT / "output" / "hybrid_training_sweep"


def deep_update(base: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def variants() -> list[tuple[str, dict]]:
    common = {
        "PIPE": {
            "TSDF": {
                "engine": "multi",
                "input_mode": "on_demand",
                "input_prefetch": True,
                "use_gt_pose": True,
                "saved_engine": "tsdf_engine/",
                "multi": {
                    "force_single_submap": False,
                    "loop_closure": False,
                    "global_adjustment": False,
                    "overlap_frames": 3,
                    "trial_frames": 6,
                    "sampled_original_blocks": 300,
                    "max_frames": 25,
                    "hash_profile": "compact_64k",
                    "paging": True,
                    "resident_gpu_maps": 2,
                    "max_render_maps": 4,
                },
            },
            "gaussian_submaps": {
                "enable": True,
                "debug": True,
                "ownership": "tsdf",
                "coordinate_frame": "local",
                "max_new_gaussians_per_frame": 0,
                "gaussian_insert_min_age_frames": 0,
                "gaussian_insert_min_sdf_blocks": 0,
                "handoff_optimize": True,
                "candidate_warming": True,
                "boundary_observation_handoff": True,
                "candidate_warm_interval": 1,
                "handoff_frames": 8,
                "handoff_opt_iters": 1,
                "handoff_max_opt_passes": 2,
                "render_policy": "sdf_guided_adaptive_fill",
                "render_owner_alpha_fill_thres": 0.001,
                "render_fill_blend_weight": 0.20,
                "render_fill_require_sdf_hole": True,
                "render_adaptive_kernel": 5,
                "render_adaptive_alpha_scale": 2.0,
                "render_adaptive_min_weight": 0.02,
                "render_adaptive_boundary_feather": 0.25,
                "render_adaptive_depth_thres": 0.05,
                "render_single_submap_passthrough": True,
                "render_single_submap_passthrough_requires_single_logical_map": False,
                "render_use_local_sdf_reference": True,
                "page_frozen_gaussians": True,
                "resident_gaussian_submaps": 2,
                "save_global_merged_model": False,
            },
        },
        "MODEL": {"sh_degree": 1},
    }
    rows = []
    for name, patch in [
        ("safe_opt1", {"PIPE": {"new_gs_sample_ratio": 0.03, "local_opt_iters": 1}}),
        ("opt2", {"PIPE": {"new_gs_sample_ratio": 0.03, "local_opt_iters": 2}}),
        ("opt3", {"PIPE": {"new_gs_sample_ratio": 0.03, "local_opt_iters": 3}}),
        ("sample005_opt2", {"PIPE": {"new_gs_sample_ratio": 0.05, "local_opt_iters": 2}}),
        ("sh3_opt2", {"PIPE": {"new_gs_sample_ratio": 0.03, "local_opt_iters": 2}, "MODEL": {"sh_degree": 3}}),
    ]:
        cfg = deepcopy(common)
        deep_update(cfg, patch)
        rows.append((name, cfg))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--full-res", action="store_true")
    parser.add_argument("--binary", type=Path, default=PROJECT_ROOT / "build_compact" / "slam_trainer")
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--heldout-interval", type=int, default=-1, help="set READER.test_split_interval and render PIPE.eval_split=val")
    args = parser.parse_args()

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Refusing to run: {free_gb:.1f} GB free, need {args.min_free_gb:.1f} GB")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    half = not args.full_res
    gt_dir = GT_HALF if half else GT_FULL
    quality_eval_split = "val" if args.heldout_interval > 0 else "train"
    rows = []
    for variant_name, overrides in variants():
        split_suffix = f"_heldout{args.heldout_interval}" if args.heldout_interval > 0 else ""
        name = f"train_sweep_{variant_name}_{args.frames}f_{'half' if half else 'full'}{split_suffix}"
        cfg = load_base(args.frames, half, name, args.heldout_interval)
        cfg["workspace_dir"] = f"output/hybrid_training_sweep/{name}"
        deep_update(cfg, overrides)
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
        out_dir = PROJECT_ROOT / cfg["workspace_dir"]
        if out_dir.exists():
            shutil.rmtree(out_dir)

        proc = run_cmd([str(args.binary), str(cfg_path)], PROJECT_ROOT)
        log_path = PROJECT_ROOT / f"output_{name}.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        row = {"name": name, "status": "OK" if proc.returncode == 0 else "CRASH", "config": str(cfg_path), "log": str(log_path)}
        if proc.returncode == 0:
            quality = run_quality_eval(args.binary, cfg, cfg_path, out_dir, gt_dir, name, quality_eval_split)
            row.update(quality)
            time_log = out_dir / "time_log.txt"
            if time_log.exists():
                for line in time_log.read_text(errors="ignore").splitlines():
                    if "[PIPELINE AVG TIME]" in line:
                        for part in line.split(","):
                            if ":" not in part:
                                continue
                            key, value = part.split(":", 1)
                            key = key.replace("[PIPELINE AVG TIME]", "").strip().replace(" ", "_")
                            try:
                                row[key] = float(value.strip())
                            except ValueError:
                                pass
            if "hybrid_PSNR" in row and "sdf_PSNR" in row:
                row["delta_PSNR"] = row["hybrid_PSNR"] - row["sdf_PSNR"]
            if "hybrid_SSIM" in row and "sdf_SSIM" in row:
                row["delta_SSIM"] = row["hybrid_SSIM"] - row["sdf_SSIM"]
            if "hybrid_LPIPS" in row and "sdf_LPIPS" in row:
                row["delta_LPIPS"] = row["hybrid_LPIPS"] - row["sdf_LPIPS"]
            row["ATE_actual_RMSE_cm"] = evaluate_ate(out_dir, gt_dir)
        rows.append(row)
        print(json.dumps(row, indent=2))

    split_suffix = f"_heldout{args.heldout_interval}" if args.heldout_interval > 0 else ""
    summary_path = OUTPUT_ROOT / f"summary_{args.frames}f_{'half' if half else 'full'}{split_suffix}.csv"
    with summary_path.open("w", newline="") as f:
        fields = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
