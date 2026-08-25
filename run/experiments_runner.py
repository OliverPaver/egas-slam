#!/usr/bin/env python3
"""
Batch experiment runner for GPS-SLAM speed/ATE trade-off exploration.
Generates configs, runs slam_trainer, extracts timing + ATE.
"""
import os
import sys
import yaml
import subprocess
import json
import shutil
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
OUTPUT_BASE = PROJECT_ROOT / "output" / "experiments"
SLAM_TRAINER = PROJECT_ROOT / "build" / "slam_trainer"
ATE_SCRIPT = PROJECT_ROOT / "scripts" / "ate_general.py"
GT_POSE_DIR = PROJECT_ROOT / "data" / "replica" / "office0" / "camera"

# Base config (the working half-res tracking config)
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "release" / "replica" / "office0_tracking_halfres_fixed.yaml"
MIN_FREE_GB = float(os.environ.get("GPS_SLAM_MIN_FREE_GB", "20"))


@dataclass
class Experiment:
    name: str
    overrides: dict  # nested dict matching YAML structure


EXPERIMENTS = [
    Experiment("exp_01_baseline", {}),
    Experiment("exp_02_iters_10", {"PIPE": {"local_opt_iters": 10}}),
    Experiment("exp_03_iters_5", {"PIPE": {"local_opt_iters": 5}}),
    Experiment("exp_04_interval_10", {"PIPE": {"local_opt_interval": 10}}),
    Experiment("exp_05_sh_1", {"MODEL": {"sh_degree": 1}}),
    Experiment("exp_06_sh_0", {"MODEL": {"sh_degree": 0}}),
    Experiment("exp_07_voxel_016", {"PIPE": {"TSDF": {"voxel_size": 0.016, "trunc_dist": 0.064}}}),
    Experiment("exp_08_kf_max_4", {"PIPE": {"keyframe_select_max": 4}}),
    Experiment("exp_09_local_win_1", {"PIPE": {"localframe_cam_window_length": 1}}),
    Experiment("exp_10_combo_fast", {
        "PIPE": {"local_opt_iters": 10, "local_opt_interval": 10, "keyframe_select_max": 4},
        "MODEL": {"sh_degree": 1}
    }),
    Experiment("exp_11_combo_aggressive", {
        "PIPE": {"local_opt_iters": 5, "local_opt_interval": 10, "keyframe_select_max": 3,
                 "TSDF": {"voxel_size": 0.016, "trunc_dist": 0.064}},
        "MODEL": {"sh_degree": 0}
    }),
    Experiment("exp_12_combo_balanced", {
        "PIPE": {"local_opt_iters": 15, "local_opt_interval": 8, "keyframe_select_max": 5,
                 "TSDF": {"voxel_size": 0.012, "trunc_dist": 0.048}},
        "MODEL": {"sh_degree": 1}
    }),
]


def deep_update(base, update):
    """Recursively update nested dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def generate_config(exp: Experiment) -> Path:
    """Generate a config YAML for an experiment."""
    with open(BASE_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Update workspace dir
    cfg["workspace_dir"] = f"output/experiments/{exp.name}"

    # Apply overrides
    deep_update(cfg, exp.overrides)

    out_path = CONFIGS_DIR / f"{exp.name}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return out_path


def run_experiment(exp: Experiment) -> dict:
    """Run slam_trainer and collect metrics."""
    cfg_path = CONFIGS_DIR / f"{exp.name}.yaml"
    out_dir = OUTPUT_BASE / exp.name
    pose_dir = out_dir / "val" / "pose"
    time_log = out_dir / "time_log.txt"
    ate_out = out_dir / "val" / "pose_eval.txt"

    print(f"\n{'='*60}")
    print(f"Running: {exp.name}")
    print(f"Config:  {cfg_path}")
    print(f"{'='*60}")

    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        print(f"[ERROR] Refusing to start {exp.name}: only {free_gb:.1f} GB free, need {MIN_FREE_GB:.1f} GB")
        return {"name": exp.name, "status": "LOW_DISK", "fps": 0, "ate_cm": None}

    largest = sorted(
        ((p, sum(f.stat().st_size for f in p.rglob('*') if f.is_file())) for p in OUTPUT_BASE.glob("exp_*") if p.is_dir()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    if largest:
        print("[DISK] largest output/experiments dirs:")
        for path, size in largest:
            print(f"  {path.name}: {size / (1024 ** 3):.2f} GB")

    # Clean previous run
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Run slam_trainer
    result = subprocess.run(
        [str(SLAM_TRAINER), str(cfg_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[ERROR] {exp.name} crashed with code {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")
        return {"name": exp.name, "status": "CRASH", "fps": 0, "ate_cm": None}

    # Parse timing
    fps = 0.0
    gs_num = 0
    fusion_ms = 0.0
    local_opt_ms = 0.0
    if time_log.exists():
        with open(time_log) as f:
            for line in f:
                if "[PIPELINE AVG TIME]" in line:
                    # Format: GS num: N, per frame fusion time: X, ..., localOptimize time: Y, FPS: Z
                    parts = line.split(",")
                    for p in parts:
                        if "FPS:" in p:
                            fps = float(p.split(":")[-1].strip())
                        if "GS num:" in p:
                            gs_num = int(p.split(":")[-1].strip())
                        if "per frame fusion time:" in p:
                            fusion_ms = float(p.split(":")[-1].strip())
                        if "localOptimize time:" in p:
                            local_opt_ms = float(p.split(":")[-1].strip())

    # Compute ATE
    ate_cm = None
    valid_frames = 0
    total_frames = 0
    if pose_dir.exists():
        # Count valid poses
        for f in sorted(pose_dir.glob("frame*.txt")):
            total_frames += 1
            try:
                import numpy as np
                p = np.loadtxt(f)
                if not (np.isnan(p).any() or np.isinf(p).any()):
                    valid_frames += 1
            except Exception:
                pass

        if valid_frames > 10 and valid_frames == total_frames:
            ate_result = subprocess.run(
                ["python", str(ATE_SCRIPT),
                 "--gt_path", str(GT_POSE_DIR),
                 "--est_path", str(pose_dir)],
                capture_output=True,
                text=True,
            )
            if ate_result.returncode == 0 and ate_out.exists():
                with open(ate_out) as f:
                    for line in f:
                        if line.startswith("ATE actual RMSE:"):
                            ate_cm = float(line.split(":", 1)[1].strip())
                            break
            else:
                # Try computing manually
                try:
                    import numpy as np
                    ests, gts = [], []
                    for f in sorted(pose_dir.glob("frame*.txt")):
                        idx = int(f.stem.replace("frame", ""))
                        est = np.loadtxt(f)
                        gt_file = GT_POSE_DIR / f"pose{idx:06d}.txt"
                        if gt_file.exists():
                            gt = np.loadtxt(gt_file)
                            if not (np.isnan(est).any() or np.isinf(est).any()):
                                ests.append(est)
                                gts.append(gt)
                    if len(ests) > 10:
                        est_traj = np.stack([e[:3, 3] for e in ests]).T
                        gt_traj = np.stack([g[:3, 3] for g in gts]).T
                        model_zc = est_traj - est_traj.mean(1).reshape((3, -1))
                        data_zc = gt_traj - gt_traj.mean(1).reshape((3, -1))
                        W = np.zeros((3, 3))
                        for c in range(model_zc.shape[1]):
                            W += np.outer(model_zc[:, c], data_zc[:, c])
                        U, d, Vh = np.linalg.svd(W.transpose())
                        S = np.identity(3)
                        if np.linalg.det(U) * np.linalg.det(Vh) < 0:
                            S[2, 2] = -1
                        rot = U @ S @ Vh
                        trans = gt_traj.mean(1).reshape((3, -1)) - rot @ est_traj.mean(1).reshape((3, -1))
                        aligned = rot @ est_traj + trans
                        err = np.sqrt(np.sum((aligned - gt_traj) ** 2, axis=0))
                        ate_cm = float(err.mean() * 100)
                except Exception as e:
                    print(f"[WARN] Manual ATE failed: {e}")

    status = "OK" if valid_frames == total_frames and total_frames > 0 else "TRACK_LOST"
    if ate_cm is None and status == "OK":
        status = "ATE_FAIL"

    return {
        "name": exp.name,
        "status": status,
        "valid_frames": valid_frames,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "fusion_ms": round(fusion_ms, 2),
        "local_opt_ms": round(local_opt_ms, 2),
        "gs_num": gs_num,
        "ate_cm": round(ate_cm, 2) if ate_cm is not None else None,
    }


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        generate_config(exp)
        res = run_experiment(exp)
        results.append(res)
        print(json.dumps(res, indent=2))

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<25} {'Status':<12} {'FPS':>8} {'ATE(cm)':>10} {'GS Num':>10} {'Valid':>8}")
    print("-" * 80)
    for r in results:
        ate_str = f"{r['ate_cm']:.2f}" if r["ate_cm"] is not None else "N/A"
        print(f"{r['name']:<25} {r['status']:<12} {r['fps']:>8.2f} {ate_str:>10} {r['gs_num']:>10} {r['valid_frames']:>8}/{r['total_frames']}")

    # Save CSV
    csv_path = OUTPUT_BASE / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("name,status,fps,ate_cm,gs_num,valid_frames,total_frames\n")
        for r in results:
            ate_str = f"{r['ate_cm']:.2f}" if r["ate_cm"] is not None else ""
            f.write(f"{r['name']},{r['status']},{r['fps']},{ate_str},{r['gs_num']},{r['valid_frames']},{r['total_frames']}\n")
    print(f"\nSummary saved to: {csv_path}")


if __name__ == "__main__":
    main()
