#!/usr/bin/env python3
"""Evaluate the locked Thor deployment promotion gates from benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ATE_REFERENCE_CM = 2.8446
ATE_TOLERANCE_CM = 0.10
RPE_MAX_RATIO = 1.05
PSNR_MAX_DROP_DB = 0.10
SSIM_MAX_DROP = 0.001
LPIPS_MAX_INCREASE = 0.003
GAUSSIAN_MAX_FRACTIONAL_CHANGE = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"summary has no rows: {path}")
    return rows


def parse_named(values: list[str], option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"invalid or duplicate candidate name {name!r}")
        parsed[name] = Path(raw_path)
    return parsed


def finite(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        raw = row.get(field, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def truthy(row: dict[str, str], field: str) -> bool:
    return str(row.get(field, "")).strip().lower() in {"1", "true", "yes", "pass", "ok"}


def sequence_name(row: dict[str, str]) -> str:
    value = row.get("sequence_input_dir", "")
    if value:
        return Path(value).name
    return row.get("run", row.get("name", "unknown"))


def scene_name(row: dict[str, str]) -> str:
    return row.get("scene_name", "") or Path(row.get("input_dir", "unknown")).name


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty value list")
    return sum(values) / len(values)


def evaluate_tum(reference: list[dict[str, str]], candidate: list[dict[str, str]]) -> dict[str, object]:
    required_sequences = sorted({sequence_name(row) for row in reference})
    candidate_sequences = sorted({sequence_name(row) for row in candidate})
    checks: list[dict[str, object]] = []

    for row in candidate:
        sequence = sequence_name(row)
        fields = {
            "processing_fps": finite(row, ("realtime_fixed_processing_fps",)),
            "final_backlog_ms": finite(row, ("realtime_fixed_final_backlog_ms",)),
            "maximum_backlog_ms": finite(row, ("realtime_fixed_maximum_backlog_ms",)),
            "maximum_recovery_ms": finite(row, ("realtime_fixed_maximum_recovery_ms",)),
            "online_ate_cm": finite(row, ("online_end_ate_actual_rmse_cm",)),
            "online_rpe_cm": finite(row, ("online_end_rpe_translation_rmse_cm",)),
            "peak_rss_kb": finite(row, ("runtime_process_peak_rss_kb",)),
        }
        passed = bool(
            all(fields[key] is not None for key in fields if key != "peak_rss_kb")
            and fields["processing_fps"] >= 30.0
            and fields["final_backlog_ms"] <= 1000.0 / 30.0
            and fields["maximum_backlog_ms"] <= 500.0
            and fields["maximum_recovery_ms"] <= 1000.0
            and truthy(row, "realtime_all_frames_processed")
            and truthy(row, "online_end_complete")
            and truthy(row, "realtime_fixed_frame_continuity")
            and int(float(row.get("policy_tracking_failed", "0") or 0)) == 0
        )
        checks.append({"sequence": sequence, **fields, "pass": passed})

    online_ates = [float(check["online_ate_cm"]) for check in checks if check["online_ate_cm"] is not None]
    online_rpes = [float(check["online_rpe_cm"]) for check in checks if check["online_rpe_cm"] is not None]
    reference_rpes = [
        value
        for row in reference
        if (
            value := finite(
                row,
                (
                    "corrected_rpe_translation_rmse_cm",
                    "online_end_rpe_translation_rmse_cm",
                    "rpe_translation_rmse_cm",
                ),
            )
        )
        is not None
    ]
    mean_ate = average(online_ates) if online_ates else None
    mean_rpe = average(online_rpes) if online_rpes else None
    reference_mean_rpe = average(reference_rpes) if reference_rpes else None
    ate_pass = mean_ate is not None and mean_ate <= ATE_REFERENCE_CM + ATE_TOLERANCE_CM
    rpe_pass = (
        mean_rpe is not None
        and reference_mean_rpe is not None
        and mean_rpe <= reference_mean_rpe * RPE_MAX_RATIO
    )
    coverage_pass = candidate_sequences == required_sequences
    return {
        "required_sequences": required_sequences,
        "candidate_sequences": candidate_sequences,
        "sequence_coverage_pass": coverage_pass,
        "runs": checks,
        "online_end_mean_ate_cm": mean_ate,
        "ate_limit_cm": ATE_REFERENCE_CM + ATE_TOLERANCE_CM,
        "ate_pass": ate_pass,
        "online_end_mean_rpe_cm": mean_rpe,
        "reference_mean_rpe_cm": reference_mean_rpe,
        "rpe_ratio_limit": RPE_MAX_RATIO,
        "rpe_pass": rpe_pass,
        "pass": coverage_pass and bool(checks) and all(bool(check["pass"]) for check in checks) and ate_pass and rpe_pass,
    }


def metric(row: dict[str, str], name: str) -> float | None:
    return finite(
        row,
        (
            f"finalmap_hybrid_{name}",
            f"hybrid_{name}",
            f"render_{name.lower()}",
        ),
    )


def gaussian_count(row: dict[str, str]) -> float | None:
    return finite(
        row,
        (
            "runtime_final_gaussian_count",
            "gs_num",
            "final_gaussians_total",
            "final_gaussians_sum",
        ),
    )


def evaluate_replica(reference: list[dict[str, str]], candidate: list[dict[str, str]]) -> dict[str, object]:
    reference_by_scene = {scene_name(row): row for row in reference}
    candidate_by_scene = {scene_name(row): row for row in candidate}
    checks: list[dict[str, object]] = []
    for scene in sorted(reference_by_scene):
        reference_row = reference_by_scene[scene]
        candidate_row = candidate_by_scene.get(scene)
        if candidate_row is None:
            checks.append({"scene": scene, "pass": False, "error": "missing candidate row"})
            continue
        ref_psnr, cand_psnr = metric(reference_row, "PSNR"), metric(candidate_row, "PSNR")
        ref_ssim, cand_ssim = metric(reference_row, "SSIM"), metric(candidate_row, "SSIM")
        ref_lpips, cand_lpips = metric(reference_row, "LPIPS"), metric(candidate_row, "LPIPS")
        ref_gs, cand_gs = gaussian_count(reference_row), gaussian_count(candidate_row)
        gaussian_fraction = (
            abs(cand_gs - ref_gs) / ref_gs
            if ref_gs is not None and cand_gs is not None and ref_gs > 0.0
            else None
        )
        passed = bool(
            None not in (ref_psnr, cand_psnr, ref_ssim, cand_ssim, ref_lpips, cand_lpips, gaussian_fraction)
            and cand_psnr >= ref_psnr - PSNR_MAX_DROP_DB
            and cand_ssim >= ref_ssim - SSIM_MAX_DROP
            and cand_lpips <= ref_lpips + LPIPS_MAX_INCREASE
            and gaussian_fraction <= GAUSSIAN_MAX_FRACTIONAL_CHANGE
            and str(candidate_row.get("status", "OK")).upper() == "OK"
        )
        checks.append(
            {
                "scene": scene,
                "reference_psnr": ref_psnr,
                "candidate_psnr": cand_psnr,
                "reference_ssim": ref_ssim,
                "candidate_ssim": cand_ssim,
                "reference_lpips": ref_lpips,
                "candidate_lpips": cand_lpips,
                "reference_gaussians": ref_gs,
                "candidate_gaussians": cand_gs,
                "gaussian_fractional_change": gaussian_fraction,
                "pass": passed,
            }
        )
    coverage_pass = sorted(candidate_by_scene) == sorted(reference_by_scene)
    return {
        "required_scenes": sorted(reference_by_scene),
        "candidate_scenes": sorted(candidate_by_scene),
        "scene_coverage_pass": coverage_pass,
        "runs": checks,
        "pass": coverage_pass and bool(checks) and all(bool(check["pass"]) for check in checks),
    }


def evaluate_telemetry(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    required = (
        "nvpmodel_maxn",
        "clocks_locked",
        "fan_max",
        "throttle_counters_unchanged",
        "thermal_throttle_clean",
        "run_valid_for_headline",
    )
    return {
        "path": str(path.resolve()),
        **{key: bool(payload.get(key, False)) for key in required},
        "pass": all(bool(payload.get(key, False)) for key in required),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-reference", type=Path, required=True)
    parser.add_argument("--tum-candidate", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--replica-reference", type=Path, required=True)
    parser.add_argument("--replica-candidate", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--telemetry", action="append", default=[], metavar="NAME=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tum_candidates = parse_named(args.tum_candidate, "--tum-candidate")
    replica_candidates = parse_named(args.replica_candidate, "--replica-candidate")
    telemetry = parse_named(args.telemetry, "--telemetry")
    names = sorted(set(tum_candidates) | set(replica_candidates) | set(telemetry))
    if not names or any(name not in tum_candidates or name not in replica_candidates or name not in telemetry for name in names):
        parser.error("every candidate needs TUM, Replica, and telemetry inputs with the same NAME")

    tum_reference = read_csv(args.tum_reference)
    replica_reference = read_csv(args.replica_reference)
    candidates: dict[str, object] = {}
    for name in names:
        tum = evaluate_tum(tum_reference, read_csv(tum_candidates[name]))
        replica = evaluate_replica(replica_reference, read_csv(replica_candidates[name]))
        hardware = evaluate_telemetry(telemetry[name])
        candidates[name] = {
            "tum": tum,
            "replica": replica,
            "hardware": hardware,
            "promoted": bool(tum["pass"] and replica["pass"] and hardware["pass"]),
        }
    report = {
        "schema_version": 1,
        "thresholds": {
            "online_end_mean_ate_limit_cm": ATE_REFERENCE_CM + ATE_TOLERANCE_CM,
            "rpe_max_ratio": RPE_MAX_RATIO,
            "psnr_max_drop_db": PSNR_MAX_DROP_DB,
            "ssim_max_drop": SSIM_MAX_DROP,
            "lpips_max_increase": LPIPS_MAX_INCREASE,
            "gaussian_max_fractional_change": GAUSSIAN_MAX_FRACTIONAL_CHANGE,
        },
        "candidates": candidates,
    }
    atomic_json(args.output, report)
    for name, result in candidates.items():
        print(f"{name}: {'PROMOTE' if result['promoted'] else 'REJECT'}")
    return 0 if any(bool(result["promoted"]) for result in candidates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
