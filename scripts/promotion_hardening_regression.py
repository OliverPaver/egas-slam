#!/usr/bin/env python3
"""Fast regression for deployment promotion gate calculations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evaluate_deployment_promotion import evaluate_replica, evaluate_telemetry, evaluate_tum


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tum_row(sequence: str, ate: float, rpe: float, fps: float = 31.0) -> dict[str, str]:
    return {
        "sequence_input_dir": f"/data/{sequence}",
        "online_end_ate_actual_rmse_cm": str(ate),
        "online_end_rpe_translation_rmse_cm": str(rpe),
        "realtime_fixed_processing_fps": str(fps),
        "realtime_fixed_final_backlog_ms": "0",
        "realtime_fixed_maximum_backlog_ms": "100",
        "realtime_fixed_maximum_recovery_ms": "500",
        "realtime_all_frames_processed": "1",
        "realtime_fixed_frame_continuity": "1",
        "online_end_complete": "1",
        "policy_tracking_failed": "0",
    }


def replica_row(scene: str, psnr: float, ssim: float, lpips: float, gs: int) -> dict[str, str]:
    return {
        "scene_name": scene,
        "status": "OK",
        "hybrid_PSNR": str(psnr),
        "hybrid_SSIM": str(ssim),
        "hybrid_LPIPS": str(lpips),
        "runtime_final_gaussian_count": str(gs),
    }


def main() -> None:
    reference_tum = [tum_row("a", 2.8, 1.0), tum_row("b", 2.8, 1.0)]
    candidate_tum = [tum_row("a", 2.90, 1.04), tum_row("b", 2.90, 1.04)]
    require(evaluate_tum(reference_tum, candidate_tum)["pass"], "valid TUM candidate")
    overloaded = [tum_row("a", 2.90, 1.04, 29.9), tum_row("b", 2.90, 1.04)]
    require(not evaluate_tum(reference_tum, overloaded)["pass"], "sub-30 FPS candidate")

    reference_replica = [replica_row("office0", 30.0, 0.90, 0.10, 1000)]
    candidate_replica = [replica_row("office0", 29.91, 0.8991, 0.102, 1040)]
    require(evaluate_replica(reference_replica, candidate_replica)["pass"], "valid Replica candidate")
    candidate_replica[0]["hybrid_PSNR"] = "29.89"
    require(not evaluate_replica(reference_replica, candidate_replica)["pass"], "PSNR regression")

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "telemetry.json"
        payload = {
            "nvpmodel_maxn": True,
            "clocks_locked": True,
            "fan_max": True,
            "throttle_counters_unchanged": True,
            "thermal_throttle_clean": True,
            "run_valid_for_headline": True,
        }
        path.write_text(json.dumps(payload))
        require(evaluate_telemetry(path)["pass"], "clean telemetry")
        payload["throttle_counters_unchanged"] = False
        path.write_text(json.dumps(payload))
        require(not evaluate_telemetry(path)["pass"], "over-current telemetry")

    print("promotion hardening regression passed (3/3)")


if __name__ == "__main__":
    main()
