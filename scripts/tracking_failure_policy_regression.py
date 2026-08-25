#!/usr/bin/env python3
"""Compile and exhaustively test the side-effect-free tracking failure policy."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CPP_SOURCE = r"""
#include "ITMLib/Utils/ITMTrackingFailurePolicy.h"

#include <iostream>
#include <stdexcept>

using namespace ITMLib;

static void expect(ITMTrackingState::TrackingResult raw,
                   ITMLibSettings::FailureMode mode,
                   ITMTrackingState::TrackingResult policy,
                   bool rollback,
                   bool suppress)
{
    const ITMTrackingFailureDecision decision = DecideTrackingFailure(raw, mode);
    if (decision.policyResult != policy ||
        decision.rollbackPose != rollback ||
        decision.suppressFusion != suppress)
        throw std::runtime_error("tracking failure decision mismatch");
}

int main()
{
    const auto good = ITMTrackingState::TRACKING_GOOD;
    const auto poor = ITMTrackingState::TRACKING_POOR;
    const auto failed = ITMTrackingState::TRACKING_FAILED;

    expect(good, ITMLibSettings::FAILUREMODE_IGNORE, good, false, false);
    expect(poor, ITMLibSettings::FAILUREMODE_IGNORE, good, false, false);
    expect(failed, ITMLibSettings::FAILUREMODE_IGNORE, good, false, false);

    expect(good, ITMLibSettings::FAILUREMODE_STOP_INTEGRATION, good, false, false);
    expect(poor, ITMLibSettings::FAILUREMODE_STOP_INTEGRATION, poor, false, true);
    expect(failed, ITMLibSettings::FAILUREMODE_STOP_INTEGRATION, poor, true, true);

    expect(good, ITMLibSettings::FAILUREMODE_CONTAIN_FAILED, good, false, false);
    expect(poor, ITMLibSettings::FAILUREMODE_CONTAIN_FAILED, good, false, false);
    expect(failed, ITMLibSettings::FAILUREMODE_CONTAIN_FAILED, poor, true, true);

    expect(good, ITMLibSettings::FAILUREMODE_RELOCALISE, good, false, false);
    expect(poor, ITMLibSettings::FAILUREMODE_RELOCALISE, poor, false, true);
    expect(failed, ITMLibSettings::FAILUREMODE_RELOCALISE, failed, false, true);

    std::cout << "tracking failure policy regression passed (12/12 cases)\n";
    return 0;
}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="gps_slam_tracking_policy_") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "tracking_policy_test.cpp"
        binary = tmp_path / "tracking_policy_test"
        source.write_text(CPP_SOURCE)
        subprocess.run(
            [
                args.cxx,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-class-memaccess",
                "-I",
                str(PROJECT_ROOT / "InfiniTAM"),
                "-I",
                str(args.cuda_root / "include"),
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
