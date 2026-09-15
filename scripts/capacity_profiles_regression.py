#!/usr/bin/env python3
"""CPU-only compile/run checks for exact capacity dimensions and overflow rejection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

PROFILES = {
    "legacy_large": ([], 262144, 1048576, 131072),
    "compact_64k": (["GPS_SLAM_COMPACT_HASH"], 65536, 262144, 32768),
    "medium_128k": (["GPS_SLAM_MEDIUM_HASH"], 131072, 524288, 65536),
    "legacy_pool_512k": (["GPS_SLAM_CAPACITY_CONTROL=1"], 524288, 1048576, 131072),
    "legacy_pool_1m": (["GPS_SLAM_CAPACITY_CONTROL=2"], 1048576, 4194304, 524288),
    "legacy_256k_hash4m": (["GPS_SLAM_CAPACITY_CONTROL=3"], 262144, 4194304, 524288),
    "legacy_pool_2m": (["GPS_SLAM_CAPACITY_CONTROL=4"], 2097152, 8388608, 1048576),
    "legacy_256k_hash8m": (["GPS_SLAM_CAPACITY_CONTROL=5"], 262144, 8388608, 1048576),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    header = root / "InfiniTAM/ITMLib/Objects/Scene/ITMHashCapacity.h"
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="capacity_cpu_", dir=args.output.parent) as temporary:
        temporary = Path(temporary)
        source = temporary / "test.cpp"
        source.write_text('#include "capacity.h"\n#include <cstdio>\nint main() { '
                          'printf("%s %d %d %d %d\\n", GPS_SLAM_COMPILED_HASH_PROFILE, '
                          'SDF_LOCAL_BLOCK_NUM, SDF_BUCKET_NUM, SDF_EXCESS_LIST_SIZE, SDF_HASH_MASK); }\n')
        candidate = temporary / "capacity.h"
        candidate.write_bytes(header.read_bytes())
        for name, (defines, blocks, buckets, excess) in PROFILES.items():
            command = ["g++", "-std=c++17", "-Wall", "-Werror", *["-D" + d for d in defines],
                       str(source), "-o", str(temporary / "test")]
            subprocess.run(command, check=True, capture_output=True)
            found = subprocess.check_output([str(temporary / "test")]).decode().split()
            expected = [name, *map(str, (blocks, buckets, excess, buckets - 1))]
            results.append({"case": name, "pass": found == expected, "actual": found,
                            "voxel_pool_bytes": blocks * 512 * 8})
        for name, defines, replace in [
            ("conflicting_profiles", ["GPS_SLAM_COMPACT_HASH", "GPS_SLAM_MEDIUM_HASH"], None),
            ("unknown_capacity_tier", ["GPS_SLAM_CAPACITY_CONTROL=99"], None),
            ("four_million_blocks_overflow", ["GPS_SLAM_CAPACITY_CONTROL=4"],
             ("#define SDF_LOCAL_BLOCK_NUM 0x200000", "#define SDF_LOCAL_BLOCK_NUM 0x400000")),
            ("non_power_of_two_hash", ["GPS_SLAM_CAPACITY_CONTROL=4"],
             ("#define SDF_BUCKET_NUM 0x800000", "#define SDF_BUCKET_NUM 0x800001")),
        ]:
            text = header.read_text()
            candidate.write_text(text.replace(*replace) if replace else text)
            check = subprocess.run(["g++", "-std=c++17", *["-D" + d for d in defines],
                str(source), "-o", str(temporary / "test")], capture_output=True, text=True)
            results.append({"case": name, "pass": check.returncode != 0, "compiler_error": check.stderr})
    report = {"status": "pass" if all(row["pass"] for row in results) else "fail", "gpu_used": False,
              "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(), "cases": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(results), "gpu_used": False}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
