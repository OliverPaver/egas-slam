# Reproducibility Guide

## Scope

The exact evaluated source is tagged `thor-paper-20260819-r2-source`. The paper release is algorithm-equivalent but uses clone-independent paths and embeds a different Git revision in newly built executables. Historical metrics belong to the binary/config/dataset hashes in their manifests; do not silently combine outputs from another build.

## Host preparation

Use Jetson AGX Thor with JetPack 7.1, Docker, the NVIDIA container runtime, an ext4/NVMe build filesystem, and sufficient free space. Configure a stable power mode, lock clocks, and set maximum fan before headline timing. Any run with changed over-current or thermal counters is invalid.

Set external roots before invoking the wrapper:

```bash
export GPS_SLAM_DATA_ROOT=/absolute/path/to/datasets
export GPS_SLAM_WEIGHT_ROOT=/absolute/path/to/torch_hub
export GPS_SLAM_RUN_ROOT=/absolute/path/to/runs
```

Expected data layout:

```text
$GPS_SLAM_DATA_ROOT/
├── replica/{office0..office4,room0..room2}/
# or replica_prepared/{office0..office4,room0..room2}/
└── tum_rgbd/processed/rgbd_dataset_freiburg*/

$GPS_SLAM_WEIGHT_ROOT/checkpoints/
├── vgg.pth
└── vgg16-397923af.pth
```

Datasets and weights are mounted read-only at `/workspace/data` and the Torch cache. The run root is writable at `/workspace/runs`. Set `GPS_SLAM_LEGACY_TRANSFER_ROOT` only to reproduce old commands that explicitly name `/workspace/transfer`.

## Build and validate

```bash
bash scripts/thor_container.sh image
bash scripts/thor_container.sh check
bash scripts/thor_container.sh build
bash scripts/thor_container.sh baseline-build
bash scripts/thor_container.sh regressions
bash scripts/thor_container.sh source-check
```

For a new machine/build, create an exact local comparison envelope after both builds:

```bash
export GPS_SLAM_BENCHMARK_FREEZE=/workspace/runs/release-freeze.json
bash scripts/thor_container.sh run python3 scripts/benchmark_freeze.py create \
  --freeze-id v1.0.0-paper-local \
  --manifest /workspace/runs/release-freeze.json
bash scripts/thor_container.sh run python3 scripts/benchmark_freeze.py verify \
  --manifest /workspace/runs/release-freeze.json
```

Keep that manifest with every resulting run. The published compact artifact carries the authoritative R2 manifests and does not include datasets or executables.

## Headline protocols

Replica uses eight scenes, all 2,000 frames at 1200×680, ground-truth poses to isolate reconstruction, and complete image-metric coverage. TUM uses estimated poses at 640×480 and reports ATE/RPE only with complete pairing and rigid SE(3) alignment without scale. See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) before running.

Example proposed Replica run:

```bash
bash scripts/thor_container.sh telemetry-run python3 run/hybrid_validation.py \
  --benchmark-protocol gps_slam_replica_paper_v1 \
  --benchmark-role modified --benchmark-dataset-hash inventory \
  --binary build_thor/slam_trainer --labels multi_hybrid_warm_fill \
  --full-res --frames 2000 --quality --quality-final-map \
  --gaussian-profile quality --hybrid-submap-profile normal \
  --output-root /workspace/runs/replica_modified
```

Example proposed TUM run:

```bash
bash scripts/thor_container.sh telemetry-run python3 run/tum_loop_validation.py \
  --benchmark-protocol gps_slam_tum_tracking_v1 \
  --benchmark-role modified --benchmark-dataset-hash inventory \
  --tum-root /workspace/data/tum_rgbd/processed \
  --trainer build_thor/slam_trainer --variants multi_sdf_orb_apply \
  --loop-profile recommended_longspan_robust \
  --tsdf-lifecycle-profile paper_parity \
  --sequences fr1_desk fr2_xyz fr3_office \
  --full-sequence --repeats 5 --run \
  --output-root /workspace/runs/tum_modified
```

Run the faithful baseline and integrated Gaussian arm with the paired commands in [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md). Never compare Replica GT-pose quality to tracking accuracy, cross-hardware FPS as an efficiency ranking, or an invalid/incomplete baseline trajectory as a headline ATE.

## Required records

Retain generated YAML, `benchmark_manifest.json`, exact trajectories, pose pairing, timing/tracking traces, submap and pose-graph records, metric coverage/status, stdout/stderr, process peak RSS, environment capture, and telemetry. Report input loading, frame-loop service, terminal work, read-only metric rendering, memory, and power separately.

Successful output has all expected frames, finite poses/metrics, no CUDA error or NaN, unchanged throttle counters, and a zero process exit. The R2 acceptance thresholds and known limitations are recorded in [PAPER_READINESS_LOCK_2026-08-25.md](PAPER_READINESS_LOCK_2026-08-25.md).
