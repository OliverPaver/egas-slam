# Locked Benchmark Protocol

The comparison code is frozen by `BENCHMARK_FREEZE.json`. Before generating or
running a locked benchmark, verify the exact source tree, both binaries, CMake
profiles, AArch64 platform, and container image:

```bash
bash scripts/thor_container.sh freeze-check
bash scripts/thor_container.sh regressions
```

Every new `benchmark_manifest.json` embeds the verified freeze ID and digest.
Any functional source addition, deletion, edit, rebuild, build-profile change,
or container replacement blocks locked runs until the change is reviewed,
validated, and issued as a new freeze revision. Documentation and result files
are outside the functional hash and may be updated without changing the code.

Use these commands for headline results. The lock is opt-in: ordinary smoke
and diagnostic runs are unchanged. A locked run rejects incompatible labels,
frame ranges, trackers, build profiles, or evaluation settings before launch
and writes `benchmark_manifest.json` with the config/binary hashes, dataset
inventory, container identity, Jetson model, power mode, clocks, CUDA, and
Torch versions. Headline execution also refuses to start unless MAXN and
locked CPU/GPU clock constraints are observable; config-only generation is
still allowed in another power state.

Only `gps_slam_replica_paper_v1`, `gps_slam_tum_tracking_v1`, and
`gps_slam_tum_integrated_v1` are publication-locked. The two `*_deployed_v1`
protocols are retained solely to reproduce the rejected deployment screen and
require `--allow-unpromoted-protocol`; their outputs are not paper-eligible.
Immediate Gaussian scheduling remains the comparison default.

## Build the Two Targets

```bash
bash scripts/thor_container.sh build
bash scripts/thor_container.sh baseline-build
```

`build_thor/` is the embedded `medium_128k` method build. `build_legacy/` uses
the original `legacy_large` TSDF capacity and is mandatory for the faithful
BasicEngine baseline. Both are native AArch64/SM 11.0 binaries; Docker is not
CPU or GPU emulation.

## Replica Rendering Comparison

The protocol fixes 2,000 frames at 1200x680, frame step 1, GT poses, 5 mm
voxels, the original 20-iteration/10-frame optimizer schedule, train-split
evaluation, SH degree 3, and the original L1-only loss. Post-sequence Gaussian
insertion, graduation optimization, and frozen refinement are disabled. The
final-map pass only reloads and renders the map already produced online; it
does not reintegrate frames or optimize Gaussians.

```bash
# Faithful GPS-SLAM baseline
bash scripts/thor_container.sh run python3 run/hybrid_validation.py \
  --benchmark-protocol gps_slam_replica_paper_v1 --benchmark-role baseline \
  --benchmark-dataset-hash inventory --binary build_legacy/slam_trainer \
  --labels basic_unified_gaussian --full-res --frames 2000 --quality \
  --gaussian-profile quality --output-root /workspace/runs/replica_baseline

# Proposed submap method
bash scripts/thor_container.sh run python3 run/hybrid_validation.py \
  --benchmark-protocol gps_slam_replica_paper_v1 --benchmark-role modified \
  --benchmark-dataset-hash inventory --binary build_thor/slam_trainer \
  --labels multi_hybrid_warm_fill --full-res --frames 2000 --quality \
  --quality-final-map --gaussian-profile quality \
  --hybrid-submap-profile normal --output-root /workspace/runs/replica_modified
```

Compare baseline `hybrid_PSNR/SSIM/LPIPS` with modified
`finalmap_hybrid_PSNR/SSIM/LPIPS` using the recorded train split.

## TUM Accuracy Comparison

TUM accuracy is a tracking/loop-closure experiment, so both arms intentionally
run `work_mode: recon` without Gaussians. They use the same legacy InfiniTAM
tracker and complete sequence. Five same-seed repeats expose lifecycle
nondeterminism. Run the baseline and method separately because they require
different compiled TSDF capacities.

```bash
# Substitute the desired paper sequence list in both commands.
bash scripts/thor_container.sh run python3 run/tum_loop_validation.py \
  --benchmark-protocol gps_slam_tum_tracking_v1 --benchmark-role baseline \
  --tum-root /workspace/transfer/datasets/tum_rgbd/processed \
  --trainer build_legacy/slam_trainer --variants basic_sdf \
  --sequences fr1_desk fr2_xyz fr3_office --full-sequence --repeats 5 --run \
  --output-root /workspace/runs/tum_baseline

bash scripts/thor_container.sh run python3 run/tum_loop_validation.py \
  --benchmark-protocol gps_slam_tum_tracking_v1 --benchmark-role modified \
  --tum-root /workspace/transfer/datasets/tum_rgbd/processed \
  --trainer build_thor/slam_trainer --variants multi_sdf_orb_apply \
  --loop-profile recommended_longspan_robust \
  --tsdf-lifecycle-profile paper_parity \
  --sequences fr1_desk fr2_xyz fr3_office --full-sequence --repeats 5 --run \
  --output-root /workspace/runs/tum_modified
```

Report raw and corrected ATE/RPE. Also report `steady_state_fps`, terminal scan
latency, and `terminal_inclusive_pipeline_fps`. The latter is the C++ wall
clock from immediately before the frame loop through end-of-sequence loop
closure; it excludes dataset loading, checkpoint saving, and metric rendering.
Do not describe tracking-only TUM FPS as full Gaussian-system FPS. Use
`gps_slam_tum_integrated_v1` with `multi_hybrid_orb_apply` for that separate
measurement.

Run `bash scripts/thor_container.sh regressions` before publishing results.
