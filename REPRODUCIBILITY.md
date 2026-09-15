# Build and run the final implementation

This source distribution targets Jetson AGX Thor, JetPack 7.1, native AArch64 and CUDA SM 110. It includes the proposed submap system, a separately configured capacity-augmented original-algorithm control, saved-map rendering and evaluation utilities. Benchmark campaigns, comparator implementations, thesis results and data are maintained separately.

## External directories

Install Docker with the NVIDIA container runtime and build on NVMe/ext4. The pinned NVIDIA PyTorch image supplies compatible Torch/CUDA/C++ libraries; do not substitute desktop LibTorch binaries.

```bash
git clone https://github.com/OliverPaver/gps-slam-submaps-thor.git
cd gps-slam-submaps-thor
export GPS_SLAM_DATA_ROOT=/absolute/path/to/datasets
export GPS_SLAM_WEIGHT_ROOT=/absolute/path/to/torch_hub
export GPS_SLAM_RUN_ROOT=/absolute/path/to/writable/runs
```

Use a dedicated results directory. A training workspace is managed by the trainer and may be cleared. Data and weights are mounted read-only. Container paths are `/workspace/project`, `/workspace/data`, and `/workspace/runs`. If your account cannot access Docker, the wrapper supports `GPS_SLAM_DOCKER_SUDO=1` for your existing sudo workflow.

Expected external layout:

```text
datasets/
  replica_prepared/{office0,office1,office2,office3,office4,room0,room1,room2}/
    camera/frame000000.jpg, pose000000.txt, intrinsics.txt, img_shape.txt, ...
    depth/depth000000.png, ...
  tum_rgbd/processed/
    rgbd_dataset_freiburg1_desk/
    rgbd_dataset_freiburg2_xyz/
    rgbd_dataset_freiburg3_long_office_household/
      camera/frame000000.png, pose000000.txt, intrinsics.txt, img_shape.txt, ...
      depth/depth000000.png, ...
      metadata.json
torch_hub/checkpoints/
  vgg.pth
  vgg16-397923af.pth
```

Reference pose files support evaluation. Final estimated-pose templates use `READER.pose_source: identity` and `TSDF.use_gt_pose: false`; training does not derive camera tracking or scene scale from reference poses.

## Build profiles

```bash
bash scripts/thor_container.sh image
bash scripts/thor_container.sh check
bash scripts/thor_container.sh build
bash scripts/thor_container.sh baseline-build
bash scripts/thor_container.sh source-check
```

| Command | Binary | TSDF hash profile | BasicEngine capacity instrumentation |
|---|---|---|---|
| `build` | `build_thor/slam_trainer` | `medium_128k` | OFF |
| `baseline-build` | `build_original/slam_trainer` | `legacy_pool_2m` | ON |

The BasicEngine-specific compile flag does not disable the proposed system's normal timing, tracking, paging or loop diagnostics. The proposed build uses MultiEngine. The original control uses BasicEngine with 2,097,152 voxel blocks, 8,388,608 hash buckets and 1,048,576 excess entries. It retains the original tracker/fusion policy with additional allocation capacity; it is not an exact reproduction of an unpublished paper recipe.

The wrapper selects fast math OFF, TensorBoard OFF, the optional remote viewer OFF and native SM 110. It sets `NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` to preserve the numerical environment: TF32 is disabled and the cuBLAS workspace configuration is explicit. Preserve both variables for equivalent native invocations.

The image tag is `gps-slam-thor:final-20260915`; its Dockerfile pins the NVIDIA base by digest. The default build concurrency is six workers; override with `GPS_SLAM_BUILD_JOBS`. For a native build with a compatible Torch development environment already installed:

```bash
GPS_SLAM_TSDF_HASH_PROFILE=medium_128k \
  GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION=OFF bash scripts/thor_build.sh
GPS_SLAM_TSDF_HASH_PROFILE=legacy_pool_2m \
  GPS_SLAM_BASIC_CAPACITY_INSTRUMENTATION=ON \
  GPS_SLAM_THOR_BUILD_DIR="$PWD/build_original" bash scripts/thor_build.sh
```

Clean builds compile the bundled rasterizer. Leave the optional SHA-checked rasterizer-archive reuse variables unset for normal use. `GPS_SLAM_BUILD_REMOTE_VIEWER=1` enables the optional viewer.

## Run the proposed system

The 11 training templates under `configs/final/{tum,replica}` retain the final proposed-method numeric settings, full resolution, frame range, seed and immediate Gaussian scheduling. Only paths and campaign-only labels are adapted. Pipeline pose-graph processing is enabled for both datasets; the separate internal MultiEngine loop switch remains disabled. Candidate verification gates decide whether to apply a correction.

```bash
bash scripts/thor_container.sh run python3 scripts/thor_preflight.py \
  configs/final/tum/fr1_desk.yaml
bash scripts/thor_container.sh run ./build_thor/slam_trainer \
  configs/final/tum/fr1_desk.yaml
bash scripts/thor_container.sh run python3 scripts/thor_preflight.py \
  configs/final/replica/office0.yaml
bash scripts/thor_container.sh run ./build_thor/slam_trainer \
  configs/final/replica/office0.yaml
```

Generic preflight tuning warnings about prefetch and checkpoint saving are advisory. The final reproduction presets deliberately keep prefetch off and required map saving on; preserve these settings when reproducing them.

Outputs go under `/workspace/runs/final/proposed/{dataset}/{scene}`. Copy a template and choose a distinct workspace for another run. Outside Docker, replace the two container paths with host paths. Keep all other settings fixed for a controlled correspondence check.

Use `configs/final/original/tum/{fr1_desk,fr2_xyz,fr3_office}.yaml` with `build_original/slam_trainer` for the original-algorithm control. Do not run a capacity-control config with the proposed binary. Older `configs/release/` templates and config generators remain compatibility utilities; they are not final proposed presets.

## Render and evaluate a saved map

After training the matching Replica scene:

```bash
bash scripts/thor_container.sh run ./build_thor/slam_trainer \
  configs/final/replica/office0_eval.yaml
```

Each `_eval.yaml` shares its training workspace and contains 500 fixed frame indices. Its final-map selection settings are distinct from online rendering. The included `saved_overlap_v2` fix restores recorded frame/submap overlap membership when rendering a reloaded map.

For a separately named all-view evaluation, copy the eval YAML, remove `PIPE.eval_frame_indices`, retain `eval_stride: 1`, `eval_start_offset: 0`, `eval_max_frames: 0`, and use a new `PIPE.eval_path`, such as `/val_full2000`. Retain the map and input sequence. This renders saved state without retraining or replacing the original evaluation. Never overwrite results you need to retain.

```bash
bash scripts/thor_container.sh run python3 scripts/metric_general.py \
  --gt_path /workspace/data/replica_prepared/office0 \
  --render_path /workspace/runs/final/proposed/replica/office0/eval_final_500/render
```

Check the generated eval layout before choosing a render path. The scorer reports PSNR, SSIM and VGG-LPIPS with normalized `[-1,1]` LPIPS inputs. Depth masking is opt-in; actual frame support is recorded. Offline scoring is separate from processing FPS. `scripts/ate_general.py` accepts GT and estimated per-frame pose directories and separately reports mean error and true rigid SE(3) RMSE, without scale fitting.

## Prepare data

Obtain Replica and TUM RGB-D from their original distributions under their terms. Data and weights are not in Git. Preparation requires OpenCV, NumPy, SciPy, Pillow, PyYAML and tqdm, supplied in the image. It writes new data, so the usual read-only dataset mount must not be its output destination.

```bash
python3 scripts/replica_prepare_scene.py \
  --raw-root /path/to/raw/Replica --scene office0 \
  --output-root "$GPS_SLAM_DATA_ROOT/replica_prepared" --target-frames 2000
python3 scripts/tum_preprocess.py \
  --data-root "$GPS_SLAM_DATA_ROOT/tum_rgbd" --paper-only --copy
```

For controlled comparisons preserve raw images, depth units, calibration and associations. Final GPS TUM templates retain the native depth divisor of 5000. The separate comparator calibration protocol is outside this repository.

## Validate and record

```bash
bash scripts/thor_container.sh regressions
bash scripts/thor_container.sh run python3 scripts/capacity_profiles_regression.py \
  --output /workspace/runs/capacity-check.json
bash scripts/thor_container.sh run python3 scripts/final_native_regression.py \
  --output-dir /workspace/runs/final-native-check --build-dir build_thor
```

The native regression uses a fresh output directory and short Replica input. It checks missing/poisoned GT isolation, complete saved-pose requirements and render sampling. Keep diagnostics separate from measurements. Source-manifest checks cover metric scripts, dependencies and runtime wrappers; they do not claim independently built executables have identical hashes.

Record configuration, source/binary/image hashes, input support, seed, tracking/completion state, timing boundaries and hardware state for new measurements. `telemetry-run` wraps a single command with Thor telemetry. Throughput depends on the declared timed work and is not a sensor-rate deadline guarantee. The benchmark supervisor, comparator harness, result tables and thesis package are deliberately outside this implementation distribution.
