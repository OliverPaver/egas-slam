# GPS-SLAM Full Codebase Guide

Last updated: 2026-08-17

This document is the single consolidated guide for understanding this GPS-SLAM
working tree. It covers the original project, the current research extensions,
the main runtime flow, the most important files, the configuration surface,
datasets, metrics, diagnostics, and the practical rules for safely navigating
and modifying the code.

It is written for a reader who may know SLAM, TSDF fusion, or Gaussian
Splatting at a high level, but does not yet know how this specific repository is
organized.

## 1. Project Purpose

GPS-SLAM is a real-time RGB-D SLAM and reconstruction system. The base paper
system combines two map representations:

- A colorized SDF/TSDF map, built quickly from RGB-D fusion.
- A 3D Gaussian residual/detail model, optimized online to recover visual
  detail that the SDF raycast does not capture.

The original released path uses one global TSDF map and one global Gaussian
model. This working tree preserves that path, but adds opt-in research
extensions:

- MultiEngine SDF submaps.
- One local Gaussian model per SDF submap.
- Bounded SDF and Gaussian residency for long runs and embedded systems.
- Online pose-graph loop closure that moves rigid submap transforms.
- TUM RGB-D dataset support for ATE/RPE and loop-closure evaluation.
- Replica multi-scene validation runners for image quality and geometry.
- Structured diagnostics for submaps, timing, paging, loop candidates, and
  render-selection decisions.

The core design rule for the research path is:

```text
Submap-local TSDF voxels and Gaussian tensors stay local.
Loop closure updates only each submap's local_to_global/global_to_local transform.
```

That rule keeps online loop correction cheap enough for an embedded target such
as Jetson Thor.

## 2. High-Level Runtime Model

The main executable is `slam_trainer`.

At a high level:

1. A YAML config is loaded.
2. `DatasetReader` reads RGB-D frames, poses, intrinsics, and optional point
   clouds.
3. `createTsdfEngine()` creates an InfiniTAM TSDF engine through a facade.
4. `SLAMPipeline` is configured.
5. `SLAMGaussianModel` is configured.
6. Training/reconstruction runs frame by frame.
7. The TSDF engine tracks/fuses each RGB-D frame.
8. The pipeline raycasts the TSDF map.
9. The Gaussian model is locally optimized against the SDF-guided target.
10. New Gaussians are inserted where reconstruction error is high.
11. Diagnostics are written.
12. At the end, the system saves poses, optional corrected poses, optional
    Gaussian models/submaps, optional TSDF checkpoints, optional meshes, and
    optional rendered evaluation images.

The three `work_mode` values are:

| Mode | Meaning |
| --- | --- |
| `train` | Full SDF fusion plus Gaussian training. |
| `recon` | SDF reconstruction/tracking only. Gaussian work is skipped. |
| `eval` | Load saved state and render evaluation images. |

The original path remains:

```yaml
PIPE:
  TSDF:
    engine: basic
  gaussian_submaps:
    enable: false
```

The hybrid submap path is opt-in:

```yaml
PIPE:
  TSDF:
    engine: multi
  gaussian_submaps:
    enable: true
    ownership: tsdf
    coordinate_frame: local
```

## 3. Repository Map

Top-level files and directories:

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Main build file. Builds `slam_trainer`, `remote_viewer`, `gsplat`, `rasterizer`, and InfiniTAM. |
| `slam_trainer.cpp` | Main training/reconstruction/evaluation executable entry point. |
| `remote_viewer.cpp` | TCP server for the external SIBR viewer. |
| `include/` | Core non-SLAM headers: dataset, base pipeline, Gaussian model, tensor utilities. |
| `src/` | Implementations for `include/`. |
| `slam/` | SLAM-specific pipeline, Gaussian insertion, TSDF facade, and InfiniTAM wrapper. |
| `slam/TsdfFusion/` | CLI/UI engines that feed frames into InfiniTAM. |
| `gsplat/` | C++/CUDA Gaussian rasterizer wrapper and kernels. |
| `InfiniTAM/` | Bundled TSDF/tracking/reconstruction library. This tree has been modified for MultiEngine submaps, paging, diagnostics, and pose graph support. |
| `ThirdLibs/` | Third-party dependency sources/installs. Local binary installs are machine-specific. |
| `configs/` | Release configs and generated/experiment configs. |
| `scripts/` | Dataset preparation, metrics, rescoring, and diagnostics. |
| `run/` | Batch runners and reproducible experiment generators. |
| `docs/` | Research documentation, status notes, config guide, output guide, and this document. |
| `data/` | Local datasets. Not part of code. Often omitted from transfers. |
| `output/` | Local run outputs. Not part of code. Often pruned/archived. |
| `build*` | Local build directories. Not portable across machines. |

The most important source files to understand first:

| Path | Why it matters |
| --- | --- |
| `slam_trainer.cpp` | Owns process-level control flow for train/recon/eval. |
| `slam/slam_pipeline.h` | Declares almost all SLAM, submap, pose graph, render, and diagnostic state. |
| `slam/slam_pipeline.cpp` | Implements the frame loop, Gaussian submaps, loop closure, rendering, saving/loading, and diagnostics. |
| `slam/InfiniTAM_tools.h/.cpp` | Facade between GPS-SLAM and BasicEngine/MultiEngine InfiniTAM. |
| `slam/TsdfFusion/CLIEngine.h/.cpp` | Feeds preloaded or on-demand images into InfiniTAM. |
| `src/dataset_reader.cpp` | Dataset parsing, lazy/on-demand image loading, TUM PNG support, frame IDs, pose IO. |
| `src/raw_gs_model.cpp` | Gaussian forward pass, loss, densification, pruning, optimizer state. |
| `slam/slam_gs_model.cpp` | Online Gaussian insertion from RGB-D/SDF evidence. |
| `InfiniTAM/ITMLib/Core/ITMMultiEngine.tpp` | MultiEngine frame processing, map residency, combined raycasts, save/load. |
| `InfiniTAM/ITMLib/Engines/MultiScene/ITMActiveMapManager.cpp` | Active/candidate SDF submap lifecycle and handoff policy. |
| `InfiniTAM/MiniSlamGraphLib/` | Small SE3 pose graph solver used by loop closure. |

## 4. Build System

The root CMake project is named `opensplat`, but the important executables are:

- `slam_trainer`
- `remote_viewer`

The build depends on:

- C++17
- CUDA
- LibTorch
- OpenCV
- Eigen3
- yaml-cpp
- tensorboard_logger
- tinyply
- indicators
- Pangolin
- Boost.Asio for the viewer path
- InfiniTAM internal libraries
- gsplat/rasterizer CUDA kernels

Important CMake options:

| Option | Values | Meaning |
| --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | e.g. `86`, `110` | CUDA architecture target. Must be set correctly for the machine. |
| `GPS_SLAM_TSDF_HASH_PROFILE` | `legacy_large`, `medium_128k`, `compact_64k` | Compile-time TSDF hash capacity profile. Runtime YAML must match this for MultiEngine. |
| `OPENSPLAT_MAX_CUDA_COMPATIBILITY` | `ON`, `OFF` | Attempts broad arch compatibility. Usually leave off and set architecture explicitly. |
| `GPS_SLAM_USE_NVML` | `ON`, `OFF` | Uses NVML for memory diagnostics when found. With NVML disabled or unavailable, diagnostics use `cudaMemGetInfo`. |

The default CMake CUDA architectures are `70;75;80`. That is not right for all
systems. For this desktop with RTX 3070-class hardware, use the local tested
builds. For Jetson Thor, use:

```bash
cmake -S . -B build_thor \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=110 \
  -DGPS_SLAM_TSDF_HASH_PROFILE=medium_128k
cmake --build build_thor -j$(nproc)
```

Hash-profile builds used in this project:

| Build dir | Intended profile |
| --- | --- |
| `build_legacy/` | `legacy_large`, mainly BasicEngine/paper-style baselines. |
| `build_medium/` | `medium_128k`, main MultiEngine quality profile. |
| `build_compact/` | `compact_64k`, embedded stress profile. |
| `build/` | Local default build. Check its CMake cache before comparing. |

`CMakeLists.txt` prepends local `ThirdLibs/libtorch` and `ThirdLibs/install`
only when they exist, while preserving command-line/toolchain prefix paths.
Protobuf and NVML no longer use x86-specific library paths. The dependency
install and LibTorch binaries are still architecture-specific and must be
rebuilt/reinstalled on Thor; `scripts/thor_build.sh` performs the native build.

## 5. Main Executable: `slam_trainer.cpp`

`slam_trainer.cpp` is the process entry point.

Main functions:

- `runTrainer(argc, argv)`
- `main(argc, argv)`
- `evalSplitCameras(...)`
- `selectEvalCameras(...)`
- `writeEvalSamplingManifest(...)`

Runtime flow:

1. Require one YAML file argument.
2. Load YAML with `YAML::LoadFile`.
3. Read `workspace_dir` and `work_mode`.
4. Validate `PIPE.TSDF` through `validateTsdfConfig`.
5. Set `CUDA_VISIBLE_DEVICES` from `dev_id`.
6. Create `DatasetReader` and load cameras.
7. Create TSDF engine through `createTsdfEngine`.
8. Create `SLAMPipeline` and attach `TsdfEngineFacade`.
9. Create and configure `SLAMGaussianModel`.
10. For `train` or `recon`, run `pipe.SLAMTrainCams(...)`.
11. Optionally save Gaussian model/submaps, TSDF engine, and mesh.
12. Save raw poses to `eval_path/pose`.
13. If loop/submap corrected poses exist, save them to `eval_path/corrected_pose`.
14. If `eval_after_train` is true, render eval images.
15. For `eval`, load saved engine/model/submaps and render selected cameras.

The `PIPE.bypass_libtorch_atexit` flag calls `std::_Exit(exit_code)` after all
outputs are flushed. It exists because some long/eval runs showed crashes during
LibTorch static finalization after useful outputs were already written. It does
not change SLAM computation, but it skips normal C++ static cleanup at exit.

## 6. Dataset Reader

Key files:

- `include/dataset_reader.h`
- `src/dataset_reader.cpp`

Main classes/structs:

- `Points`
- `Camera`
- `DatasetReader`
- `RandomSelector<T>`

### `Camera`

`Camera` is the central per-frame record.

Important fields:

| Field | Meaning |
| --- | --- |
| `id` | Sequential index in the loaded train/val vector. |
| `width`, `height`, `fx`, `fy`, `cx`, `cy` | Image size and intrinsics. |
| `image`, `depth` | Torch tensors for RGB and depth. Can be lazily loaded. |
| `c2w` | Ground-truth camera-to-world pose, normalized relative to first frame. |
| `c2w_slam` | Estimated SLAM camera-to-world pose. Raw trajectory output uses this. |
| `c2w_local` | Camera pose in owner SDF submap local frame. |
| `has_local_pose` | Whether `c2w_local` is valid. |
| `owner_submap_id` | SDF submap that owns this frame. |
| `overlap_submap_ids` | Active/overlap SDF submaps for this frame. |
| `frame_id_str` | Stable string ID such as `000123`, used for filenames and metrics. |
| `release_after_frame` | If true, image/depth tensors are released after processing. |

Lazy loading:

- `ensureLoaded()` loads RGB/depth if missing.
- `releaseImages()` clears tensors if `release_after_frame` is true.

This is important for Thor because preloading full image sequences can consume
shared memory. On-demand input relies on `release_after_frame`.

### `DatasetReader`

`DatasetReader` reads:

- RGB frames
- depth frames
- pose files
- optional initial point cloud
- intrinsics and image shape from YAML

Relevant YAML keys:

```yaml
READER:
  input_dir: data/replica/office0
  image_path: camera
  pose_path: camera
  depth_path: depth
  image_prefix: frame
  image_extension: .jpg
  depth_prefix: depth
  depth_extension: .png
  pose_prefix: pose
  pose_extension: .txt
  preload_images: true
  depth_scale: 6553.5
  start_frame: 0
  end_frame: 1999
  frame_step: 1
  intrinsics: [600, 600, 599.5, 339.5]
  image_shape: [1200, 680]
  downscale_factor: 1
  test_split_interval: -1
```

TUM support uses PNG RGB:

```yaml
READER:
  image_extension: .png
  depth_extension: .png
  depth_scale: 5000.0
  image_shape: [640, 480]
```

Pose normalization:

- The first input pose is treated as the sequence origin.
- Every pose is transformed by `inverse(first_pose) * current_pose`.
- That normalized frame is what ATE scripts use when comparing saved output to
  the normalized input camera poses.

Train/val split:

- If `test_split_interval < 0`, every frame is both train and val.
- Otherwise frames divisible by `test_split_interval` go to `val_vec`; the rest
  go to `train_vec`.

## 7. Base Pipeline

Key files:

- `include/pipeline.h`
- `src/pipeline.cpp`

`Pipeline` is the generic Gaussian training/evaluation base class. It does not
know about live TSDF submaps. `SLAMPipeline` inherits from it.

Important responsibilities:

- Parse common `PIPE` config.
- Create output directories.
- Own TensorBoard logger.
- Run generic `raw` or `ges` Gaussian training loops.
- Save global Gaussian model.
- Render evaluation images for the non-submap path.
- Log scalar/image diagnostics.

Important fields:

| Field | Meaning |
| --- | --- |
| `max_iterations` | Generic Gaussian optimization iterations. |
| `random_seed` | Reproducible sampling seed. |
| `enable_densify` | Whether Gaussian densify/prune logic is active. |
| `eval_after_train` | Render eval images after train/recon. |
| `save_after_train` | Save models/engine/mesh after train/recon. |
| `eval_split`, `eval_stride`, `eval_max_frames`, `eval_start_offset` | Eval camera sampling. |
| `workspace_dir`, `model_path`, `log_path`, `eval_path`, `tb_path` | Output layout. |
| `weight_configs` | Loss weights. |
| `vis_configs` | Visualization ranges. |

`Pipeline::rawTrainCams()` is the normal 3DGS-style train loop.

`Pipeline::gesTrainCams()` trains against SDF/mesh-rendered base images and
depths.

The online GPS-SLAM path usually goes through `SLAMPipeline::SLAMTrainCams()`
instead of these generic loops.

## 8. Gaussian Model Stack

Key files:

- `include/raw_gs_param.h`
- `src/raw_gs_param.cpp`
- `include/raw_gs_model.h`
- `src/raw_gs_model.cpp`
- `slam/slam_gs_model.h`
- `slam/slam_gs_model.cpp`

### `RawGaussianParams`

Owns the actual tensors:

- `means`
- `scales`
- `quats`
- `featuresDc`
- `featuresRest`
- `opacities`
- `exposure`

Important methods:

- `init(...)`
- `clear()`
- `requireGrad(...)`
- `toGPU()`, `toCPU()`, `toDevice(...)`
- `cloneDetached()`
- `add(...)`
- `remove(...)`
- `savePly(...)`
- `saveTensor(...)`
- `loadTensor(...)`

The model stores scales/opacities in parameter space. `getRealScales()` applies
`exp`; `getRealOpacities()` applies `sigmoid`.

### `RawGaussianModel`

Adds rendering, losses, optimization, densification, and saving.

Important methods:

- `loadConfig(...)`
- `forward(...)`
- `rawForward(...)`
- `gesForward(...)`
- `computeLoss(...)`
- `stepPostBackward(...)`
- `densifiyGs(...)`
- `prunePoints(...)`
- `initOptimizers(...)`
- `optimizersStep()`
- `clearOptimizers()`

Render modes:

| Mode | Meaning |
| --- | --- |
| `raw` | Standard Gaussian rendering. |
| `ges` | Gaussian-Enhanced SDF rendering. Uses TSDF/SDF base color/depth plus Gaussian residual/detail contribution. |

`gesForward()` is central to GPS-SLAM quality: it composites Gaussian
contributions with SDF raycast evidence rather than asking Gaussians to model
the entire scene alone.

### `SLAMGaussianModel`

Adds:

- `addGaussians(...)`

This creates new Gaussians online from selected pixels in the current frame,
using the current RGB-D/normal/raycast evidence. It is called by
`SLAMPipeline::initNewGaussians(...)`.

## 9. Rasterizer

Key files:

- `gsplat/gsplat_wapper.hpp`
- `gsplat/gsplat_wapper.cpp`
- `gsplat/rasterizer/*.cu`

The rasterizer is a custom CUDA/LibTorch binding derived from gsplat-style
kernels.

Important kernel groups:

| Files | Purpose |
| --- | --- |
| `fully_fused_projection_*` | Project Gaussians to screen and prepare tile intersections. |
| `rasterize_to_pixels_*` | Rasterize Gaussian color/depth/alpha. |
| `rasterize_to_pixels_*_ges*` | GES-specific rasterization/compositing support. |
| `compute_sh_*` | Spherical harmonics color evaluation. |
| `quat_scale_to_covar_preci_*` | Convert Gaussian rotation/scale to covariance/precision. |
| `world_to_cam_*` | Coordinate transform kernels. |
| `ssim.*` | SSIM support. |
| `simple_knn.*` | KNN support for initialization/scale estimation. |

The C++ wrapper exposes autograd-aware functions to LibTorch so Gaussian
parameters can be optimized by standard Torch optimizers.

## 10. TSDF/InfiniTAM Integration

Key files:

- `slam/InfiniTAM_tools.h`
- `slam/InfiniTAM_tools.cpp`
- `slam/TsdfFusion/CLIEngine.h`
- `slam/TsdfFusion/CLIEngine.cpp`
- `InfiniTAM/ITMLib/Core/ITMBasicEngine.*`
- `InfiniTAM/ITMLib/Core/ITMMultiEngine.*`

### `CLIEngine`

`CLIEngine` is a thin frame feeder around InfiniTAM.

It supports:

- Preloaded input: vectors of `ITMUChar4Image*` and `ITMShortImage*`.
- On-demand input: a callback that loads one frame into reusable InfiniTAM
  image buffers.

Methods:

- `Initialise(...)`
- `InitialiseOnDemand(...)`
- `ProcessFrame()`
- `Shutdown()`

On-demand mode is important for full-resolution and Thor testing because it
avoids keeping every RGB-D frame resident.

The on-demand callback also has two deployment controls:

- `PIPE.TSDF.input_tensor_bridge: direct` copies CPU Torch RGB-D tensors into
  the reusable InfiniTAM input buffers without allocating intermediate OpenCV
  images. `opencv_legacy` preserves the old A/B path.
- `PIPE.TSDF.input_tensor_bridge_validate: true` executes and compares both
  bridges for every frame, aborting on any RGB/depth mismatch. It requires
  direct on-demand input and is a validation mode, not a throughput mode.

When `PIPE.TSDF.input_prefetch` is enabled, a single persistent worker loads
the next camera while the current frame runs. The worker owns its lifecycle,
propagates loader exceptions, and joins during shutdown; the old per-frame
`std::async` creation is no longer used.

### `TsdfEngineFacade`

`TsdfEngineFacade` hides whether the backend is BasicEngine or MultiEngine.

Important methods:

- `processFrame()`
- `getGlobalCameraPose()`
- `getPrimarySubmapId()`
- `getSubmapCount()`
- `getActiveSubmapIds()`
- `consumeRecentTransitionEvents()`
- `getGlobalToLocal(submap_id)`
- `setGlobalToLocal(submap_id, transform)`
- `getSubmapBlockCount(submap_id)`
- `getSubmapCapacity(submap_id)`
- `getSubmapLocalBounds(submap_id)`
- `isSubmapResident(submap_id)`
- `prefetchRaycastMaps(...)`
- `trimResidency(...)`
- `synchronizePaging()`
- `runRaycast(...)`
- `save(...)`, `load(...)`, `saveMesh(...)`

This facade was added to avoid scattered direct downcasts from
`ITMMainEngine` to `ITMBasicEngine` or `ITMMultiEngine`.

### `validateTsdfConfig`

`validateTsdfConfig()` rejects unsupported or internally inconsistent YAML
before the run starts.

It checks:

- `PIPE.TSDF.engine` is `basic` or `multi`.
- `PIPE.TSDF.input_mode` is `preload` or `on_demand`.
- MultiEngine boundary/transition policies are legal.
- Occupancy thresholds are valid.
- Paging is not requested with `legacy_large`.
- Runtime `PIPE.TSDF.multi.hash_profile` matches the binary's compile-time
  `GPS_SLAM_TSDF_HASH_PROFILE`.
- Tracking-boundary and primary-failure handoff settings are valid.

Tracking profiles are resolved before either engine is constructed:

- `legacy` keeps the original robust extended tracker (20 coarse / 50 fine
  maximum iterations).
- `efficient` uses the same tracker and thresholds with a 20/20 cap.
- `robust_rpe` additionally lowers `framesToSkip` from 20 to 10, admitting
  moderately mature TSDF correspondences while retaining confidence weights.

`legacy` is the default. The measured frontend study and rejected alternatives
are recorded in `docs/tracking_frontend_research_20260817.md`.

This validation is important because many research profiles are generated by
scripts. Bad configs should fail early.

## 11. InfiniTAM BasicEngine Path

The BasicEngine path is the original single-map TSDF route. It should remain
the baseline for comparisons.

In BasicEngine mode:

- There is one TSDF scene.
- `TsdfEngineFacade::getPrimarySubmapId()` returns `0`.
- `getSubmapCount()` returns `1`.
- `getActiveSubmapIds()` returns `{0}`.
- Gaussian submaps are usually disabled.
- Unified Gaussian model behavior should match the original project unless an
  opt-in config changes it.

BasicEngine is still affected by shared code around:

- Dataset loading.
- `slam_trainer` control flow.
- `Pipeline`/`SLAMPipeline` common logic.
- Evaluation scripts.
- CMake and dependency setup.

So when checking "original baseline still works," use a clean BasicEngine
config and a `legacy_large` build.

## 12. InfiniTAM MultiEngine SDF Submaps

Key files:

- `InfiniTAM/ITMLib/Core/ITMMultiEngine.h`
- `InfiniTAM/ITMLib/Core/ITMMultiEngine.tpp`
- `InfiniTAM/ITMLib/Engines/MultiScene/ITMActiveMapManager.h`
- `InfiniTAM/ITMLib/Engines/MultiScene/ITMActiveMapManager.cpp`
- `InfiniTAM/ITMLib/Objects/Scene/ITMLocalMap.h`
- `InfiniTAM/ITMLib/Objects/Scene/ITMLocalVBA.h`
- `InfiniTAM/ITMLib/Engines/MultiScene/ITMMapGraphManager.*`

MultiEngine uses multiple rigid local TSDF maps.

Important concepts:

| Term | Meaning |
| --- | --- |
| Primary local map | The SDF submap currently owning/tracking the live camera. |
| Candidate/new local map | A prospective new submap during a transition. |
| Passive map | An older submap not actively tracking/fusing. |
| Active map set | Primary plus transition/candidate maps and other active links. |
| `global_to_local` | Transform from global SLAM frame to a local submap frame. |
| `local_to_global` | Inverse transform that places a submap into global space. |

### Active-map lifecycle

`ITMActiveMapManager` owns most lifecycle decisions.

Important methods:

- `configureSequentialPolicy(...)`
- `shouldStartNewArea()`
- `initiateNewLocalMap(...)`
- `maintainActiveData(...)`
- `CheckSuccess_newlink(...)`
- `AcceptNewLink(...)`
- `shouldMovePrimaryLocalMap(...)`
- `canPromoteCandidateAfterPrimaryFailure(...)`
- `preserveCandidateGlobalPoseForHandoff(...)`
- `retireNewLocalMap(...)`
- `recordTrackingResult(...)`
- `recordRawTrackingResult(...)`
- `logTransitionEvent(...)`

The current robust policy separates:

- Creating/warming candidates from weak tracking.
- Promoting candidates only when stronger handoff gates pass.

This matters because earlier experiments showed weak tracking could trigger
over-eager primary handoffs on easy TUM scenes. The current paper-parity
settings use `primary_failure_handoff_raw_gate: failed_only` so weak frames can
warm a candidate without immediately forcing promotion.

### Capacity and hash profiles

`GPS_SLAM_TSDF_HASH_PROFILE` controls compile-time TSDF capacity.

Profiles:

| Profile | Intended use |
| --- | --- |
| `legacy_large` | Original large-map baseline. Not suitable for bounded paging. |
| `medium_128k` | Main MultiEngine quality profile. |
| `compact_64k` | Embedded/Thor-style stress profile. |

Runtime YAML must match the compiled profile:

```yaml
PIPE:
  TSDF:
    multi:
      hash_profile: medium_128k
```

### Residency and paging

`ITMLocalMap` now supports:

- resident GPU data;
- CPU page cache;
- disk page snapshots;
- async page-out;
- async page-in;
- sparse serialization;
- dirty-map safety.

Important safety rule:

```text
Do not evict a dirty passive SDF map until its snapshot is complete.
```

`ITMMultiEngine::enforceResidency(...)` decides which maps stay resident. The
pipeline adds pins for primary, candidate, render-selected maps, and loop
observation endpoints.

### Combined raycasting

MultiEngine has internal combined rendering, but the current GPS-SLAM path
prefers a bounded compositor in `SLAMPipeline::runRaycastByCamWithTsdfSubmaps`.

That compositor:

1. Selects a bounded list of SDF submaps.
2. Raycasts each selected local map.
3. Transforms local vertices into global/camera frame.
4. Combines maps by owner-priority, nearest-depth, or handoff-aware policy.
5. Produces a single global color/depth/vertex/confidence tensor.

This is the rendering path that responds to loop-closure transform updates.

## 13. `SLAMPipeline`

`SLAMPipeline` is the central class for this project.

Key files:

- `slam/slam_pipeline.h`
- `slam/slam_pipeline.cpp`

It extends `Pipeline` with:

- TSDF engine facade ownership.
- Online SLAM frame loop.
- Keyframe management.
- Local frame raycasting.
- Gaussian insertion.
- Gaussian local optimization.
- Gaussian submaps.
- SDF submap event handling.
- TSDF/Gaussian residency.
- Pose graph scaffolding and loop closure.
- Corrected trajectory export.
- Submap/global rendering.
- Structured diagnostics.

### Frame-loop overview

`SLAMPipeline::SLAMTrainCams(...)` is the main online loop.

Per frame:

1. Assert the camera index matches `tsdf_engine->currentFrameNo()`.
2. `tsdf_engine->processFrame()` tracks and fuses the frame.
3. Read raw/policy tracking result.
4. Update frontend health counters.
5. Fail if configured strict raw tracking gates are violated.
6. Get primary SDF submap.
7. Get global camera pose from TSDF engine.
8. Store `cams[i].c2w_slam`.
9. Store owner/overlap submap IDs.
10. Compute `cams[i].c2w_local = owner.global_to_local * c2w_slam`.
11. Store `frame_owner_submap` and `frame_local_pose`.
12. Log submap state.
13. Move current camera to GPU.
14. If local Gaussian submaps are enabled, switch `curr_cam.c2w_slam` to the
    local pose for local optimization.
15. Create/sync Gaussian submaps if needed.
16. Process TSDF transition events.
17. Update keyframes/local windows.
18. Record training/loop observations.
19. Warm ORB loop observations if enabled.
20. Optionally run online pose graph updates.
21. If `work_mode == recon`, skip Gaussian work.
22. Raycast local frame.
23. Insert new Gaussians.
24. Local optimize.
25. Prune redundant Gaussians.
26. Check keyframe error.
27. Optimize handoff/finalize/refine submaps as configured.
28. Write timing and diagnostics.

At sequence end:

1. Finalize expired handoff submaps.
2. Run optional Gaussian quality finalization.
3. Flush pending pose graph transform updates.
4. Run final pose graph update if configured.
5. Write final stats.

### Important `SLAMPipeline` state groups

Baseline frame/keyframe state:

- `curr_frame_id`
- `curr_dataset_frame_id`
- `curr_cam`
- `localframe_cam_window`
- `localframe_raycast_window`
- `keyframe_cam_list`
- `keyframe_loss_dict`
- `opt_cam_list`
- `opt_raycast_list`

TSDF state:

- `TsdfEngineFacade *tsdf_engine`
- `ITMMainEngine *main_engine`
- `voxel_size`
- raycast temporary images
- TSDF raw tracking health counters

Gaussian submap state:

- `use_gaussian_submaps`
- `gaussian_submaps`
- `active_submap_idx`
- `frame_owner_submap`
- `frame_overlap_submaps`
- `frame_local_pose`
- render-selection diagnostic counters

Pose graph state:

- `hybrid_pose_graph_nodes`
- `hybrid_pose_graph_sequential_constraints`
- `hybrid_pose_graph_loop_constraints`
- `hybrid_pose_graph_verified_loop_pool`
- ORB feature cache
- BoW histograms
- pending transform updates
- deferred observation residency pins
- frontend health counters

Diagnostics:

- `submap_events_log`
- `frame_assignments_log`
- `submap_stats_log`
- `tracking.csv`
- `timing.csv` when timing is compiled in

## 14. Gaussian Submaps

Gaussian submaps are controlled by:

```yaml
PIPE:
  gaussian_submaps:
    enable: true
    ownership: tsdf
    coordinate_frame: local
```

The main record is `GaussianSubmap` in `slam/slam_pipeline.h`.

Important fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable hybrid submap ID. |
| `sdf_submap_id` | Owner SDF submap ID from MultiEngine. |
| `start_frame_id`, `end_frame_id` | Dataset frame range. |
| `frozen` | Whether normal online training should stop. |
| `freeze_pending` | Waiting for handoff/finalization before becoming frozen. |
| `candidate_warming` | Candidate SDF submap exists but is not accepted primary yet. |
| `outgoing_handoff` | Previous primary gets bounded handoff optimization. |
| `retired` | Candidate or invalid submap should not render/train. |
| `gaussian_resident` | Gaussian tensors are currently resident. |
| `transform_epoch` | Incremented when loop closure changes transform. |
| `global_to_local`, `local_to_global` | Shared SDF/Gaussian submap transform. |
| `model` | `SLAMGaussianModel` containing local Gaussian tensors. |
| `keyframe_cam_list` | Local keyframes. |
| `training_observation_cam_list` | Stored local observations for later refinement. |
| `loop_observation_cam_list` | Stored local observations for loop detection/verification. |

In local-coordinate mode:

```text
camera.c2w_slam for Gaussian optimization = camera.c2w_local
Gaussian means are in owner SDF local coordinates
Gaussian render converts global camera pose into each selected submap's local frame
```

### Submap lifecycle

Creation:

- Initial Gaussian submap is created for the first active SDF submap.
- Later Gaussian submaps are created from TSDF transition events.
- One accepted SDF local map should have exactly one Gaussian submap.

Candidate warming:

- A candidate SDF map can receive a Gaussian submap before it becomes primary.
- Warming lets the candidate avoid freezing with zero/young Gaussians.
- Candidate insertion can be delayed or capped by maturity settings.

Handoff:

- When the active map changes, the outgoing Gaussian submap enters a bounded
  handoff period.
- It can keep rendering and optionally get a final optimization pass.
- The goal is to avoid visible boundary quality loss.

Graduation/freeze:

- A submap may run bounded graduation optimization.
- Readiness diagnostics compare owner SDF RGB L1 against hybrid RGB L1.
- If the hybrid is already better enough, extra graduation can be skipped.
- Freezing should never block indefinitely.

Retirement:

- Rejected TSDF candidates can retire their Gaussian submap.
- Retired submaps should not render or train.

Residency:

- Active/candidate/recent Gaussian submaps stay GPU-resident.
- Frozen submaps can page to CPU depending on config.
- Optimizer state is recreated only when a submap becomes trainable again.

### Gaussian submap invariants

Do not break these:

```text
Every non-retired Gaussian submap has a valid sdf_submap_id.
Its global_to_local matches the TSDF engine's global_to_local for that SDF map.
local_to_global is the inverse of global_to_local.
Local Gaussian tensors are not transformed by loop closure.
Future camera local/global conversions use the current submap transform.
```

## 15. Gaussian Insertion and Optimization

Important functions:

- `localFrameRaycast()`
- `initNewGaussians(...)`
- `SLAMGaussianModel::addGaussians(...)`
- `localOptimize(...)`
- `removeRedundantGs(...)`
- `checkKeyFrameError(...)`

The online Gaussian process is:

1. Raycast TSDF for local frame/keyframes.
2. Render current Gaussian model with GES.
3. Compute color/depth/error masks.
4. Select candidate pixels for new Gaussian insertion.
5. Add Gaussians in the correct coordinate frame.
6. Optimize over recent local frames/keyframes.
7. Prune redundant/low-opacity/oversized Gaussians.

Insertion policies include:

- `random`
- `error_topk`
- `tile_error_topk`
- `adaptive_tile_error_topk`

The tiled policies were added because pure top-k can concentrate all new
Gaussians in one high-error image region. Tiling spreads insertions across the
image, which helps submap quality and reduces scene-specific behavior.

Budgeting:

```yaml
PIPE:
  gaussian_submaps:
    gaussian_budget_policy: sdf_block_density
```

The `sdf_block_density` policy computes a cap roughly from:

```text
cap = clamp(base + density * sdf_blocks, min, max)
```

At budget pressure, pruning is prioritized. At hard cap, new insertion is
skipped and logged. TSDF tracking must not be blocked by Gaussian budget logic.

## 16. Rendering

There are two major render paths.

### Unified/original rendering

`SLAMPipeline::renderEvalImgs(...)` and `RawGaussianModel::forward(...)` render
with a single Gaussian model and a TSDF raycast.

In BasicEngine/unified mode:

1. Raycast global TSDF.
2. Render one global Gaussian model.
3. Save RGB/depth/GT images for metrics.

### Gaussian-submap rendering

`SLAMPipeline::renderEvalImgsSubmaps(...)` renders the hybrid submap path.

The key function is:

```cpp
TensorDict SLAMPipeline::renderLocalGaussianSubmaps(...)
```

Submap render process:

1. Compute or infer owner SDF submap for the camera.
2. Select SDF submaps for the base raycast.
3. Compose a global SDF base color/depth from selected SDF submaps.
4. Select Gaussian submaps for the camera.
5. For each selected Gaussian submap:
   - convert global camera pose into the submap local frame;
   - optionally raycast local SDF reference;
   - render that local Gaussian model;
   - collect color/depth/alpha/weight contributions.
6. Composite selected Gaussian contributions over the SDF base once.
7. Save render outputs and diagnostics.

Render selection policies:

| Policy | Meaning |
| --- | --- |
| `temporal` | Historical owner/neighbor selection by frame/submap order. |
| `sdf_bounds` | Project transformed SDF bounds into the camera and rank visible submaps. |
| `owner_only` | Render only owner submap. Useful for debugging. |
| `owner_neighbors` / variants | Render owner plus nearby sequential submaps. |

Overlap compositors include:

- owner-priority with neighbor hole filling;
- nearest-depth;
- weight-normalized;
- handoff-aware fill;
- SDF-guided adaptive fill;
- overlap depth-gated variants.

The stable design goal is:

```text
Use one SDF base, then add local Gaussian residual/detail from selected submaps.
Do not concatenate local Gaussian tensors into a fake global model for normal rendering.
```

The merged Gaussian export path exists for saving/inspection, but normal eval
rendering should use local submaps and transforms.

## 17. Pose Graph and Loop Closure

Loop closure lives mostly in `slam/slam_pipeline.cpp` and uses
`InfiniTAM/MiniSlamGraphLib`.

Important structs:

- `HybridPoseGraphNode`
- `HybridPoseGraphConstraint`
- `PendingPoseGraphTransformUpdate`
- `PoseGraphOrbFeaturePacket`
- `PoseGraphBowHistogramPacket`

Important functions:

- `buildLivePoseGraphScaffold(...)`
- `collectPoseGraphLoopCandidates(...)`
- `runPoseGraphDryRun(...)`
- `runPoseGraphSolveOnly(...)`
- `runPoseGraphApply(...)`
- `runLivePoseGraphUpdate(...)`
- `maybeRunOnlinePoseGraphUpdate(...)`
- `applySubmapGlobalToLocalTransform(...)`
- `applyPoseGraphTransformUpdate(...)`
- `saveCorrectedPoses(...)`
- `correctedGlobalPoseForCamera(...)`

### What the pose graph optimizes

Each graph node is a submap pose.

For hybrid runs:

```text
node = one SDF submap plus its matching Gaussian submap
state = local_to_global transform
```

For SDF-only MultiEngine runs:

```text
node = one SDF submap
state = local_to_global transform
```

Edges:

- Sequential edges come from accepted transitions between neighboring submaps.
- Loop edges come from verified non-neighbor revisits.

### Loop candidate generation

Loop candidate detection is CPU-side and bounded.

Evidence sources:

- ORB keypoints/descriptors.
- Depth-backed local 3D points.
- BoW-like ORB word histograms.
- Pose/submap separation rules.
- Stored loop observations per submap.

Candidate rules:

- Ignore same/recent/sequential-neighbor submaps unless explicitly configured.
- Cap checked candidate pairs per update.
- Cap accepted loop candidates per update.
- Prefer high-support, geometrically plausible candidates.

Verification can include:

- BF-Hamming ORB matching.
- Ratio/mutual matching.
- Spatial quadrant spread check.
- Depth-backed 3D-3D RANSAC.
- Optional PnP fallback.
- ICP/RANSAC refinement.
- Drift-aware delta gates.
- Multi-observation support.
- Pairwise consistency / PCM-style loop selection.

The current robust loop profiles keep this work on CPU so GPU time remains for
tracking, fusion, raycasting, and Gaussian rendering/optimization.

### Graph solve and apply

The solver uses `MiniSlamGraphLib`:

- `GraphNodeSE3`
- `GraphEdgeSE3`
- `PoseGraph`
- `LevenbergMarquardtMethod`

Loop edges can use:

- scalar information;
- normalized diagonal information;
- confidence diagonal information;
- robust kernels such as Cauchy.

Apply modes:

| Mode | Meaning |
| --- | --- |
| `disabled` | No correction. |
| `sequential_noop` | Safety check that should leave transforms unchanged. |
| `synthetic` | Applies/restores a known transform for testing. |
| `loop_candidates` | Applies accepted loop-candidate corrections. |

Actual apply rule:

```text
Only update submap transforms.
Do not rewrite TSDF voxels.
Do not globally transform Gaussian tensors.
Increment transform_epoch.
Invalidate/update render and pose export state.
```

Corrected past pose:

```text
corrected_c2w = current_owner_submap.local_to_global * stored_camera_local_pose
```

Raw trajectory goes to:

```text
val/pose/
```

Corrected trajectory goes to:

```text
val/corrected_pose/
```

### Online fairness rule

For thesis/paper claims, a loop result is fair only if:

- candidate detection happened during the run;
- verification happened during the run;
- graph solve happened during the run;
- transform apply happened during the run;
- no post-run TSDF reintegration was used for the main result;
- no post-run Gaussian retraining was used for the main result.

Post-run export/render from already-corrected submaps is allowed as an export
step, not as an extra optimization stage.

## 18. Configuration Surface

The config root has:

```yaml
workspace_dir: ...
dev_id: 0
work_mode: train
READER: ...
PIPE: ...
MODEL: ...
```

### `READER`

Controls dataset layout, image/depth formats, intrinsics, frame range, and
preload behavior.

Important keys:

- `input_dir`
- `image_path`, `depth_path`, `pose_path`
- `image_prefix`, `image_extension`
- `depth_prefix`, `depth_extension`
- `pose_prefix`, `pose_extension`
- `preload_images`
- `depth_scale`
- `intrinsics`
- `image_shape`
- `downscale_factor`
- `start_frame`, `end_frame`, `frame_step`
- `test_split_interval`

### `PIPE`

Controls training/eval behavior and all SLAM extensions.

Important top-level keys:

- `train_mode`
- `max_iterations`
- `enable_densify`
- `eval_after_train`
- `eval_sdf_only`
- `eval_split`
- `eval_stride`
- `eval_max_frames`
- `eval_start_offset`
- `save_after_train`
- `random_seed`
- `bypass_libtorch_atexit`
- `new_gs_sample_ratio`
- `local_opt_iters`
- `local_opt_interval`
- `keyframe_select_max`
- `log_slam_state`
- `submap_stats_interval_frames`
- `weight_configs`
- `vis_configs`
- `TSDF`
- `gaussian_submaps`
- `pose_graph`

### `PIPE.TSDF`

Controls InfiniTAM:

- `engine`: `basic` or `multi`
- `input_mode`: `preload` or `on_demand`
- `input_prefetch`
- `input_tensor_bridge`: `direct` or `opencv_legacy`
- `input_tensor_bridge_validate`: exact diagnostic A/B, normally `false`
- `tracking_failure_mode`
- `reject_nonfinite_tracking_pose`
- `voxel_size`
- `trunc_dist`
- `viewFrustum_min`, `viewFrustum_max`
- `use_gt_pose`
- `require_raw_tracking_success`
- `raw_tracking_failure_grace_frames`
- `saved_mesh`
- `saved_engine`
- `saved_images`
- `multi`

### `PIPE.TSDF.multi`

Controls MultiEngine:

- submap lifecycle;
- tracking-boundary handoff;
- occupancy/capacity boundaries;
- sequential transform policy;
- paging/residency;
- render submap selection;
- diagnostics;
- hash profile compatibility.

See `docs/hybrid_submap_config_guide.md` for the full current list.

Failure containment is intentionally separate from raw diagnostics. The
recommended `contain_failed` mode records the raw result as `FAILED`, restores
the previous finite pose, returns policy `POOR` so the outer stream continues,
and skips TSDF fusion. `tracking.csv` records both raw/policy results plus
`containment_pose_rollback` and `containment_fusion_suppressed`. The absent-key
default remains `legacy_ignore` for reproducible comparisons.

### `PIPE.gaussian_submaps`

Controls local Gaussian submaps:

- enable/ownership/coordinate frame;
- candidate warming;
- handoff;
- graduation/readiness;
- insertion policy;
- budgets;
- frozen refinement;
- cross-boundary observations;
- render selection/compositing;
- Gaussian residency;
- save/load behavior.

See `docs/hybrid_submap_config_guide.md` for the full current list.

### `PIPE.pose_graph`

Controls loop closure:

- candidate detection;
- ORB/BoW cache;
- candidate scheduling;
- verification gates;
- graph solve;
- robust edge weights;
- apply mode;
- online updates;
- health gates;
- observation gates;
- deferred nonresident map handling.

Runner profiles in `run/tum_loop_validation.py` are usually safer than hand
editing every pose-graph key.

### `MODEL`

Controls Gaussian model:

- SH degree and schedule;
- learning rates;
- initial scale/opacities;
- render method;
- densify/prune thresholds;
- GES residual/base weights;
- exposure optimization.

`SLAMGaussianModel::loadConfig` reuses `RawGaussianModel::loadConfig`, so most
model keys affect both unified and submap Gaussian paths.

## 19. Runners

Key docs:

- `run/README.md`
- `scripts/README.md`
- `docs/experiment_outputs_guide.md`

### Original runners

| Script | Purpose |
| --- | --- |
| `run/train.py` | Run all configs in a directory. |
| `run/eval.py` | Batch metric/eval helper. |
| `run/read_results.py` | Parse result logs into CSV. |
| `run/run.sh` | Original train/eval/read wrapper. |

### Research runners

| Script | Purpose |
| --- | --- |
| `run/hybrid_validation.py` | Main Replica quality and hybrid submap runner. |
| `run/tum_loop_validation.py` | Main TUM ATE/RPE and loop-closure runner. |
| `run/render_policy_sweep.py` | Eval-only render selection/compositor sweeps. |
| `run/hybrid_training_sweep.py` | Smaller Gaussian training/handoff sweeps. |

### Cleanup runners

| Script | Purpose |
| --- | --- |
| `run/cleanup_experiments.py` | Archive/prune old local experiment folders. |
| `run/prune_run_artifacts.py` | Archive/prune heavy external-drive run artifacts. |

For current claims:

- Use `hybrid_validation.py` for Replica quality.
- Use `tum_loop_validation.py` for TUM ATE/RPE.
- Use summary CSVs, not raw console logs.
- Verify `input_dir`, `gt_dir`, binary path, binary hash profile, and runtime
  TSDF hash profile before comparing rows.

## 20. Datasets

### Replica

Best for:

- PSNR
- SSIM
- LPIPS
- geometry accuracy/completion/F1
- controlled quality comparisons

Scripts:

- `download_replica.sh`
- `scripts/replica_preprocess.py`
- `scripts/replica_prepare_scene.py`

Typical prepared layout:

```text
data/replica/office0/
data/replica/office0_half_clean/
data/cull_replica_mesh/office0.ply
```

Replica full resolution is `1200x680`. Half resolution is `600x340`.

Half resolution is suitable for fast controlled ablations, but it does not
predict native-resolution fine detail. A half-trained room0 map dropped from
33.636 PSNR / 0.95833 SSIM / 0.06081 LPIPS at half resolution to
30.306 / 0.90696 / 0.19214 when rendered against native-resolution GT. Use
`hybrid_validation.py --quality-native-full` to record this cross-resolution
check, and use native full-resolution training for final image-quality claims.

### TUM RGB-D

Best for:

- ATE
- RPE
- loop-closure behavior
- fair comparison to GPS-SLAM paper TUM ATE scenes

Script:

- `scripts/tum_preprocess.py`

Paper ATE scenes:

- `rgbd_dataset_freiburg1_desk`
- `rgbd_dataset_freiburg2_xyz`
- `rgbd_dataset_freiburg3_long_office_household`

Extra loop-heavy validation scenes:

- `rgbd_dataset_freiburg1_room`
- `rgbd_dataset_freiburg2_pioneer_slam`
- `rgbd_dataset_freiburg2_pioneer_slam2`

TUM preprocessing fairness:

- Preserve raw PNG RGB/depth.
- Use depth scale `5000`.
- Keep full `640x480`.
- No denoising/cropping/resizing/color correction/depth filtering.
- Associate RGB/depth by timestamp.
- Interpolate GT pose at RGB timestamp.

## 21. Metrics

Core scripts:

| Script | Metric |
| --- | --- |
| `scripts/metric_general.py` | PSNR, SSIM, LPIPS. |
| `scripts/metric_sampled_eval.py` | Sampled PSNR/SSIM/LPIPS for long runs. |
| `scripts/ate_general.py` | ATE. Preserves legacy output while also supporting actual RMSE summaries. |
| `scripts/rpe_general.py` | Relative pose error. |
| `scripts/geo_general.py` | Mesh accuracy/completion/F1. |

Fair comparison rules:

- Replica quality should use GT-pose rendering when isolating reconstruction
  quality.
- TUM ATE/RPE should use standard preprocessed associations, not hand-edited
  pose/data filtering.
- Do not compare full-res and half-res quality rows as if they were identical.
- Do not compare different hash profiles as if they isolate only one variable.
- Do not cite old runs unless the generated config confirms the dataset path,
  binary, hash profile, and metric mode.
- Use `thor_quality_reconciled` as the L1-only Gaussian-submap baseline and
  `thor_quality_reconciled_perceptual` as the opt-in 0.2 D-SSIM quality mode.
  The latter improved PSNR/SSIM/LPIPS on all eight matched 320-frame Replica
  scenes but costs about 8.5% throughput.

## 22. Diagnostics and Output Files

Common run outputs:

| File/dir | Meaning |
| --- | --- |
| `val/pose/` | Raw live tracker trajectory. |
| `val/corrected_pose/` | Pose-graph corrected trajectory, if available. |
| `val/render/` | Rendered RGB evaluation images. |
| `val/gt/` | Eval GT images for sampled/eval metrics when generated. |
| `gs_model/` | Unified Gaussian model output. |
| `gaussian_submaps/` or submap save dirs | Local Gaussian submap outputs. |
| `tsdf_engine/` | Saved TSDF engine checkpoint. |
| `tsdf_mesh.ply` | Exported TSDF mesh. |
| `time_log.txt` | Legacy timing/FPS output. |
| `timing.csv` | Structured timing by stage, when enabled. |
| `tracking.csv` | Tracking result and raw/policy frontend state. |
| `submap_events.jsonl` | Structured submap and pose-graph events. |
| `frame_assignments.csv` | Frame to owner/overlap/loop-observed submap assignments. |
| `submap_stats.csv` | Per-frame/per-stage submap stats. |
| `paging.csv` | TSDF paging events where configured. |
| `render_debug.csv` or summaries | Render selection/compositor diagnostics. |
| `pose_graph_updates.jsonl` | Pose graph update summaries. |
| `loop_candidates*.json` | Candidate loop evidence and verification output. |

Useful diagnostic scripts:

- `scripts/analyze_hybrid_submap_run.py`
- `scripts/compare_hybrid_submap_runs.py`
- `scripts/analyze_submap_handoff.py`
- `scripts/summarize_submap_transitions.py`
- `scripts/rescore_saved_trajectories.py`
- `scripts/summarize_tum_runs.py`
- `scripts/compare_tum_run_stability.py`
- `scripts/tum_loop_gt_audit.py`
- `scripts/thor_preflight.py`

`PIPE.submap_stats_interval_frames` controls normal `submap_stats.csv`
sampling. `1` preserves per-frame logging, `10` is the embedded profiling
default used by the runners, and `0` records only the mandatory final state.
Submap events and transitions remain independently logged.

## 23. Remote Viewer

Key file:

- `remote_viewer.cpp`

The viewer path:

1. Loads the same YAML style as `slam_trainer`.
2. Loads dataset and TSDF engine.
3. Loads saved Gaussian model from `workspace_dir/gs_model/model.pt`.
4. Loads saved TSDF engine.
5. Opens a Boost.Asio TCP server on `config["port"]`.
6. Receives camera pose/FOV/resolution JSON from SIBR viewer.
7. Raycasts TSDF and renders Gaussian output.
8. Sends rendered RGB, input RGB placeholder, raycast RGB, raycast depth, pose,
   info string, and MVP matrix.

Security note:

The viewer has no authentication. Use firewall/private network controls if it
is run on a shared network.

Research caveat:

The viewer path primarily follows the unified model flow. Before relying on it
for local Gaussian submap visualization, verify that the submap save/load/render
path is wired for the desired config.

## 24. Thor and Embedded Readiness

Thor differs from the desktop in three important ways:

1. It is aarch64, so desktop x86 binaries cannot be reused.
2. Its GPU architecture needs CUDA arch `110`.
3. CPU and GPU share physical memory, so preloaded images, TSDF pages, Gaussian
   tensors, and caches all compete for the same memory pool.

Thor-oriented rules:

- Use `PIPE.TSDF.input_mode: on_demand`.
- Enable `PIPE.TSDF.input_prefetch` and use
  `PIPE.TSDF.input_tensor_bridge: direct`.
- Keep `input_tensor_bridge_validate`, Gaussian insertion diagnostics, render
  overlap diagnostics, and Gaussian submap debug disabled during throughput
  measurement.
- Set `PIPE.submap_stats_interval_frames: 10` initially; use `1` only when
  diagnosing individual frames.
- Use bounded MultiEngine residency.
- Prefer `medium_128k` first, then `compact_64k` for stress.
- Keep heavy output outside the repo.
- Run `scripts/thor_preflight.py` on configs.
- Start with 10/50/120-frame smoke tests.
- Monitor `tegrastats`.
- Do not start with full-resolution 2000-frame Replica Gaussian runs.
- Keep CPU tasks on CPU: ORB, loop scheduling, pose graph, paging coordination.
- Keep CUDA for tracking, fusion, raycasting, Gaussian optimization/rendering.
- Do not infer Gaussian-raster or TSDF speed from Thor's advertised FP4 AI
  throughput. Profile these CUDA paths directly at the intended power mode.
- Keep the 8-submap/280k-Gaussian render ceiling initially. A room0 2,000-frame
  test found no quality gain from doubling it, while Gaussian render time rose
  33%.
- Treat higher full-resolution insertion budgets as a quality mode. A tested
  2x budget gained 0.171 dB but added 31% Gaussians and lost another 8% FPS.

The completed desktop hot-path audit and the exact native acceptance sequence
are in `docs/thor_readiness_evaluation_20260817.md`. Its desktop measurements
are readiness evidence, not predicted Thor FPS.

USB transfer docs:

- `THOR_SETUP_GUIDE.md`
- `DATASET_SETUP_GUIDE.md`
- `TRANSFER_MANIFEST.md`

These live on the USB transfer folder, not necessarily in this repo.

## 25. Common Development Tasks

### Find where a YAML key is read

Use:

```bash
rg -n "key_name" .
```

Common readers:

- `src/dataset_reader.cpp`
- `src/pipeline.cpp`
- `slam/slam_pipeline.cpp`
- `slam/InfiniTAM_tools.cpp`
- `run/hybrid_validation.py`
- `run/tum_loop_validation.py`

### Add a new Gaussian submap toggle

1. Add a member variable in `slam/slam_pipeline.h`.
2. Parse and validate it in `SLAMPipeline::loadConfig`.
3. Add it to generated runner profiles if needed.
4. Add it to `docs/hybrid_submap_config_guide.md`.
5. Add diagnostics if it affects behavior.
6. Test with off/default mode and enabled mode.

### Add a new TSDF/MultiEngine toggle

1. Add config parsing in `slam/InfiniTAM_tools.cpp` or the relevant InfiniTAM
   class.
2. Add validation to `validateTsdfConfig`.
3. Thread it into `ITMMultiEngine::configureHybridSubmaps(...)` or
   `ITMActiveMapManager::configureSequentialPolicy(...)`.
4. Add structured diagnostics.
5. Make sure BasicEngine ignores it safely.
6. Make sure runtime YAML `hash_profile` still matches the binary profile.

### Add a new loop-closure verification gate

1. Add config fields in `SLAMPipeline`.
2. Parse them in `loadConfig`.
3. Add defaults to `run/tum_loop_validation.py` profiles only if validated.
4. Log reject reasons in loop candidate diagnostics.
5. Test dry-run, solve-only, and apply modes.
6. Test negative controls: identity, synthetic, bad loop, missing depth.

### Debug bad ATE

Check:

1. `tracking.csv`
2. `frame_assignments.csv`
3. `submap_events.jsonl`
4. `pose_graph_updates.jsonl`
5. `val/pose` ATE
6. `val/corrected_pose` ATE
7. raw tracking failure/non-good ratios
8. number and reason of submap handoffs
9. loop candidates accepted/rejected
10. graph residual before/after solve

Useful scripts:

```bash
python scripts/summarize_tum_runs.py --run-root <root>
python scripts/summarize_submap_transitions.py <run_dir>
python scripts/rescore_saved_trajectories.py --gt-path <gt_camera_dir> --run-dir <run_dir>
```

### Debug bad quality

Check:

1. whether quality run used GT poses;
2. whether image sizes match metric GT path;
3. `metric_mode` in summary;
4. Basic SDF-only vs Basic unified Gaussian;
5. Multi SDF-only vs hybrid Gaussian;
6. number of submaps and Gaussian counts;
7. render-selected submap counts;
8. selected Gaussian counts and budget caps;
9. novelty suppression ratios;
10. boundary-frame PSNR/SSIM/LPIPS;
11. readiness deltas;
12. render overlap depth-gate diagnostics.

Useful scripts:

```bash
python run/hybrid_validation.py --help
python scripts/analyze_hybrid_submap_run.py <run_dir>
python scripts/compare_hybrid_submap_runs.py <old_run> <new_run>
python run/render_policy_sweep.py --help
```

### Debug memory or crashes near long-run frame counts

Check:

1. `PIPE.TSDF.input_mode`
2. `READER.preload_images`
3. `PIPE.save_after_train`
4. TSDF hash profile
5. `resident_gpu_maps`
6. Gaussian submap residency
7. `tsdf_pages` growth
8. output disk space
9. `timing.csv` P95/P99 spikes
10. `paging.csv`

Use:

```bash
python scripts/thor_preflight.py <config.yaml>
df -h
du -sh output/* | sort -h
```

## 26. Important Invariants

These are the assumptions most likely to cause subtle bugs if broken.

### Pose/frame invariants

```text
Camera::c2w is normalized GT pose.
Camera::c2w_slam is raw estimated global pose unless local Gaussian training temporarily replaces it.
Camera::c2w_local is owner-submap local pose.
frame_local_pose stores CPU local poses for corrected trajectory export.
```

### SDF/Gaussian submap invariants

```text
Each accepted SDF submap has at most one non-retired Gaussian submap.
Local Gaussian tensors stay in owner SDF local frame.
Loop closure changes transforms only.
Transform updates must increment transform_epoch.
Render caches/selection must not use stale transforms after correction.
```

### Residency invariants

```text
Primary map must stay resident.
Transition/candidate maps should stay resident.
Dirty maps cannot be evicted before snapshot completion.
Observation-gate endpoint maps may be temporarily pinned.
Gaussian tensors moved off GPU cannot be trained/rendered until made resident.
```

### Evaluation invariants

```text
Compare only matching datasets, frame ranges, image scales, and pose modes.
Replica quality claims should use GT-pose eval when isolating quality.
TUM ATE/RPE claims should use fair raw TUM associations.
Paper-style BasicEngine baselines should use legacy_large build/profile.
MultiEngine runs must report binary hash profile and runtime hash profile.
```

## 27. Known Limitations and Risk Areas

Current known risks:

- The codebase is a research working tree with many opt-in knobs. Use runner
  profiles instead of hand-combining unfamiliar options.
- BasicEngine should remain baseline-safe, but shared code changes still need
  regression checks on paper scenes.
- MultiEngine lifecycle tuning can affect ATE, especially on short/easy TUM
  scenes where unnecessary handoffs are harmful.
- Loop closure can improve global ATE while slightly worsening local RPE if a
  correction introduces boundary discontinuities.
- Hybrid Gaussian quality is sensitive to render selection, overlap handling,
  candidate warming, and submap count.
- Some frozen-refinement/top-up policies are diagnostic ablations, not stable
  defaults.
- Full-resolution Replica on an 8 GB desktop GPU is not representative of Thor
  and can OOM or destabilize the session.
- Thor still needs native dependency/build validation.
- Newer JetPack-compatible LibTorch APIs and ABI must still be validated by a
  native Thor build; the desktop used Torch 2.3.1 while Thor requires its
  NVIDIA-supported AArch64/CUDA pairing.

## 28. Documentation Map

Use this document first for orientation, then go deeper:

| Document | Use |
| --- | --- |
| `README.md` | Original project overview plus research branch notes. |
| `docs/README.md` | Documentation index. |
| `docs/hybrid_submap_config_guide.md` | Detailed config toggle guide. |
| `docs/hybrid_submap_development_log.md` | Chronological development and experiment log. |
| `docs/loop_closure_pose_graph_current_state.md` | Deep explanation of loop closure/pose graph implementation. |
| `docs/tum_paper_ate_comparison_20260814.md` | TUM paper-style ATE comparison notes. |
| `docs/thor_readiness_accuracy_quality_audit_20260816.md` | Thor readiness audit and risks. |
| `docs/experiment_outputs_guide.md` | Where outputs and trusted summaries live. |
| `run/README.md` | Runner script roles. |
| `scripts/README.md` | Metric, dataset, and diagnostic script roles. |

## 29. Recommended Reading Order for a New Engineer

1. `README.md`
2. This file.
3. `slam_trainer.cpp`
4. `include/dataset_reader.h` and `src/dataset_reader.cpp`
5. `slam/InfiniTAM_tools.h/.cpp`
6. `slam/slam_pipeline.h`
7. `SLAMPipeline::SLAMTrainCams` in `slam/slam_pipeline.cpp`
8. `RawGaussianModel` and `SLAMGaussianModel`
9. `ITMMultiEngine.tpp`
10. `ITMActiveMapManager.cpp`
11. `docs/hybrid_submap_config_guide.md`
12. `run/hybrid_validation.py`
13. `run/tum_loop_validation.py`
14. `docs/loop_closure_pose_graph_current_state.md`

## 30. Quick Glossary

| Term | Meaning |
| --- | --- |
| ATE | Absolute Trajectory Error. Global pose accuracy metric. |
| RPE | Relative Pose Error. Local motion consistency metric. |
| SDF/TSDF | Signed/truncated signed distance field, used for geometry and base color. |
| Gaussian | 3D Gaussian primitive with position, scale, rotation, opacity, SH color. |
| GES | Gaussian-Enhanced SDF render path. |
| BasicEngine | Original single global TSDF engine. |
| MultiEngine | InfiniTAM multi-local-map engine. |
| SDF submap | One local TSDF map. |
| Gaussian submap | One local Gaussian model tied to one SDF submap. |
| Primary submap | Current SDF submap used for live tracking/fusion. |
| Candidate submap | New SDF submap being warmed/evaluated during handoff. |
| Handoff | Transition period between outgoing and incoming submaps. |
| Graduation | Final bounded optimization/readiness step before freezing a Gaussian submap. |
| Frozen submap | Submap no longer normally optimized online. |
| Residency | Whether a TSDF/Gaussian submap is on GPU, CPU, or disk. |
| Loop candidate | Possible non-neighbor revisit constraint. |
| Pose graph | Graph whose nodes are submap poses and edges are sequential/loop constraints. |
| Corrected pose | Past frame pose recomputed from current submap transform and stored local camera pose. |

## 31. Practical Rule of Thumb

When modifying this codebase, identify which layer owns the behavior:

- Dataset/file naming belongs in `DatasetReader` or preprocessing scripts.
- Per-frame online SLAM behavior belongs in `SLAMPipeline`.
- Raw Gaussian math belongs in `RawGaussianModel` or rasterizer kernels.
- Online Gaussian insertion belongs in `SLAMGaussianModel` and
  `SLAMPipeline::initNewGaussians`.
- SDF tracking/fusion/raycasting belongs in InfiniTAM and `TsdfEngineFacade`.
- SDF submap lifecycle belongs in `ITMActiveMapManager` and `ITMMultiEngine`.
- Loop candidate scheduling, verification, graph solve, and transform apply
  belong in `SLAMPipeline`.
- Experiment generation belongs in `run/`.
- Metrics and offline diagnostics belong in `scripts/`.

Do not fix a metric result by changing the metric script unless the metric
script is demonstrably wrong. Prefer preserving metrics and changing only the
runtime path being evaluated.
