# AGX Thor Porting and Thesis Helper

**Status:** paper comparison freeze `thor-paper-20260819-r2`, verified 25 August 2026

**Purpose:** thesis/paper methods, implementation, reproducibility, and limitation record

**Results boundary:** the 61-run frozen parent-system matrix is complete. Use [PAPER_READINESS_LOCK_2026-08-25.md](../PAPER_READINESS_LOCK_2026-08-25.md) for final R2 aggregates and claim boundaries; same-platform external-system results remain pending.

## 1. Scope and Provenance

This document separates three bodies of work that must not be conflated in the thesis:

1. **Original GPS-SLAM:** the Gaussian-Plus-SDF system by Peng, Zhou, and Shao.
2. **Research modifications made before the Thor transfer:** local SDF/Gaussian submaps, loop detection, geometric verification, sparse pose-graph correction, tracking containment, and bounded map residency. These are the proposed research contribution, not consequences of AArch64 or Docker.
3. **Thor port and paper hardening:** native AArch64/SM 11.0 build support, framework isolation, dependency/ABI fixes, embedded memory handling, fail-closed evaluation, real-time instrumentation, locked protocols, and the comparison freeze.

The received state is recoverable from the private `GPS-SLAM_Thor_Source_2026-08-18.tar.gz` archive and described by its private `SOURCE_FILES.sha256`. A content-hash comparison against that archive identifies **40 received project files changed after arrival on Thor**. The current tree also adds **21 functional container, protocol, analysis, and regression files**. A further **87 files under `ThirdLibs/install/` are generated AArch64 dependency-install artifacts**, not hand-written algorithm changes. The final comparison envelope is recorded in [BENCHMARK_FREEZE.json](../BENCHMARK_FREEZE.json).

This is a stronger provenance boundary than the small bundled Git history, because the transfer snapshot contained substantial uncommitted research work. Widespread executable-bit changes are packaging metadata and do not change program semantics.

## 2. Thesis-Ready Contribution Summary

The project extends GPS-SLAM from a single global TSDF/Gaussian representation to a collection of rigid local SDF and Gaussian submaps. RGB-D tracking and fusion remain local and online. Loop candidates are ranked using bounded CPU-side ORB descriptors, verified geometrically using depth-backed rigid alignment, filtered with consistency and safety gates, and inserted into a sparse SE(3) pose graph. Accepted corrections update submap transforms rather than reintegrating frames, refitting Gaussians offline, or deforming dense map contents. This preserves the causal online reconstruction model while enabling global drift correction.

The Thor port preserves that architecture and produces native AArch64 machine code plus native CUDA SM 11.0 kernels. The NVIDIA container supplies a compatible CUDA-enabled LibTorch and reproducible user-space dependency set; it is **not x86 emulation and not a virtualized GPU**. Docker should therefore be described as environment isolation, not an alternative processor architecture.

## 3. System State Before Target-Side Bring-Up

The transfer already contained the following research functionality:

- InfiniTAM BasicEngine and MultiEngine TSDF tracking/fusion.
- Local SDF submaps and locally owned Gaussian submaps.
- Shared local-to-global submap transforms for SDF and Gaussian rendering.
- Candidate observation capture, CPU ORB retrieval, depth-backed ORB/ICP verification, loop pooling, PCM-style consistency selection, sparse pose-graph solve/apply, and corrected-pose export.
- Bounded GPU residency with CPU/disk page backing, asynchronous page-out/page-in, lifecycle handoff, and diagnostic traces.
- Legacy, containment, and `robust_rpe` tracking profiles; legacy behavior remained selectable.
- Paper-parity and embedded-stress TSDF capacity profiles.

Several Thor-oriented optimizations were also completed on the desktop before transfer:

- A **direct Torch-to-InfiniTAM RGB-D bridge** removed intermediate Torch/OpenCV allocation, RGB/BGR conversion, and duplicate copies. An 80-frame 1200×680 validator compared 65,280,000 RGB-D pixel tuples with zero mismatch.
- A **persistent prefetch worker** replaced one `std::async` creation per frame. In the matched desktop trace, `pthread_create` calls fell from 659 to 35, fusion mean fell from 16.98 to 14.51 ms, and pipeline throughput rose from 28.63 to 30.35 FPS. These are desktop optimization results, not Thor headline results.
- Sampled `submap_stats.csv` output and gated diagnostic reductions reduced logging/synchronization overhead without changing selection decisions.
- On-demand input, CPU-backed bounded residency, bridge validation, debug controls, and preflight options were exposed through both runners.

These changes belong in the implementation chapter, but the chronology must state that they were validated before the physical Thor bring-up.

## 4. Why the Container Was Selected

The desktop application depended on x86-64 LibTorch 2.3.1/CUDA 11.8 artifacts. Thor requires AArch64 libraries and SM 11.0 kernels; copied desktop binaries cannot execute or link. At bring-up, the host JetPack 7.1 installation provided the driver/toolkit but did not provide the complete compatible Torch C++ development environment required by this project.

The selected image is pinned as:

```text
nvcr.io/nvidia/pytorch:26.07-py3
index digest sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
```

It supplies AArch64 Torch 2.13.0a0 nv26.07 and CUDA 13.3 development files. The host provides the NVIDIA kernel driver and GPU device. Source and outputs reside on NVMe/ext4 bind mounts. Consequently:

- CUDA computation executes directly on the Thor GPU.
- C++ executes directly on the Thor Arm CPU.
- there is no instruction-set translation;
- container startup and namespace/filesystem handling may add small overhead, but the dominant SLAM kernels are not emulated;
- all paper runs must use the same pinned image so any container effect is common to both arms.

A native host build remains technically possible if a matching CUDA-enabled AArch64 LibTorch/CMake package and all compatible libraries are installed. It would not inherently be faster enough to justify the present reproducibility and ABI risk. The container was therefore an engineering choice, not an algorithmic necessity imposed by the GPU architecture.

### 4.1 Observed Hardware and Software Envelope

| Item | Observed on `apthor` |
|---|---|
| Processor/platform | Jetson AGX Thor, AArch64, 14 online CPU cores |
| OS/BSP | Ubuntu 24.04-based Jetson Linux r38.4; kernel `6.8.12-tegra` |
| Physical/shared memory | 122.8 GiB visible, no swap during bring-up |
| Storage | Internal NVMe, ext4; approximately 835 GiB free at initial inspection |
| Host CUDA compiler | CUDA 13.2.51; `compute_110` advertised |
| Host driver compatibility | Driver 580.00, CUDA 13.0 capability reported to the container |
| Container framework | NVIDIA release 26.07; Torch `2.13.0a0+9186a08b2c.nv26.07`; CUDA 13.3 |
| GPU seen by Torch | `NVIDIA Thor`, capability `(11, 0)` |
| Maximum-clock ceiling used for controlled tests | MAXN; CPU 2.601 GHz, GPU 1.575 GHz, EMC 4.266 GHz; fan PWM 255 |

The mismatch between host compiler, driver capability, and container toolkit versions is why both the system checker and a real Torch CUDA kernel are part of the run gate. The thesis should report observed versions, not infer them solely from the JetPack marketing matrix.

## 5. Target-Side Porting Changes and Rationale

### 5.1 CUDA and CMake

| Change | Reason | Experimental effect |
|---|---|---|
| CUDA is mandatory; the invalid CPU fallback was removed. | Core sources and rasterizer require CUDA, so the old fallback could configure but not produce a valid trainer. | Fails early instead of creating a misleading build. |
| `CMAKE_CUDA_ARCHITECTURES=110` is explicit and preserved through InfiniTAM. | InfiniTAM previously overwrote it with `70;75;80`; CUDA 13 removed offline compilation for Volta/SM 70. | Produces executable Thor kernels and blocks accidental wrong-architecture builds. |
| C++17 and CUDA 17 are required. | Matches Torch and project source requirements. | Build contract only. |
| Modern targets (`CUDA::cudart`, `yaml-cpp::yaml-cpp`) replace raw `cuda`, library-directory, and archive names. | Raw names were toolchain- and architecture-dependent. | Linkage correction only. |
| `TORCH_CXX_FLAGS` is applied to all relevant targets. | Prevents C++ ABI mismatches with the selected Torch build. | ABI correctness. |
| OpenCV explicitly includes `imgcodecs`. | RGB/depth loaders use image codec symbols. | Fixes incomplete linkage. |
| CUDA fast math is an explicit option and is **OFF** in the freeze. | The legacy flag path was obsolete/inert in places, and fast math can alter numeric results. | May differ numerically from older desktop binaries; all comparisons are therefore rerun from the frozen build. |
| The trainer and optional viewer share `gps_slam_core`. | Removes duplicate source compilation and inconsistent target settings. | Build efficiency; no runtime algorithm change. |
| Pangolin, OpenGL/GLUT, Assimp, and FreeType are removed from the headless default. | Active trainer code does not use them. | Smaller dependency surface; no mapping change. |
| Legacy UI and TCP viewer are opt-in. | The embedded benchmark is headless. | Viewer functionality is absent unless explicitly built; trainer behavior is unchanged. |
| `cudaThreadSynchronize()` became `cudaDeviceSynchronize()` in the optional UI. | CUDA 13 removed the legacy symbol. | Compatibility only; UI is excluded from paper binaries. |

### 5.2 Third-Party Build

`build_third_libs.sh` now builds outside vendored sources, selects Ninja when available, enables position-independent code, fixes the install library directory, validates parallel-job values, and writes an architecture/compiler completion marker. The Thor build consumes only indicators, tensorboard_logger source, tinyply, and yaml-cpp; unused desktop GUI/import dependencies are not built. Tinyply examples are disabled.

`scripts/thor_build.sh` now discovers `/usr/local/cuda*/bin/nvcc`, validates `compute_110`, validates the live Torch CUDA device and C++ ABI before the expensive dependency build, checks the dependency marker, permits a Makefiles fallback, selects the TSDF hash profile explicitly, and runs post-build ELF, linkage, CMake-cache, compile-flag, and `cuobjdump` checks.

### 5.3 Framework and ABI Failures Resolved

The first container import exposed mismatched HPC-X libraries:

- `libtorch_cpu.so: undefined symbol: ompi_mpi_short_float`;
- `libucc.so.1: undefined symbol: ucs_config_doc_nop`.

Prepending only a UCX directory was insufficient because UCC, UCX, and OpenMPI must be version-matched. The image also lacked the expected `/opt/hpcx/hpcx-init.sh`. `containers/thor/hpcx_exec.sh` therefore constructs an ordered, validated HPC-X environment without relying on that missing script. The system check verifies that UCC resolves `/opt/hpcx/ucx/lib/libucs` and Torch resolves `/opt/hpcx/ompi5/lib/libmpi` before importing Torch.

The native compile then exposed a second ABI collision. The bundled TensorBoard logger generated protobuf classes against the system protobuf, while NVIDIA Torch placed private protobuf headers first. The resulting `tensorflow::Summary` class did not implement the private header's required `Message::New()` signature. The C++ TensorBoard writer is now optional and **OFF for Thor**, ensuring one process never mixes the two protobuf ABIs. Image output, CSV/JSON diagnostics, quality metrics, and checkpoints remain available; only TensorBoard event files are omitted.

The container also:

- pins `trimesh==4.12.2` by wheel hash;
- verifies offline VGG/LPIPS weights;
- sets NVIDIA's `NVIDIA_IMEX_CHANNELS=0` workaround for the affected NVSHMEM release;
- verifies a real CUDA tensor reduction, not merely `torch.cuda.is_available()`;
- accepts CUDA minor-version compatibility only after that kernel probe succeeds.

### 5.4 Runtime Safety and Embedded Memory

| Change | Purpose and behavior |
|---|---|
| Workspace validation in Python and C++ | Rejects empty paths, filesystem root, files, home or its parent, source/current/config overlap, and dataset overlap before the deliberately destructive workspace clear. A marker records managed ownership. |
| Output containment | Model, log, and evaluation paths must resolve below `workspace_dir`; path traversal and accidental absolute escape are rejected. |
| Dataset validation | Checks intrinsics length/finite values, image shape, positive focal lengths/depth scale/frame step, split validity, readable RGB/depth files, correct end-frame arithmetic, and non-empty training sets. |
| Pose validation | Rejects undefined, non-4×4, non-float32, and non-finite transforms at the Torch/InfiniTAM boundary. Runtime assertions that disappear in release builds were replaced with exceptions. |
| Frame-index check | TSDF and camera sequence indices must remain synchronized. |
| Estimated-pose memory | Ground-truth InfiniTAM matrices are no longer allocated or converted when `use_gt_pose=false`. |
| Reconstruction-frame release | In SDF-only `recon` mode, RGB/depth tensors are released after InfiniTAM consumes them. Bounded loop observations reload lazily from file paths. This reduces shared-memory retention without changing fused data. |
| GPU-memory fallback | Any unsupported/failed NVML query falls back to `cudaMemGetInfo`, which is usable on Thor's shared-memory path. |
| Scoped ownership | Gaussian Adam optimizers, scheduler, and optional TensorBoard logger use `std::unique_ptr`; scheduler destruction precedes optimizer destruction. Raw raycast buffers are initialized and destroyed. |
| Empty-model guards | Local optimization and redundant-Gaussian removal are skipped safely when no Gaussian exists. |
| Required artifacts fail closed | Required CSV/JSON streams throw on open, write, flush, or close failure. Atomic temporary-file/rename publication is used for key JSON summaries. |
| Resource output | `runtime_resources.json` records current/peak process RSS, available system memory, CUDA-visible shared-memory use, Gaussian count, schedule, and algorithmic wall time. |

These are correctness and robustness changes. Except for fixing erroneous input/range cases, releasing data already consumed, and disabling numerical fast math, they do not change valid-run mapping decisions.

## 6. Paper and Evaluation Hardening

### 6.1 Image, Pose, and Geometry Metrics

The original metric scripts could silently score only the intersection of available files, force CUDA imports even for unrelated metrics, or leave stale success JSON after a failure. The hardened tools now:

- parse unique frame identifiers and reject duplicate, missing, unexpected, or mismatched files;
- require complete primary coverage unless an explicitly labelled partial diagnostic is requested;
- validate image dimensions and finite pose/geometry arrays;
- resolve `auto`, CPU, or a specific CUDA device explicitly;
- import Torch/image dependencies lazily so a geometry or pose check does not depend on Torch;
- publish atomic success JSON or a structured error artifact and return non-zero on failure;
- use a shared strict pose loader for ATE and RPE;
- perform rigid SE(3) trajectory alignment without scale for RGB-D evaluation;
- report pose/frame coverage and the exact number of RPE pairs;
- use `trimesh` plus SciPy rather than unavailable AArch64 Open3D;
- deterministically sample mesh surfaces with a recorded seed and sample counts;
- reject empty/non-finite/degenerate geometry and report accuracy, completion, precision, recall, and F1 thresholds.

PSNR, SSIM, and VGG-LPIPS remain the image-quality metrics. Their weights are offline and hash-verified so a benchmark cannot download or silently substitute a model.

### 6.2 Locked Publication Protocols

Only the following protocols are paper-eligible under r2:

| Protocol | Question isolated | Locked essentials |
|---|---|---|
| `gps_slam_replica_paper_v1` | Does the proposed representation maintain final rendering quality? | 2,000 frames, 1200×680, frame step 1, GT poses, 5 mm voxels, train split, SH degree 3, L1-only loss, 20 immediate Gaussian iterations every 10 frames, no post-sequence insertion/refinement. |
| `gps_slam_tum_tracking_v1` | Do submaps and loop closure improve trajectory accuracy? | Native 640×480 RGB-D, full sequences, estimated poses, legacy tracker in both arms, no Gaussians, at least five same-seed repeats, raw and corrected ATE/RPE. |
| `gps_slam_tum_integrated_v1` | What is the throughput/resource cost of the complete proposed pipeline? | Full estimated-pose TSDF + Gaussian + loop path with immediate optimization and terminal-inclusive accounting. |

The faithful baseline uses `build_legacy/slam_trainer` with the original `legacy_large` BasicEngine capacity. The proposed method uses `build_thor/slam_trainer` with `medium_128k` MultiEngine capacity. Both were rebuilt from the same final instrumented source. The capacity difference is intentional and must be disclosed: the baseline represents faithful original single-map behavior, while the proposed build represents the bounded embedded deployment. It must not be described as a single-variable micro-ablation.

Replica GT poses deliberately isolate mapping/rendering quality from tracking. TUM deliberately disables Gaussians in the tracking protocol because the GPS-SLAM paper's TUM trajectory table evaluates tracking, not photorealistic mapping. The integrated TUM protocol separately measures the full system; its FPS must not be substituted with tracking-only FPS.

The parity target in the development records was defined against **GPS-SLAM arXiv v1, especially Table 3**. The default arXiv page now serves a revised v2. The thesis must cite the exact version used for numerical comparison (`2509.11574v1`) and separately note if prose or claims are taken from v2; silently mixing versions would weaken reproducibility.

### 6.3 Real-Time Measurement

`timing.csv` now records, per input frame:

- complete algorithmic service time;
- time until the estimated pose is available;
- online pose-graph latency;
- Gaussian scheduler mode, trigger, iterations executed, and remaining work;
- existing fusion, raycast, insertion, local optimization, handoff, and pose-graph components.

`scripts/analyze_realtime_trace.py` deterministically replays the measured FIFO service times against both original TUM timestamps and a normalized 30 Hz arrival stream. It emits `realtime_trace.csv` and `realtime_summary.json` with throughput, p50/p95/p99/max service and response latency, deadline misses, maximum/final backlog, recovery time, and frame continuity. The sustained gate permits individual deadline misses but requires every frame, at least 30 FPS, final backlog no greater than one 30 Hz period, maximum backlog no greater than 500 ms, and recovery within one second.

The `online_end_poses` trajectory is exported **before terminal candidate scanning**. This is the only trajectory eligible for a causal deployed real-time accuracy claim. A final candidate scan is end-of-sequence work that ranks/verifies revisit pairs after the last sensor arrival. It is fair to include in an offline/final trajectory if its latency is reported, but it cannot be used to claim the pose was available online at capture time.

For parity runs, terminal work is retained and reported separately as steady-state FPS, terminal latency, and terminal-inclusive pipeline FPS. Dataset preload, checkpoint save, metric rendering, and external evaluation remain outside the frame-loop runtime and are disclosed as such.

### 6.4 Thor Telemetry

`scripts/thor_telemetry.py` captures the child process and `tegrastats`, including process peak RSS, system RAM, VIN and component power, temperatures, CPU/GPU/EMC clocks, power mode, fan PWM/RPM where exposed, cooling states, and before/after over-current counters. The wrapper returns failure when the run is incomplete, the hardware envelope changes, or required telemetry is missing.

MAXN, locked clocks, and maximum fan define a controlled ceiling configuration, not a guaranteed sustainable deployment mode. NVIDIA documents that MAXN can trigger power capping and is not necessarily the best prolonged-workload mode. Runs with changed over-current or thermal counters are invalid. If repeated MAXN runs throttle, all compared arms must be rerun in the same highest stable supported mode (for example 120 W) rather than weakening hardware limits.

### 6.5 Runner and Configuration Failure Semantics

Both experiment runners validate user-derived path tokens, known labels/sequences, frame ranges, binary capacity, and locked protocol roles before constructing output paths. Protocol locking is applied before the final preflight and memory estimate so the checked configuration is the one actually executed. Optional timeouts become explicit `TIMEOUT` rows; trainer return codes become `OK` or `CRASH`; hard-gate failures make the runner return non-zero. Benchmark manifests are written for generated, completed, and failed states and include hashes of configs, datasets, binaries, and evaluation artifacts.

`scripts/thor_preflight.py` independently checks destructive workspace overlap, estimated TSDF residency/cache pressure, scheduler completion capacity, and contradictory terminal policies. In particular, `final_candidate_scan` and `terminal_online_scan` cannot both be presented as active: the former supersedes the latter, so the preflight rejects that misleading combination. `scripts/summarize_tum_runs.py` marks missing/invalid artifacts as acceptance blockers and reports terminal work without double-counting pose-graph updates already inside frame service time.

## 7. Low-Risk Optimization Added but Not Promoted

An opt-in Gaussian scheduler was implemented:

```yaml
PIPE:
  gaussian_schedule:
    mode: sliced
    iterations_per_frame: 2
```

It retains 20 optimization iterations per 10-frame interval while spreading them over ten frames. It snapshots the trigger camera, deterministic camera/raycast list, and sample order; initializes the optimizer once; performs cleanup only after iteration 20; and drains before submap vector growth, ownership changes, paging, handoff, retirement, graduation, frozen-map refinement, quality finalization, another optimizer use, or sequence end. Overlap or incomplete work is an error—work is never silently dropped. A pointer-invalidation defect found during the screen was fixed by draining before Gaussian-submap vector growth.

The accompanying `recommended_longspan_robust_streaming` profile scans every 60 frames, ranks eight pairs, verifies one, reserves one long-span pair, uses a 512-entry ORB cache, disables event bursts and terminal exhaustive scanning, and locks paging to eight resident maps, a 12 GiB CPU cache, three asynchronous page-outs/2 GiB, and two asynchronous page-ins/1 GiB.

This candidate reduced service-time tails and improved throughput in the one-pass screen, but failed the conjunctive 30 Hz backlog/recovery and online ATE gates on `fr2_xyz`/`fr3_office`. Therefore:

- `immediate` remains the default;
- the two `*_deployed_v1` protocols are marked unpromoted and require an explicit override;
- their results are not paper-eligible under r2;
- the present runtime claim is average throughput, not sustained or hard real time.

This negative result is valuable thesis evidence: an optimization was evaluated against predeclared latency, accuracy, quality, map-size, and hardware gates rather than retained because average FPS improved.

## 8. Reproducibility and Freeze

The frozen identity is:

| Item | r2 value |
|---|---|
| Functional files | 3,998 |
| Functional digest | `aa9b8a5ff9b1678d75f852465f6eb9ee5d77ea260708e7b6fae8d70fe9248785` |
| Freeze manifest SHA-256 | `5db75de5f471ca0da922d861ccd5270b9dc7f93a5fa016163b05331460ea68de` |
| Baseline binary | `fb7ec61b785b939d116305af095064f4f4427a9377b40c176a5efe6e9d075992` |
| Proposed binary | `e1114f0dd20feb6c779c41f95d8802405eedea92fdd5178e0d1c4198698df7f1` |
| Container image ID | `sha256:45cde99a435ccb44bd72857b757c5b2adeb7c9a283a5e013a4a3158a2afd71b0` |
| Recoverable r2 archive | `../GPS-SLAM_thor-paper-20260819-r2.tar.zst` |
| Archive SHA-256 | `e377620a523632ddf9f0f604bc92d406f33ab5162c87416812614d81420bcce0` |

Before every locked run:

```bash
bash scripts/thor_container.sh source-check
bash scripts/thor_container.sh regressions
```

The current freeze passes tracking 12/12, metrics 9/9, freeze 3/3, benchmark 6/6, real-time 4/4, promotion 3/3, validation hardening, and native pose-graph solver checks. Any functional edit, binary rebuild, build-profile change, or container replacement requires r3 and rerunning all affected comparison arms. Documentation and result summaries are outside the functional digest.

## 9. Exact Post-Transfer File Ledger

### Changed files received in the archive (40)

- **Build/toolchain (8):** `CMakeLists.txt`, `InfiniTAM/CMakeLists.txt`, `InfiniTAM/InputSource/CMakeLists.txt`, `InfiniTAM/cmake/SetCUDALibTarget.cmake`, `InfiniTAM/cmake/UseCUDA.cmake`, `ThirdLibs/tinyply/CMakeLists.txt`, `build_third_libs.sh`, `scripts/thor_build.sh`.
- **Core headers (4):** `include/dataset_reader.h`, `include/file_utils.h`, `include/pipeline.h`, `include/raw_gs_model.h`.
- **Core/runtime source (12):** `src/cv_utils.cpp`, `src/dataset_reader.cpp`, `src/file_utils.cpp`, `src/pipeline.cpp`, `src/raw_gs_model.cpp`, `src/tensor_math.cpp`, `slam/InfiniTAM_tools.cpp`, `slam/InfiniTAM_tools.h`, `slam/TsdfFusion/UIEngine.cpp`, `slam/slam_pipeline.cpp`, `slam/slam_pipeline.h`, `slam_trainer.cpp`.
- **Runners/preflight/summary (4):** `run/hybrid_validation.py`, `run/tum_loop_validation.py`, `scripts/thor_preflight.py`, `scripts/summarize_tum_runs.py`.
- **Evaluation (6):** `scripts/ate_general.py`, `scripts/rpe_general.py`, `scripts/metric.py`, `scripts/metric_general.py`, `scripts/metric_sampled_eval.py`, `scripts/geo_general.py`.
- **Documentation/contributor guidance (6):** `AGENTS.md`, `THOR_TRANSFER_GUIDE.md`, `docs/README.md`, `docs/hybrid_submap_config_guide.md`, `run/README.md`, `scripts/README.md`.

### New functional files (21)

- **Container (3):** `containers/thor/Dockerfile`, `containers/thor/hpcx_exec.sh`, `containers/thor/requirements-evaluation.txt`.
- **Benchmark contract (2):** `BENCHMARK_PROTOCOL.md`, `scripts/benchmark_protocol.py`.
- **Freeze/provenance (2):** `scripts/benchmark_freeze.py`, `scripts/benchmark_freeze_regression.py`.
- **Real-time/telemetry (4):** `scripts/analyze_realtime_trace.py`, `scripts/realtime_hardening_regression.py`, `scripts/thor_telemetry.py`, `run/thor_deployment_screen.py`.
- **System/build inspection (3):** `scripts/thor_container.sh`, `scripts/thor_system_check.py`, `scripts/thor_post_build_check.py`.
- **Shared evaluation and gates (4):** `scripts/evaluation_common.py`, `scripts/pose_evaluation_common.py`, `scripts/evaluate_deployment_promotion.py`, `scripts/promotion_hardening_regression.py`.
- **Regression contracts (3):** `scripts/metrics_hardening_regression.py`, `scripts/benchmark_hardening_regression.py`, `scripts/validation_hardening_regression.py`.

New documentation/result records are `docs/thor_initial_bringup_20260818.md`, `docs/thor_paper_hardening_20260819.md`, `BENCHMARK_FREEZE_2026-08-19.md`, `THOR_DEPLOYMENT_SCREEN_2026-08-19.md`, and `THOR_HEADLINE_BENCHMARK_2026-08-19.md`. Generated experiment YAML/manifests and run outputs are evidence, not source modifications.

## 10. Papers That Informed the Implementation

### Direct foundations and architectural influence

1. Z. Peng, K. Zhou, and T. Shao, **“Gaussian-Plus-SDF SLAM: High-fidelity 3D Reconstruction at 150+ fps,”** arXiv:2509.11574, 2025. Parent representation, scheduling, datasets, and parity target. <https://arxiv.org/abs/2509.11574>
2. B. Kerbl et al., **“3D Gaussian Splatting for Real-Time Radiance Field Rendering,”** ACM TOG 42(4), 2023, DOI: 10.1145/3592433. Gaussian parameterization, optimization, densification, and quality background. <https://doi.org/10.1145/3592433>
3. R. A. Newcombe et al., **“KinectFusion: Real-Time Dense Surface Mapping and Tracking,”** ISMAR 2011, DOI: 10.1109/ISMAR.2011.6092378. TSDF fusion and frame-to-model ICP foundation. <https://doi.org/10.1109/ISMAR.2011.6092378>
4. V. A. Prisacariu et al., **“InfiniTAM v3: A Framework for Large-Scale 3D Reconstruction with Loop Closure,”** arXiv:1708.00783, 2017. Bundled tracking/fusion framework and rigid local-map architecture. <https://arxiv.org/abs/1708.00783>
5. V. Reijgwart et al., **“Voxgraph: Globally Consistent, Volumetric Mapping Using Signed Distance Function Submaps,”** IEEE RA-L, DOI: 10.1109/LRA.2019.2953859. Motivated local TSDF maps plus sparse transform optimization. <https://arxiv.org/abs/2004.13154>
6. R. Kümmerle et al., **“g2o: A General Framework for Graph Optimization,”** ICRA 2011, DOI: 10.1109/ICRA.2011.5979949. Sparse nonlinear graph-optimization principles. The implementation uses bundled MiniSlamGraph, not g2o code. <https://ais.informatik.uni-freiburg.de/publications/papers/kuemmerle11icra.pdf>
7. E. Rublee et al., **“ORB: An Efficient Alternative to SIFT or SURF,”** ICCV 2011, DOI: 10.1109/ICCV.2011.6126544. CPU-efficient loop descriptors through OpenCV ORB. <https://doi.org/10.1109/ICCV.2011.6126544>
8. R. Mur-Artal and J. D. Tardós, **“ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras,”** IEEE T-RO 2017, DOI: 10.1109/TRO.2017.2705103. Appearance retrieval followed by geometric verification. This project does not copy ORB-SLAM's map, bundle adjustment, covisibility graph, or DBoW2 vocabulary. <https://doi.org/10.1109/TRO.2017.2705103>
9. J. G. Mangelson et al., **“Pairwise Consistent Measurement Set Maximization for Robust Multi-Robot Map Merging,”** ICRA 2018, DOI: 10.1109/ICRA.2018.8460217. Inspired bounded PCM-style consistency selection; the present single-robot implementation is a custom adaptation, not the paper's full algorithm. <https://robots.engin.umich.edu/publications/jmangelson-2018a.pdf>
10. G. H. Lee, F. Fraundorfer, and M. Pollefeys, **“Robust Pose-Graph Loop-Closures with Expectation-Maximization,”** IROS 2013, DOI: 10.1109/IROS.2013.6696406. Informed robust loop-edge treatment and failure analysis. <https://people.inf.ethz.ch/marc.pollefeys/pubs/LeeIROS13b.pdf>
11. T. Whelan et al., **“ElasticFusion: Dense SLAM Without a Pose Graph,”** RSS 2015. Informed the requirement for geometric/global consistency before accepting dense-map loop corrections. The deformation algorithm was not implemented. <https://www.roboticsproceedings.org/rss11/p01.pdf>

### Quality and evaluation literature

12. V. Yugay et al., **“Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting,”** arXiv:2312.10070. Submap scalability and Gaussian-SLAM quality context; no source copied. <https://arxiv.org/abs/2312.10070>
13. N. Keetha et al., **“SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM,”** CVPR 2024. Online Gaussian mapping and comparison context. <https://openaccess.thecvf.com/content/CVPR2024/html/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.html>
14. Z. Yu et al., **“Mip-Splatting: Alias-free 3D Gaussian Splatting,”** CVPR 2024. Used to interpret cross-resolution artifacts; its filters were not added to the frozen implementation. <https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html>
15. J. Sturm et al., **“A Benchmark for the Evaluation of RGB-D SLAM Systems,”** IROS 2012, DOI: 10.1109/IROS.2012.6385773. TUM 640×480/30 Hz dataset and ATE/RPE protocol. <https://doi.org/10.1109/IROS.2012.6385773>
16. J. Straub et al., **“The Replica Dataset: A Digital Replica of Indoor Spaces,”** arXiv:1906.05797, 2019. Photorealistic indoor quality/geometry benchmark. <https://arxiv.org/abs/1906.05797>
17. Z. Wang et al., **“Image Quality Assessment: From Error Visibility to Structural Similarity,”** IEEE TIP 2004, DOI: 10.1109/TIP.2003.819861. SSIM definition. <https://doi.org/10.1109/TIP.2003.819861>
18. R. Zhang et al., **“The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,”** CVPR 2018. LPIPS definition and perceptual interpretation. <https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html>

### Comparison literature, not used to tune r2

- H. Huang et al., **Photo-SLAM**, CVPR 2024: embedded photorealistic SLAM comparator. <https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Photo-SLAM_Real-time_Simultaneous_Localization_and_Photorealistic_Mapping_for_Monocular_Stereo_CVPR_2024_paper.html>
- Z. Peng et al., **RTG-SLAM**, SIGGRAPH 2024: compact real-time Gaussian RGB-D reconstruction comparator. <https://arxiv.org/abs/2404.19706>
- C. Feldmann et al., **DiskChunGS**, IEEE RA-L 2026, DOI: 10.1109/LRA.2026.3668704: out-of-core Gaussian chunking comparator. <https://rffr.leggedrobotics.com/works/diskchungs/>

These comparison papers should be cited in related work and used for protocol matching, but their reported FPS must not be presented as a same-hardware result unless their code is ported and run under the same Thor envelope.

### Software and Attribution Note

The repository root declares the GNU Affero General Public License v3 in `LICENSE.txt`; bundled libraries retain their own licenses, and the NVIDIA NGC image is governed by NVIDIA's software/product terms. Academic papers listed above informed design or evaluation, but their source code was not copied into the new Thor hardening files. OpenCV ORB and the already bundled InfiniTAM/MiniSlamGraph code are the concrete library implementations used. Before public code or container redistribution, preserve upstream notices and review the applicable AGPL, third-party, dataset, model-weight, and NVIDIA terms. This is an attribution checklist, not legal advice.

## 11. NVIDIA Engineering Sources

These are technical sources rather than research papers and should support the platform/methodology chapter:

- JetPack 7.1 / Jetson Linux 38.4 / CUDA 13.0 matrix: <https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.1>
- NVIDIA PyTorch–JetPack compatibility matrix: <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html>
- NVIDIA PyTorch 26.07 release notes and NVSHMEM workaround: <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html>
- CUDA 13 release notes and removed pre-Turing offline targets: <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html>
- CUDA for Tegra memory/coherency guidance: <https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/>
- Thor power, clocks, fan, over-current, and thermal behavior: <https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html>

## 12. Frozen Results and Engineering Evidence

The final R2 matrix contains 45 TUM runs and 16 complete Replica runs. The authoritative aggregate, fairness rules, rejected optimization results, and artifact paths are recorded in [PAPER_READINESS_LOCK_2026-08-25.md](../PAPER_READINESS_LOCK_2026-08-25.md). The following earlier measurements remain engineering evidence and must not be merged into the R2 tables:

- The pinned image completed `thor_container.sh check` with AArch64, SM 11.0, HPC-X, CUDA Torch, torchvision/SciPy/trimesh, offline weights, ext4, free-space, and CMake checks all passing. The Torch probe executed a GPU reduction and obtained the expected value.
- The initial 10/50/120-frame, 600×340 Replica ladder completed with on-demand direct input, bounded CPU residency, and no invalid pose, CUDA, paging-backpressure, or handoff error. A matched locked-clock 120-frame run reached 71.98 FPS versus 41.63 FPS under dynamic clocks, demonstrating that uncontrolled DVFS materially confounds short timing comparisons.
- The 320-frame `room0` engineering quality gate produced final-map hybrid PSNR 34.1636 dB, SSIM 0.962153, LPIPS 0.057538, and 51.58 training FPS, with 45,227 Gaussians across three submaps. It is a short 600×340 gate using a perceptual profile, not the locked 2,000-frame 1200×680 paper protocol.
- The corrected sliced deployment screen processed 6,673/6,673 TUM frames. Sliced FPS was 66.70/41.65/43.64 on `fr1_desk`/`fr2_xyz`/`fr3_office`, but backlog/recovery and mean online ATE failed promotion. Peak junction temperature was 48.06 °C, mean/peak VIN was 57.32/83.59 W, and over-current deltas were zero.
- All historical results in `THOR_HEADLINE_BENCHMARK_2026-08-19.md` predate the final protocols/freeze and remain contextual evidence only. They must not be relabelled as R2 results.

## 13. Claims, Limitations, and Threats to Validity

### Defensible now

- The project builds and runs natively on AArch64/SM 11.0 in a pinned NVIDIA environment.
- The research implementation performs online submap construction and loop correction without offline frame reintegration or Gaussian refinement.
- The r2 code/protocol/binary/container identity is reproducibly frozen.
- Immediate scheduling is the frozen paper default; the attempted sliced deployment optimization was rejected by predeclared gates.
- Across five repetitions, the proposed tracking-isolation protocol improves mean ATE by 1.68% on the two valid paired baseline TUM scenes and completes `fr3/office`, where all faithful single-map baseline runs fail acceptance.
- The integrated proposed pipeline averages 2.8372 cm corrected ATE across the three TUM scenes and exceeds 30 FPS average terminal-inclusive throughput on every scene.
- Across all eight full-resolution Replica scenes, the proposed final map averages 37.1495 dB PSNR, 0.95691 SSIM, and 0.11141 LPIPS at 51.21 FPS. This is 0.091 dB below the GPS-SLAM paper PSNR average and 53.1% faster than the local faithful baseline on Thor.
- A 42.98-minute continuous soak completed 24 fresh integrated `fr3/office` maps (59,640 frames) at 47.454 FPS mean with no negative early-to-late drift, over-current delta, thermal throttle, error, or restart. See `../thor_runs/paper_soak_r2_20260825/telemetry_continuous/`.

### Not defensible from the current evidence

- “Sustained 30 Hz,” “hard real time,” or “every frame meets 33.3 ms.”
- A same-hardware advantage over Photo-SLAM, DiskChunGS, or another paper not yet run on Thor.
- Generalization beyond the three TUM paper sequences and eight Replica scenes.

### Required disclosures

- The original snapshot was not a clean Git commit; use the archive/checksum/freeze identity.
- The proposed and baseline TSDF capacity profiles differ by design.
- Container CUDA 13.3 ran over a JetPack 7.1 CUDA-13.0-capable driver using compatibility mode, validated by real kernels.
- TensorBoard was removed from the Thor binary because of an ABI conflict; evaluation artifacts were not removed.
- GT-pose Replica results measure rendering/mapping isolation, not tracking.
- Corrected final TUM trajectories may include terminal work; causal deployment accuracy must use `online_end`.
- MAXN is a ceiling mode and an over-current notification invalidates the affected run even if outputs look complete.

## 14. Final Thesis Result Tables to Populate

For each **TUM** arm/sequence over five repeats, report mean ± standard deviation for raw/corrected ATE, translational/rotational RPE, raw failed/poor frames, processed/expected frames, steady-state FPS, terminal-inclusive FPS, terminal latency, p95/p99/max service time, peak RSS, and clean telemetry status. Report an unweighted three-sequence mean only alongside per-sequence values.

For each **Replica** arm over `office0–4` and `room0–2`, report PSNR, SSIM, LPIPS, geometry accuracy/completion/F1 at the declared threshold, Gaussian count, training FPS, final rendering time/FPS, peak RSS, and complete render coverage. Keep baseline, proposed parity, and any later promoted deployment protocol in separate columns.

For the **hardware soak**, report duration, repeated workload/sequence, power mode, clock lock, fan state, mean/peak VIN power, maximum junction temperature, over-current deltas, thermal/cooling state, peak RSS, errors/restarts, and performance drift from the first to last interval.

For external papers, add a protocol-compatibility column: sensor, resolution, frame count/split, pose source, online/offline work, hardware, power mode, timing boundary, and metric implementation. Use “reported by authors” for literature values and “reproduced on Thor” only for local runs. This prevents an attractive but invalid cross-paper FPS table.

## 15. Suggested Thesis Chapter Structure

1. **Background:** RGB-D SLAM, TSDF fusion, 3D Gaussian splatting, loop closure, embedded constraints.
2. **Baseline:** GPS-SLAM representation, original scheduling, datasets, strengths, and single-map limitations.
3. **Proposed method:** local SDF/Gaussian submaps, ownership/transforms, lifecycle, ORB/ICP loop frontend, consistency gates, sparse graph correction, and metadata-only apply.
4. **Thor port:** AArch64/SM 11.0, pinned container, ABI/toolchain fixes, direct input/prefetch, bounded residency, memory ownership, and safety.
5. **Experimental methodology:** frozen protocols, parity versus deployment distinction, TUM/Replica metrics, FIFO real-time definition, telemetry, promotion gates, and statistical repetitions.
6. **Results:** accuracy, rendering/geometry quality, throughput/tails, resource/power, soak, ablations, and external comparisons.
7. **Discussion:** accuracy-quality-latency trade-offs, rejected slicing result, capacity fairness, container/native implications, limitations, and future work.

The authoritative release companions are [Architecture](ARCHITECTURE.md), [Configuration](CONFIGURATION.md), [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md), [Benchmark hardening](BENCHMARK_HARDENING.md), and [BENCHMARK_FREEZE_2026-08-19.md](../BENCHMARK_FREEZE_2026-08-19.md). The chronological development log remains in the private research archive.
