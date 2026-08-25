# Jetson AGX Thor Transfer and Bring-Up Guide

This snapshot is prepared for a native NVIDIA Jetson AGX Thor build. Do not run
or link any desktop `build*`, `ThirdLibs/install`, or `ThirdLibs/libtorch`
artifact: those files are x86-64 and were deliberately excluded. Build on the
Thor's internal NVMe/ext4 filesystem, not directly on the exFAT USB drive.

## Bundle Contents

- `GPS-SLAM/`: browsable current source, configs, scripts, documentation,
  bundled third-party **source**, and Replica `office0` full/half test data.
- `GPS-SLAM_Thor_Source_2026-08-18.tar.gz`: canonical archive preserving Unix
  permissions and symlinks. Prefer extracting this on Thor.
- `offline_weights/torch_hub/checkpoints/`: VGG16 and LPIPS-VGG weights used by
  the quality scripts, so quality scoring need not download weights.
- `datasets/tum_rgbd/processed/`: the three full-resolution TUM paper scenes,
  copied as standalone files rather than machine-specific symlinks.
- `datasets/replica/room0_half_clean/`: the half-resolution Replica quality-gate
  scene. `office0` full/half is already under `GPS-SLAM/data/replica/`.
- `SHA256SUMS`: integrity hashes for the archive, Git bundle, weights, and
  transfer metadata.
- `repository_history.bundle`: the small committed upstream Git history. The
  source snapshot, not this bundle, contains the extensive current working
  changes.
- `TRANSFER_MANIFEST.md`: included/excluded paths, source commit, file counts,
  and verification information.

## 1. Prepare Thor

As checked on 2026-08-18, NVIDIA lists JetPack 7.2 / L4T r39.2 as the
latest Thor release.
This corrects the checksum-covered `README_FIRST.md` label of 7.2.1/r39.2.1;
that received transfer record remains unchanged so its integrity hash stays valid.
Record the installed versions before changing anything:

```bash
uname -a
dpkg-query -W nvidia-l4t-core nvidia-jetpack 2>/dev/null || true
/usr/local/cuda/bin/nvcc --version
python3 GPS-SLAM/scripts/thor_system_check.py || true
```

This Thor was inspected running L4T r38.4 (JetPack 7.1), with CUDA 13.2
installed at `/usr/local/cuda-13.2` and an SM 11.0 GPU.
NVIDIA's current framework matrix explicitly supports that release with the
multi-architecture PyTorch 26.07 container. JetPack 7.2 uses L4T r39.2 and
requires a BSP reinstall rather than an ordinary APT upgrade; do not reflash
only to perform the initial GPS-SLAM validation.

For a host-native toolchain, install the JetPack development stack using
NVIDIA's repository. This is optional for the container workflow below. Do
**not** use Ubuntu's similarly named `nvidia-cuda-toolkit` package:

```bash
sudo apt update
sudo apt install nvidia-jetpack
```

Thor is CUDA compute capability 11.0. Confirm the compiler supports it:

```bash
/usr/local/cuda/bin/nvcc --list-gpu-arch | grep compute_110
```

Official references:

- https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html
- https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html
- https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html

## 2. Extract to NVMe

Copy the archive from the USB, verify it, and extract it to a normal Linux
filesystem. Replace `/path/to/usb` with the mounted USB directory.

```bash
cd /path/to/usb/GPS-SLAM_Thor_Transfer_2026-08-18
sha256sum -c SHA256SUMS
sha256sum -c SOURCE_FILES.sha256 --quiet
mkdir -p /mnt/nvme/gps-slam
tar -xzf GPS-SLAM_Thor_Source_2026-08-18.tar.gz -C /mnt/nvme/gps-slam
mkdir -p /mnt/nvme/gps-slam-data
rsync -rt datasets/ /mnt/nvme/gps-slam-data/
cd /mnt/nvme/gps-slam/GPS-SLAM
```

If `/mnt/nvme` is not a real mounted filesystem on your Thor, use a directory
under the internal NVMe-backed user filesystem instead. Keep at least 30 GB
free for dependencies, builds, paging, and smoke-run outputs.

## 3. Install Native Build Dependencies

Skip this host step when using the recommended container workflow in section
5; its image installs these packages without changing the BSP.

```bash
sudo apt install build-essential cmake ninja-build git pkg-config \
  python3 python3-pip python3-venv libopencv-dev libeigen3-dev \
  libprotobuf-dev protobuf-compiler libboost-system-dev
```

All non-Torch third-party sources required by `build_third_libs.sh` are in the
bundle. The four small `FetchContent` dependencies are also vendored under
`ThirdLibs/fetchcontent`, so the CMake configure does not need GitHub. Native
third-party libraries must be rebuilt because the USB contains no desktop
binaries.

## 4. Install a JetPack-Compatible PyTorch/LibTorch

The application is C++17 and needs Torch headers, shared libraries, and
`TorchConfig.cmake`, not merely a CPU Python package. Use the NVIDIA framework
container or NVIDIA package explicitly listed for the installed JetPack
release. Do not install generic PyPI `torch` and assume it has Thor GPU support.
NVIDIA's current matrix lists recent JetPack 7 releases primarily through
framework containers, so verify the exact pairing here:

- https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html
- https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

Inside the selected native environment, verify:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.utils.cmake_prefix_path)
print(torch.cuda.get_device_name(0))
PY
```

The tested desktop project used Torch 2.3.1, but that x86/CUDA-11.8 binary is
not portable to Thor. Prefer the version supported by Thor's JetPack; do not
copy `ThirdLibs/libtorch` from the desktop.

## 5. Build on Thor

For JetPack 7.1, use the included NVIDIA PyTorch 26.07 container. Its
multi-architecture base is pinned by registry digest, and the image contains
the development headers and `TorchConfig.cmake` needed by the C++ build.
Docker and NVIDIA Container Toolkit are already present on the inspected Thor.

The least-persistent first run is an interactive, one-off sudo workflow:

```bash
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh image
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh check
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh build
```

Each command asks for sudo authentication when needed. Alternatively, an
administrator can deliberately configure rootless Docker or daemon access.
Adding a user to the `docker` group grants root-equivalent host privileges;
do not present it as a harmless convenience.

The wrapper mounts this transfer read/write at `/workspace/transfer`, keeps generated
build files owned by the calling user, and stores run output at
`/workspace/runs`. Set `GPS_SLAM_RUN_ROOT=/absolute/host/path` to change the
host output directory. The image supplies CUDA-enabled Torch and
`TorchConfig.cmake`; do not install generic PyPI Torch into it. Run the trainer
and Python launchers through this same wrapper: `build_thor/slam_trainer` links
against the container's Torch libraries and is not a host-native executable.
The wrapper also sets `NVIDIA_IMEX_CHANNELS=0`, matching NVIDIA's 26.07
workaround for a possible NVSHMEM container-launch segmentation fault.

If a supported native AArch64 LibTorch installation is available outside a
container, the native helper remains valid. It rebuilds bundled dependencies,
discovers Torch through Python (or `GPS_SLAM_TORCH_PREFIX`), targets SM 11.0,
and builds the recommended `medium_128k` hash profile:

```bash
export GPS_SLAM_BUILD_JOBS=6
bash scripts/thor_build.sh
```

Use fewer jobs if memory or thermals become constrained. To point at a manual
LibTorch installation:

```bash
export GPS_SLAM_TORCH_PREFIX=/opt/libtorch
bash scripts/thor_build.sh
```

The Thor profile disables the optional C++ TensorBoard writer. NVIDIA Torch
ships private protobuf headers that are incompatible with the system protobuf
used by that writer; keeping it out of the trainer avoids a mixed-ABI process.
Image logs, runtime summaries, and evaluation artifacts remain enabled. CMake
discovers NVML without x86 paths; if NVML is absent, GPU-memory diagnostics
fall back to `cudaMemGetInfo` and reconstruction remains enabled.

## 6. First Smoke Tests

Run cheap regressions before loading a dataset:

```bash
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh regressions
```

First run a 10-frame half-resolution hybrid smoke with GT poses. This isolates
the TSDF/submap/Gaussian path; it does **not** validate live tracking:

```bash
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh run \
  python3 run/hybrid_validation.py --frames 10 --scene-name office0 \
  --binary build_thor/slam_trainer \
  --output-root /workspace/runs/replica_smoke \
  --min-free-gb 30 --labels multi_hybrid_warm_fill \
  --gaussian-profile fast --hybrid-submap-profile stress \
  --multi-hash-profile medium_128k \
  --tsdf-residency-profile thor_cpu_bounded \
  --tsdf-input-mode on_demand --tsdf-input-prefetch on \
  --tsdf-input-tensor-bridge direct --tsdf-input-bridge-validate \
  --reader-preload-images false --submap-stats-interval-frames 10 \
  --gaussian-submaps-debug off --gaussian-insert-diagnostics off \
  --disable-render-overlap-diagnostics --no-save-after-train \
  --quiet-run-logs --thor-preflight --trainer-timeout-sec 1800 \
  --random-seed 42 --tag thor_gt_10f
```

This first run executes the direct and legacy input bridges and requires exact
agreement. Remove `--tsdf-input-bridge-validate`, repeat the command with
`--hybrid-use-estimated-pose`, and tag it `thor_est_10f`; this second run
exercises InfiniTAM tracking. If both pass,
repeat estimated-pose tests at 50 and 120 frames, changing both `--frames` and
`--tag`; the 10-frame stress run is too short to cross the profile's 25-frame
submap boundary, while the longer runs exercise handoff. The runner's
`--thor-preflight` option validates every generated YAML
before the trainer can clear its workspace or allocate SDF maps.

Run the exact three-scene accuracy matrix only after the smoke gates pass. All
paths below are container paths; heavy artifacts appear in the host's
`thor_runs/` directory unless `GPS_SLAM_RUN_ROOT` is set:

```bash
GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh run \
  python3 run/tum_loop_validation.py \
  --tum-root /workspace/transfer/datasets/tum_rgbd/processed \
  --trainer build_thor/slam_trainer \
  --output-root /workspace/runs/tum_thor \
  --config-dir /workspace/runs/tum_thor/configs \
  --sequences fr1_desk fr2_xyz fr3_office \
  --variants multi_sdf_orb_apply --tracking-profile robust_rpe \
  --tsdf-lifecycle-profile paper_parity \
  --loop-profile recommended_longspan_robust \
  --hash-profile medium_128k --tsdf-residency-profile thor_cpu_bounded \
  --tsdf-input-mode on_demand --tsdf-input-prefetch on \
  --tsdf-input-tensor-bridge direct --submap-stats-interval-frames 10 \
  --pose-graph-overlap-roundtrip-check true \
  --pose-graph-overlap-roundtrip-tolerance 1e-4 \
  --require-raw-tracking-success true \
  --raw-tracking-failure-grace-frames 5 \
  --random-seed 42 --no-save-after-train --full-sequence \
  --thor-preflight --trainer-timeout-sec 14400 --min-free-gb 30 \
  --quiet-run-logs --tag thor_initial --run
```

Do not add `--continue-on-error` to acceptance runs. A trainer failure,
timeout, invalid/missing summary, non-finite pose, or hard summary blocker must
produce a non-zero command exit.

That first matrix isolates estimated tracking, SDF submaps, and loop closure.
It does not exercise their joint Gaussian path. Next, repeat a bounded
`fr1_desk` gate with the same options, remove `--full-sequence`, and change the
sequence/variant arguments to:

```text
--sequences fr1_desk --variants multi_hybrid_apply --max-frames 120 \
--tag thor_joint_120f
```

If it passes, remove `--max-frames`, restore all three sequences, and run the
full joint matrix. The initial one-seed pass is bring-up evidence; reproduce
the established reference protocol with five explicitly recorded repeats only
after the system is stable.

## 7. Quality Metrics and Offline Weights

Quality scoring additionally needs a Torch-compatible `torchvision`; use the
version already paired with the selected NVIDIA Torch image. Geometry scoring
uses SciPy and the pure-Python `trimesh` wheel pinned by hash in
`containers/thor/requirements-evaluation.txt`; Open3D is no longer required.
Geometry is not an initial build/tracking smoke gate, but it is mandatory before
declaring quality parity or release acceptance.

The container wrapper mounts the bundled `torch_hub` directory read-only at
PyTorch's expected `$TORCH_HOME/hub` path. For a supported native environment,
create the equivalent cache layout explicitly (do not point `TORCH_HOME`
directly at `offline_weights`, because PyTorch appends `hub/checkpoints`):

```bash
mkdir -p /mnt/nvme/gps-slam-metric-cache
ln -s /path/to/transfer/offline_weights/torch_hub \
  /mnt/nvme/gps-slam-metric-cache/hub
export TORCH_HOME=/mnt/nvme/gps-slam-metric-cache
```

The established half-resolution quality gate uses GT poses. The current
reference for 320-frame Replica `room0` is 34.0090 PSNR, 0.959740 SSIM, and
0.058641 LPIPS. Compare like-for-like; do not compare half-resolution renders
against native-resolution dataset images.

## 8. Accuracy and Performance Bring-Up

After the 10/50/120-frame gates:

1. Run full-resolution Replica and record quality plus FPS.
2. Preprocess/copy TUM and run `fr1_desk`, `fr2_xyz`, then `fr3_office` with
   `paper_parity`, `robust_rpe`, and `recommended_longspan_robust`.
3. Record `sudo nvpmodel -q`, cooling, clocks, `tegrastats`, Nsight Systems,
   `timing.csv`, `paging.csv`, tracking diagnostics, ATE/RPE, and quality.
4. Start at a supported capped power mode and record its ID. The inspected
   machine was in MAXN; NVIDIA describes MAXN as experimental and does not
   recommend prolonged heavy workloads in it. Use MAXN or `jetson_clocks` only
   for a separately labelled, thermally monitored comparison.
5. Stop on OOM, non-finite poses, missing frames, sustained thermal
   throttling, paging backpressure, or unexplained quality/trajectory loss.

The current desktop TUM reference is 2.8446 cm corrected paper-scene ATE over
five repeats. The known unresolved runtime risk is a roughly 1.22 s terminal
loop-candidate scan on `fr3_office`; average FPS hides this tail.

Thor's CPU and GPU share physical memory, but coherency is not free. Keep the
current direct input bridge, on-demand prefetch, bounded residency, disabled
debug reductions, and sampled stats for the first trace. Benchmark registered,
pageable, pinned, or managed-memory changes only after observing a native
bottleneck: https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/

## Troubleshooting Checklist

- `TorchConfig.cmake` missing: use the NVIDIA Torch environment and set
  `GPS_SLAM_TORCH_PREFIX` to its CMake prefix.
- `compute_110` missing: the wrong CUDA development toolkit is active.
- Runtime cannot find Torch libraries: build and run in the same NVIDIA
  environment; inspect `ldd build_thor/slam_trainer`.
- `libucc.so.1: undefined symbol: ucs_config_doc_nop` or an
  `ompi_mpi_short_float` import failure: rebuild `image` from the current
  Dockerfile, then rerun `check`. The ARM64 image omits `hpcx-init.sh`, so the
  wrapper recreates its complete version-matched environment; the check must
  resolve both `libucs` and `libmpi` under `/opt/hpcx`. Do not replace the
  image's OpenMPI/UCC/UCX packages.
- The 26.07 image contains CUDA 13.3 while the JetPack 7.1 driver advertises
  CUDA 13.0 compatibility. Its startup warning is acceptable only when `check`
  passes the CUDA tensor probe and reports capability `(11, 0)`.
- Errors in `summary.pb.h` mentioning a pure `google::protobuf::Message::New()`
  mean an old build was configured with TensorBoard enabled. Rerun
  `scripts/thor_container.sh build`; the Thor helper now configures
  `GPS_SLAM_ENABLE_TENSORBOARD=OFF` and the post-build check enforces it.
- Hash-profile mismatch: rebuild with `medium_128k` or make YAML and binary
  profiles match exactly.
- OpenGL/Pangolin failure: the default trainer build is headless and does not
  require either library. The optional `remote_viewer` is a TCP target, enabled
  with `GPS_SLAM_BUILD_REMOTE_VIEWER=1`; the legacy GLUT UI remains disabled.
- LPIPS tries to download: use the wrapper's read-only cache mount, or make
  `$TORCH_HOME/hub/checkpoints` resolve to the bundled `torch_hub/checkpoints`.
- Do not point `workspace_dir` at source, datasets, or the USB: run startup may
  clear that workspace directory.
