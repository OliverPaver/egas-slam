# Robust Submap-Based Gaussian-plus-SDF SLAM with Loop Closure on NVIDIA Jetson AGX Thor

This repository is a research fork of [GPS-SLAM](https://github.com/MisEty/GPS-SLAM). It adds bounded SDF and Gaussian submaps, online pose-graph loop closure, tracking-failure containment, embedded-memory controls, benchmark provenance, and a native SM 11.0 deployment path for NVIDIA Jetson AGX Thor. The original BasicEngine/unified-Gaussian path is retained as the faithful parent-system baseline.

The frozen paper implementation is `thor-paper-20260819-r2-source`. Release-only changes after that tag affect paths, documentation, licensing, checks, and packaging—not the algorithm tree; see [RELEASE_EQUIVALENCE.md](RELEASE_EQUIVALENCE.md).

## Status and supported platform

The validated platform is Jetson AGX Thor with JetPack 7.1. The container builds native AArch64 code and CUDA SM 11.0 kernels; it does not emulate another architecture. Desktop configurations from upstream remain useful references but are not validated by this fork.

Locked results support average full-pipeline throughput above 30 FPS on the tested sequences, not an every-frame 33.3 ms deadline guarantee. On Thor, the proposed system completed all three TUM sequences and averaged 2.8372 cm integrated ATE. Across eight full-resolution Replica scenes it averaged 37.1495 dB PSNR, 0.95691 SSIM, 0.11141 LPIPS, and 51.21 FPS. Read the acceptance rules and important qualifications in [PAPER_READINESS_LOCK_2026-08-25.md](PAPER_READINESS_LOCK_2026-08-25.md).

## Quick start on Thor

Requirements: Docker with the NVIDIA runtime, JetPack 7.1, and access to the pinned NVIDIA PyTorch base image.

```bash
git clone https://github.com/OliverPaver/gps-slam-submaps-thor.git
cd gps-slam-submaps-thor

export GPS_SLAM_DATA_ROOT=/path/to/datasets
export GPS_SLAM_WEIGHT_ROOT=/path/to/torch_hub
export GPS_SLAM_RUN_ROOT=/path/to/writable/runs

bash scripts/thor_container.sh image
bash scripts/thor_container.sh check
bash scripts/thor_container.sh build
bash scripts/thor_container.sh baseline-build
bash scripts/thor_container.sh regressions
```

`GPS_SLAM_WEIGHT_ROOT` must contain `checkpoints/vgg.pth` and `checkpoints/vgg16-397923af.pth`. Data and weights are mounted read-only; results are written under `/workspace/runs`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for dataset preparation, exact benchmark commands, output interpretation, and environment capture.

## Repository layout

- `slam/`, `src/`, `include/`: pipeline, submaps, tracking, and Gaussian model code.
- `InfiniTAM/`, `gsplat/`: SDF/tracking and CUDA rasterization components.
- `configs/release/`: upstream and fixed release configurations.
- `run/`, `scripts/`: experiment runners, metrics, provenance, and regressions.
- `containers/thor/`: pinned Thor development/runtime image.
- `docs/`: architecture, configuration, porting, and benchmark notes.
- `ThirdLibs/`: pinned source dependencies; generated installs are never tracked.

## Datasets and outputs

Datasets, pretrained weights, models, renders, and run workspaces are intentionally excluded from Git. Replica and TUM RGB-D require preprocessing into the layouts documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). A run normally emits its generated YAML, poses, timing traces, benchmark manifest, metrics, logs, and telemetry below the selected run root.

## Licence and citation

GPS-SLAM is distributed under AGPL-3.0, but bundled research components impose additional conditions, including non-commercial restrictions. Do not treat the combined repository as unconditionally AGPL-only or commercially reusable. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), `LICENSES/`, and [PROVENANCE.md](PROVENANCE.md) before redistribution.

Use [CITATION.cff](CITATION.cff) for this software and cite the original GPS-SLAM paper separately:

> Z. Peng, K. Zhou, and T. Shao, “Gaussian-plus-SDF SLAM: High-fidelity 3D Reconstruction at 150+ fps,” *Computational Visual Media*, 2025.

Publication of this fork remains conditional on University and third-party redistribution approval. Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md).
