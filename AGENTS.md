# Repository Guidelines

## Project Structure & Module Organization

GPS-SLAM is a C++17/CUDA RGB-D SLAM system. Core interfaces live in `include/` and implementations in `src/`. SLAM orchestration and InfiniTAM adapters are under `slam/`; CUDA rasterization is under `gsplat/`. The bundled InfiniTAM dependency lives in `InfiniTAM/`. YAML experiments are in `configs/`, evaluation tools in `scripts/`, and batch workflows in `run/`.

## Build, Test, and Development Commands

- `GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh build` builds vendored dependencies and `build_thor/slam_trainer` in the pinned NVIDIA Torch environment.
- `bash scripts/thor_build.sh` performs the equivalent native build when a compatible AArch64 Torch development package is installed.
- `GPS_SLAM_BUILD_REMOTE_VIEWER=1 bash scripts/thor_build.sh` also builds the optional TCP viewer.
- `GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh regressions` runs the tracking, validation, metric, and pose-graph checks.
- `GPS_SLAM_DOCKER_SUDO=1 bash scripts/thor_container.sh run ./build_thor/slam_trainer path/to/config.yaml` runs one scene in the same Torch environment used for the build.
- `python3 scripts/thor_preflight.py path/to/config.yaml` validates destructive paths, data, resources, and configuration before a long run.

## Coding Style & Naming Conventions

Match adjacent code: four-space indentation and braces on their own lines. Use `PascalCase` for classes/structs, the established `camelCase` for C++ methods, and `snake_case` for Python names and YAML keys. CUDA files use `.cu`/`.cuh`; C++ interfaces follow the neighboring `.h` or `.hpp` convention. Keep tensor device, dtype, shape, and ownership changes explicit. No repository-wide formatter is configured; avoid unrelated formatting churn.

## Testing Guidelines

There is no C++ unit-test suite or coverage threshold. Every change must compile and receive an end-to-end smoke run with a small scene/config. For algorithm changes, compare against an unchanged baseline config and report PSNR/SSIM/LPIPS, ATE, geometry metrics, or FPS as applicable using `scripts/metric_general.py`, `scripts/ate_general.py`, and `scripts/geo_general.py`. Record the exact YAML and hardware; inspect workspace logs for NaNs, tracking loss, and CUDA errors.

## Commit & Pull Request Guidelines

The bundled history has two lowercase imperative subjects: `add remote viewer` and `update bibtex`. Prefer concise scoped variants, e.g. `slam: add submap handoff diagnostics`. Exclude outputs, datasets, builds, and checkpoints. Pull requests should explain behavior/config changes, list validation/results, link issues, and attach renders or logs when output quality changes.

## Configuration & Safety

Preserve a disabled-by-default baseline for experimental features. Document new YAML keys beside representative release configs, validate ranges at startup, and emit actionable debug messages. Remember that `workspace_dir` may be cleared during a run; never point it at source, dataset, or shared-result directories.
