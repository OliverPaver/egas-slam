# Thor build and runtime rationale

The supported deployment is native AArch64 on NVIDIA Jetson AGX Thor. The
container provides a compatible userspace and toolchain; it does not emulate a
desktop CPU or GPU. Follow [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the
maintained build and run commands.

## CUDA and Torch

The Thor build selects CUDA architecture `110` explicitly. CMake rejects an
unspecified architecture rather than accepting a compiler default. The build
wrapper checks the compiler's architecture support and the selected Torch CUDA
runtime. Native C++ and CUDA targets use C++17, and Torch's C++ ABI flags are
propagated to the relevant targets.

Use the supplied NVIDIA PyTorch image and build the project with the Torch
package from that environment. A desktop LibTorch archive, an unrelated PyPI
Torch build or a library copied from another container may have incompatible
CPU architecture, CUDA kernels or C++ ABI. Use the same container environment
for building and running the native executable.

CUDA fast math remains disabled. Enabling it changes numerical behavior and
must be treated as a distinct build. Source identity, compiler settings,
container identity and the resulting executable hash describe different parts
of a build; one does not substitute for the others.

## Dependencies

| Dependency | How it is supplied |
| --- | --- |
| CUDA, cuDNN, Torch/Torchvision | NVIDIA container/runtime environment. |
| OpenCV, Eigen, Boost and Protobuf development packages | Container system packages. |
| indicators, tensorboard_logger, tinyply, yaml-cpp | Bundled sources built by `build_third_libs.sh`. |
| cxxopts, GLM, nanoflann, nlohmann/json | Bundled `ThirdLibs/fetchcontent/` sources consumed by CMake. |
| InfiniTAM and Gaussian rasterization | Source in this repository. |
| Evaluation Python packages | Container requirements and the documented environment. |

CMake requires the bundled FetchContent source directories instead of silently
downloading replacements. Generated dependency installs under
`ThirdLibs/install/` are local build products and must be rebuilt for the target.
Even with TensorBoard disabled for the Thor executable, the current dependency
builder installs `tensorboard_logger`; retain its source and notices.

The Torch image includes communication libraries whose versions must agree.
`containers/thor/hpcx_exec.sh` selects the matching HPC-X OpenMPI/UCC/UCX stack
before starting a process. Keep this wrapper in the runtime path rather than
combining loader directories from unrelated installations.

## Build variants

The proposed preset builds `medium_128k` into `build_thor/`. The
original-algorithm preset builds `legacy_pool_2m` into `build_original/`. Build
directories and generated dependency installs belong on NVMe/ext4, not exFAT.
Do not reuse a CMake cache from another architecture or profile.

The default Thor build omits the unused GLUT UI and the optional TCP viewer and
disables TensorBoard to avoid mixing incompatible Protobuf implementations.
The viewer can be requested explicitly and requires Boost.System. Pangolin is
not a dependency of the documented headless build. NVML is optional: its absence
changes available memory diagnostics, not the requirement for CUDA execution.

The optional rasterizer-archive reuse interface requires an exact archive,
digest and dependency manifest. A normal clean-source build leaves these
variables empty and compiles the rasterizer. Compiled archives are not shipped
as a substitute for source.

## External paths and run boundaries

The wrapper obtains datasets, weights and output roots from explicit environment
variables. Data and weight mounts are read-only; outputs use a dedicated
writable run root. `/workspace/data` and `/workspace/runs` are container paths,
not required host directory names. Avoid embedding personal host paths in source
or scene templates.

The runtime uses shared system memory on Thor. CUDA allocation, process RSS,
container charge, host available memory and serialized map size have distinct
meanings. Preserve their measurement scopes. GPU frequency, power mode,
temperature and competing processes can affect observations and should be
recorded with performance measurements.

The release's source checks and CPU regressions do not assert Windows, desktop
CUDA or other JetPack support. A newly built combined source, including the
evaluation-only saved-overlap correction, has a new binary identity. See
[PROVENANCE.md](../PROVENANCE.md) for its relationship to the measured native
implementation.
