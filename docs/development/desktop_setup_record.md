# EGAS-SLAM on the RTX 3070 desktop

This is the new local development checkout, prepared on 3 November 2026. It
starts from the final GitHub release `c944954d7000d2c905975c784f30a99ff7f3ff04`,
including the accepted saved-map overlap restoration. That commit was checked
against live GitHub and restored with its complete Git history from the SanDisk
source archive. All 3,966 source-manifest entries passed verification before
desktop changes. The historical Thor benchmark results are unchanged.

## Locations

| Purpose | Location |
| --- | --- |
| Editable source and local Git branch | `$HOME/EGAS-SLAM` |
| Desktop executable | `build_desktop/slam_trainer` |
| Generated data, runs, logs and verification | `$HOME/egas_workspace` |
| Per-run config, receipt and native log | `~/egas_workspace/runs/<name>/record/` |
| Saved maps, trajectories and native timing | `~/egas_workspace/runs/<name>/map/` |
| Original source verification and cleanup record | `~/egas_workspace/setup/` |

The source archive was copied from the SanDisk before extraction. Builds and
execution use local files only. Previously existing implementation trees remain
in their original locations; old generated run payloads/build products were
cleaned separately and recorded in the cleanup manifest.

## Build

```bash
cd $HOME/EGAS-SLAM
source scripts/desktop_env.sh
bash scripts/desktop_build.sh
```

The build uses the existing x86-64 toolchain: GCC 13.3, CUDA 12.8, OpenCV 4.6,
and PyTorch/LibTorch 2.10.0+cu128 from the existing Miniconda installation.
The GPU target is **SM 86**, suitable for this RTX 3070. The PC has a Ryzen 7
7700X, 32 GB system RAM and 8 GB discrete GPU memory. Four build workers are the
default; `EGAS_BUILD_JOBS=2 bash scripts/desktop_build.sh` reduces build memory.

The small vendored libraries were rebuilt for x86-64. Thor's ARM objects and
libraries are not reused. The selected `medium_128k` SDF capacity, fast-math OFF,
TensorBoard OFF and legacy UI OFF match the final proposed-system build choices.
The environment retains `NVIDIA_TF32_OVERRIDE=0` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The headless trainer does not require the
optional remote viewer.

Two CMake portability corrections are retained. The first strips the trailing
newline from `VERSION` before placing it in compiler flags; otherwise CMake's
Unix Makefiles generator produces malformed makefile lines. The second enables
`GPS_SLAM_PREFER_HOST_CXX_RUNTIME`: the system C++ runtime must precede this
LibTorch package in the executable's dependency order. This wheel exports a
filesystem removal implementation with an unavailable weak iterator function,
which crashes when deleting a nonempty SDF paging directory. A small independent
reproduction demonstrated the failure and the link-order repair. The build now
runs `desktop_filesystem_regression` against both recursive-removal overloads.
No tracker, CUDA algorithm, Gaussian optimiser or loop-closure implementation
was changed by these repairs; the new link option defaults OFF for the unchanged
Thor build workflow.

## Run and develop

```bash
source scripts/desktop_env.sh

# Full Replica sequence, estimated poses, half resolution, saved map.
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene office0

# Short startup check with an explicitly named NEW output folder.
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene office0 \
  --frames 60 --name my_new_smoke_run

# Full TUM office sequence at native resolution.
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset tum --scene fr3_office

# Render 20 observed views from a successfully saved Replica run.
"$EGAS_PYTHON" scripts/desktop_run.py eval \
  --run $HOME/egas_workspace/runs/REPLACE_WITH_RUN_NAME --views 20
```

Run names are exclusive: the launcher refuses an existing name. Native startup
clears only its newly allocated `map` child. Receipts/logs are outside that child.
`--no-save` omits the large checkpoint export, useful when investigating tracking
or processing cost; a run made that way cannot later render its saved map.
`--dry-run` validates inputs and writes the exact YAML without starting SLAM.

The short `./egas` wrapper sources the environment automatically, for example
`./egas train --dataset replica --scene office0`, `./egas build`, and
`./egas summary $HOME/egas_workspace/runs/<name>`.

To prepare another Replica scene from the already local raw data:

```bash
"$EGAS_PYTHON" scripts/desktop_run.py prepare --scene office1
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene office1
```

Use the final templates as the starting point for research changes. Generated
per-run YAML is retained in `record/config.yaml`; it records precisely what ran.
Native timing, tracking status, submap events, poses and model exports remain
available for analysis. Further feature development belongs on this desktop
branch; the original release is available through Git.

## Resolution and measurement scope

The desktop launcher prepares Replica colour/depth at **600 × 340**, with
intrinsics `[300, 300, 299.75, 169.75]` and reader downscale factor **1**. This
keeps both the SDF engine and Gaussian cameras calibrated to the actual images.
Simply setting the final reader's downscale factor to 2 would resize cameras
without updating the SDF engine's calibration. Original full-resolution data
remains untouched. The inherited preparation method uses area interpolation for
colour, nearest-neighbour depth, and writes JPEG colour images.

All remaining algorithm selectors are inherited from `configs/final`: estimated
poses, immediate 20-iteration/10-frame Gaussian scheduling, CPU-backed SDF
paging, fresh-study Gaussian paging disabled, and the guarded pipeline loop
subsystem enabled. The separate internal MultiEngine loop switch remains off.
The SDF residency calculation uses this PC's available VRAM. Its nominal CPU
cache limit is 8 GiB; this is not a total application-memory cap.

These desktop runs are new validation observations. Resolution, preparation,
GPU, architecture and Torch version differ from Thor. Do not compare their FPS
directly with the full-resolution thesis tables. Native processing FPS and the
launcher's whole-command wall time cover different work; neither establishes an
every-frame real-time deadline.

## Validation record

The optional full-resolution visualisation workflow is documented in
[DEMO_GUIDE.md](../../DEMO_GUIDE.md). It adds a read-only capture observer and a local
browser viewer. Normal training keeps capture disabled and Replica at half
resolution unless `--resolution full` is selected explicitly.

The machine-readable checks and measured scene results are retained under
`$HOME/egas_workspace/validation`. The
[desktop completion report](desktop_validation.md) records the final
source/binary identities, test commands, scene metrics and limitations. Full
office0, room0 and TUM office runs completed at 88.37, 73.08 and 34.64 processing
FPS respectively. Use that report with each run's exact configuration and
telemetry; these are individual desktop validation observations.

The successful TUM validation run's large TSDF checkpoint is stored as a verified
lossless archive to save disk space. Its `record/RESTORE_TSDF_CHECKPOINT.txt`
contains the restoration command. Replica validation maps remain unpacked.

`SOURCE_MANIFEST.json` deliberately remains the immutable original release
manifest. It is not silently rewritten to describe desktop additions. Inspect
the desktop changes with:

```bash
git diff c944954d7000d2c905975c784f30a99ff7f3ff04
git log --oneline --decorate -5
```
