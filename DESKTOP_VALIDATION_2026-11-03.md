# EGAS-SLAM desktop setup and validation — 3 November 2026

The final Thor implementation is built and runnable locally in
`/home/oliver/EGAS-SLAM`. Full runs of Replica office0, Replica room0 and TUM
fr3_office completed successfully with estimated camera poses. Saved-map
rendering was checked for both Replica scenes. This checkout is ready for
continued desktop development; these observations are desktop validation, not
a replacement for the historical Thor benchmark.

## Source and environment

| Item | Retained identity |
| --- | --- |
| GitHub repository | `OliverPaver/gps-slam-submaps-thor` |
| Verified final upstream commit | `c944954d7000d2c905975c784f30a99ff7f3ff04` |
| Upstream tree | `50de2f460a0b342564407b61aaeb01dd42b8c805` |
| Local development branch | `desktop/rtx3070-2026-11-03` |
| Validated executable SHA256 | `92a20fbec8949fefe3d43854553110262a1a4b91053064895422a3d326c47835` |
| CPU / system RAM | Ryzen 7 7700X / 32 GB |
| GPU | RTX 3070, 8 GB, SM 86 |
| Compiler / CUDA | GCC 13.3 / CUDA 12.8.93 |
| Torch / OpenCV | PyTorch/LibTorch 2.10.0+cu128, CXX11 ABI / OpenCV 4.6 |
| Build | Release, `medium_128k`, fast math OFF, TensorBoard OFF, legacy UI OFF |

Live GitHub main matched the final source archive, including the accepted
saved-map overlap restoration. The archive was copied off the SanDisk before
extraction. All 3,966 source-manifest entries passed before desktop changes;
the original manifest and complete Git history are preserved. Source, build,
dependencies, datasets and execution outputs used for these tests are local.
Nothing was built or run on the external drive.

The validated executable was built before the local desktop commit. Its embedded
revision therefore identifies the upstream commit; it also contains the CMake
portability changes described below. Run receipts record the dirty patch and
binary hashes rather than pretending that this was an untouched upstream build.
The exact build identity, dynamic dependencies and source verification are in
`/home/oliver/egas_workspace/setup/build_identity.json` and
`/home/oliver/egas_workspace/setup/source_verification.txt`.

## Changes made for this PC

1. Added a native desktop build script using the existing CUDA-compatible Torch
   installation and SM 86. Vendored libraries were rebuilt for x86-64; archived
   ARM objects were discarded only from this new checkout's ignored build areas.
2. Stripped the newline from `VERSION` before using it in CMake compiler flags.
   This fixes malformed Unix Makefiles without changing SLAM behaviour.
3. Added an optional host-C++-runtime link-order correction, with a small native
   filesystem regression. It resolves a demonstrated LibTorch/runtime conflict
   during checkpoint cleanup. The option is enabled by the desktop build and
   defaults OFF for other build workflows.
4. Added desktop configuration overlays, a launcher, data preparation, telemetry
   and a results checker. Original `configs/final/` files remain unchanged.
5. Prepared Replica office0 and room0 at 600 × 340 with matching intrinsics
   `[300, 300, 299.75, 169.75]` and reader downscale factor 1. Both the SDF engine
   and Gaussian camera now receive the same geometry. Setting only the reader
   downscale to 2 would leave the frozen SDF engine calibrated for full size.

The half-resolution copies use the inherited preparation method: area-resampled
and JPEG-encoded colour, nearest-neighbour depth, unchanged depth units and
unchanged reference poses. Full-resolution inputs remain available locally.
Raw data for all eight Replica scenes is present; the launcher can prepare the
others when requested.

No SLAM algorithm source was modified. Coupled submaps, loop verification,
guarded pose-graph updates, corrected rendering, immediate Gaussian scheduling
and CPU-backed SDF paging retain their final implementation. Gaussian paging
remains disabled in the final fresh-training profiles. The CPU SDF cache's
nominal 8 GiB budget is not a cap on total process memory.

The launcher allocates a fresh output directory for each run, rejects reused
names and external-drive paths, checks input coverage and calibration, and
retains the effective YAML outside the trainer's clearable map folder. Evaluation
verifies that the saved training configuration still matches its recorded hash.

## Complete-scene results

All rows below use the final repaired executable. Each represents one complete
validation run, without pacing to sensor timestamps. Each run returned exit code
0 and produced the full configured trajectory and diagnostic frame coverage.

| Scene | Resolution | Frames | Processing FPS | Whole command (s) | Final-corrected ATE RMSE (cm) | Saved-map RGB PSNR (dB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Replica office0 | 600 × 340 | 2,000 | 88.37 | 25.33 | 0.174 | 41.30 |
| Replica room0 | 600 × 340 | 2,000 | 73.08 | 32.15 | 0.173 | 34.69 |
| TUM fr3_office | 640 × 480 | 2,485 | 34.64 | 187.20 | 3.803 | Not measured |

Processing FPS uses native algorithmic sequence timing, including terminal
algorithm completion and CUDA/paging synchronisation. It excludes process
startup, map export and separate online-end trajectory export. Whole-command
time includes startup and checkpoint export; it excludes the later rendering
evaluation. TUM's approximately 13 GiB uncompressed TSDF checkpoint makes export
substantial: its algorithmic sequence time was 71.74 seconds, while the command
took 187.20 seconds. Use `--no-save` for development runs that do not need a
reusable map.

ATE uses a single rigid SE(3) alignment, with no fitted scale, against the exact
configured reference-pose support. The TUM raw trajectory had 7.589 cm ATE,
online-end corrected history had 3.816 cm, and terminal-corrected history had
3.803 cm. Those are different exports of one run, not an independent loop-OFF
versus loop-ON ablation.

RGB PSNR is the mean per-view PSNR over 50 uniformly selected observed views,
using all RGB pixels at half resolution. Both saved maps loaded and rendered
all 50 requested views successfully. This is not novel-view evaluation or the
historical Thor 500-view protocol; SSIM, LPIPS and dense-surface metrics were
not measured in this desktop setup pass.

### Tracking, paging and memory observations

| Scene | Raw GOOD / POOR / FAILED frames | Nonempty Gaussian submaps | Sampled device memory maximum (MiB) | Sampled process RSS maximum (MiB) |
| --- | --- | ---: | ---: | ---: |
| Replica office0 | 2,000 / 0 / 0 | 1 | 2,193 | 1,924.7 |
| Replica room0 | 1,888 / 84 / 28 | 5 | 2,766 | 4,026.2 |
| TUM fr3_office | 2,398 / 82 / 5 | 27 | 5,908 | 11,947.6 |

The final profiles preserve the `legacy_ignore` tracking policy. Its policy-level
GOOD values must not be interpreted as absence of raw tracking failures. The
table reports the raw primary tracker states. Full pose coverage, finite values
and rigid rotation matrices were checked despite those intermittent failures.

TUM exercised 122 SDF page-outs, 99 page-ins and nine CPU-cache spills. It applied
two graph updates and ended with no pending transform updates. Room0 exercised
three page-ins; office0 required no paging events. Neither Replica run applied
a loop correction, so office0 alone would not have validated the loop/paging
paths.

Memory telemetry samples the native process's RSS and the entire GPU once per
second. Device figures include display/other applications and can miss brief
peaks; they are not allocation-accounting peaks or the thesis's cgroup metric.
The tested workloads completed within the machine's available memory without
an out-of-memory error.

The average rates do not establish hard real time. The 95th-percentile recorded
algorithmic service times were 42.03, 50.74 and 57.34 ms respectively. The tests
were not repeat-controlled benchmarks, the desktop was not isolated, and
telemetry overhead was not independently measured. Different resolution,
preparation, hardware and Torch version prevent direct speed comparisons with
the Thor thesis results.

## Native correctness checks and repaired failure

The first TUM run completed frame processing but crashed during recursive
removal of a nonempty runtime paging folder after map export. It was retained
as a failed diagnostic run and excluded from the result table.

GDB traced the failure through `SLAMPipeline::cleanupRuntimePages()` into this
LibTorch package's exported `std::filesystem::remove_all`. That implementation
called a weak recursive-iterator function absent from the host runtime, causing
a null call. A CPU-only nested-directory reproducer failed with the original
library order and passed with the host `libstdc++` placed first. Both filesystem
removal overloads are now checked against the trainer's transitive libraries.
The actual executable dependency order was verified, and the complete TUM run
then passed export, nonempty paging cleanup and shutdown. The repair changes
link resolution, not the paging algorithm.

Additional checks passed:

- Tracking-failure policy and capacity-profile regression suites (12 cases each).
- Robust pose-graph solver regression checks.
- CUDA prefix regression over seven sizes.
- The unchanged CUDA reduction helpers over 18 scalar cases and a three-component
  case, each across 257 blocks.
- CUDA helper tests under Compute Sanitizer memcheck, racecheck and synccheck:
  zero reported errors or race hazards. This covers the focused helpers, not
  every full application path under a sanitizer.
- Launcher path, calibration, frame-coverage and evaluation-overlay checks;
  altered saved training configurations were rejected before GPU execution.
- Shell syntax, Python compilation and Git whitespace checks.

Evidence is retained in `~/egas_workspace/validation/focused_native/`,
`filesystem_probe/`, `eval_config_guard/` and the TUM crash diagnostic files.
Earlier runs, including the failed attempt and a cancelled evaluation, are
distinct from the successful `validated_*` runs. The earlier, slightly higher
Replica FPS values were not selected for this report.

## Storage and retained evidence

Cleanup reclaimed **40.80 GiB** from older generated artifacts: 32.45 GiB of
obsolete run payloads and five build directories, plus 8.35 GiB by losslessly
archiving older per-frame pose files. Every archived pose file was checked by
SHA256 before removing its loose original. Existing source worktree status was
unchanged. Datasets, source implementations, final Thor evidence and thesis
files were retained. Detailed receipts are under `~/egas_workspace/setup/`.

The successful TUM checkpoint was then compressed separately to
`runs/validated_tum_office/map/tsdf_engine.tar.zst`. All 169 files were checked
by size and SHA256 after decompression before the loose directory was removed.
This saves another 9.01 GiB relative to leaving the new checkpoint unpacked.
Root has approximately **60 GiB available** after setup and validation. Restore
the exact TUM TSDF directory before any workflow that needs its unpacked files:

```bash
tar --zstd -xf /home/oliver/egas_workspace/runs/validated_tum_office/map/tsdf_engine.tar.zst \
  -C /home/oliver/egas_workspace/runs/validated_tum_office/map
```

Its checksum and member verification are in that run's `record/checkpoint_archive.json`
and `record/tsdf_engine_archive_files.json`. Replica checkpoints remain unpacked.

## Continue development

```bash
cd /home/oliver/EGAS-SLAM
./egas build
./egas train --dataset replica --scene office0
./egas train --dataset replica --scene room0 --no-save
./egas train --dataset tum --scene fr3_office --no-save
```

Use a fresh name for every run, optionally with `--name my_new_run`. Preparation
for another local Replica scene is `./egas prepare --scene office1`. Rendering
and summary examples, environment details and configuration notes are in
[DESKTOP_SETUP.md](DESKTOP_SETUP.md) and
[configs/desktop/README.md](configs/desktop/README.md).

The canonical desktop validation summary is
`/home/oliver/egas_workspace/validation/desktop_run_summary_final.json`.
The three run folders are `validated_replica_office0`,
`validated_replica_room0` and `validated_tum_office` under
`/home/oliver/egas_workspace/runs/`. Their `record/` directories retain exact
configuration, native log, run receipt, hashes, telemetry and validation summary.
No required setup work remains. Other scenes and the optional interactive
viewer have not been validated in this pass; the headless trainer and saved-map
rendering are the tested development workflow.
