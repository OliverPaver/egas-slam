# Thor Paper Comparison Freeze — 2026-08-19

## Status

The implementation is frozen for comparison-only work as
`thor-paper-20260819-r2`. The source tree, build profiles, binaries, AArch64
platform, and exact container image are machine-verified before any locked
benchmark manifest can be written. This establishes a stable implementation;
it does **not** make the remaining full benchmark matrix complete.

## Frozen Identity

| Item | Frozen value |
|---|---|
| Functional files | 3,998 |
| Functional digest | `aa9b8a5ff9b1678d75f852465f6eb9ee5d77ea260708e7b6fae8d70fe9248785` |
| Freeze manifest SHA-256 | `5db75de5f471ca0da922d861ccd5270b9dc7f93a5fa016163b05331460ea68de` |
| Faithful baseline | `build_legacy/slam_trainer`, `fb7ec61b...75992`, `legacy_large` |
| Proposed system | `build_thor/slam_trainer`, `e1114f0d...df7f1`, `medium_128k` |
| Container image ID | `sha256:45cde99a...afd71b0` |
| Runtime target | Native AArch64, CUDA SM 11.0; Docker is not emulation |

Both executables were rebuilt from the same final instrumented source. The
baseline keeps original-capacity BasicEngine behavior; the proposed binary
uses the Thor-feasible multi-map capacity. Both export the same hardened
timing and resource schema.

## Publication-Locked Comparisons

Only these protocols are paper-eligible:

- `gps_slam_replica_paper_v1`
- `gps_slam_tum_tracking_v1`
- `gps_slam_tum_integrated_v1`

They preserve immediate Gaussian optimization, parity input behavior, and the
locked tracking/quality definitions in `BENCHMARK_PROTOCOL.md`. TUM paper runs
require full sequences and at least five repetitions. The two deployed/sliced
protocols remain reproducible but are explicitly unpromoted and rejected by
default because they failed the deployment gate.

## Completed Freeze Validation

- Container and host freeze verification: clean.
- Regression suites: tracking 12/12, metrics 9/9, freeze 3/3, benchmark 6/6,
  real-time 4/4, promotion 3/3, plus validation and pose-graph checks.
- Locked Replica baseline/proposed and five-repeat integrated TUM configuration
  generation: passed preflight with clean embedded provenance.
- Instrumented `legacy_large` baseline smoke: 35/35 frames, valid ATE/RPE,
  timing/resource/FIFO artifacts, no blocker, NaN, or CUDA error.
- The TUM safety estimator was verified to run after protocol locking; it now
  reports the actual `medium_128k` profile rather than the template default.

Validation artifacts are under
`../thor_runs/freeze_validation_r2_20260819/`.

## Change Control

Before every comparison, run:

```bash
bash scripts/thor_container.sh freeze-check
bash scripts/thor_container.sh regressions
```

Any functional edit, added/removed functional file, binary rebuild, CMake
profile change, architecture change, or container replacement makes the check
fail and blocks locked manifests. A necessary bug fix must be reviewed,
revalidated, assigned `r3` (or later), and all affected comparison arms rerun.
Documentation and result summaries may be updated without changing the
functional digest. Earlier headline runs remain historical evidence and are
not retroactively relabeled as `r2` results.

Because this transfer has no Git repository, a recoverable 4,001-entry snapshot
is stored at `../GPS-SLAM_thor-paper-20260819-r2.tar.zst`. Its SHA-256 is
`e377620a523632ddf9f0f604bc92d406f33ab5162c87416812614d81420bcce0`;
verify it from the transfer root with:

```bash
sha256sum -c GPS-SLAM_thor-paper-20260819-r2.tar.zst.sha256
```
