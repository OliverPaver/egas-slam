# Paper Readiness Lock — 2026-08-25

## Release Decision

The comparison implementation is locked at freeze
`thor-paper-20260819-r2`. Source, `build_thor/slam_trainer`,
`build_legacy/slam_trainer`, build profiles, and the container image all verify
against `BENCHMARK_FREEZE.json`. No functional source change was made after the
61-run headline matrix.

This release is ready for reproducible parent-system and same-platform external
comparisons. Its defensible runtime claim is **full-pipeline average throughput
above 30 FPS on the tested proposed-system sequences**, not sustained 30 Hz or
an every-frame 33.3 ms deadline guarantee.

## Frozen Headline Evidence

The R2 matrix contains 61 completed manifests: 45 TUM runs (three arms, three
sequences, five repetitions) and 16 Replica runs (two arms, eight scenes). All
five faithful-baseline `fr3/office` runs are intentionally rejected by the
tracking acceptance policy: the raw tracker fails on 80.80% of frames. This is
a deterministic single-map baseline limitation, not missing output; changing
its tracker or containment policy would cease to be a faithful baseline.

### TUM RGB-D accuracy and throughput

Values are mean ± sample standard deviation over five same-seed repetitions.
ATE and RPE use complete pose pairing and rigid SE(3) alignment without scale.

| Arm | `fr1/desk` ATE | `fr2/xyz` ATE | `fr3/office` ATE | Three-scene mean | Terminal-inclusive FPS (`fr1/fr2/fr3`) |
|---|---:|---:|---:|---:|---:|
| Faithful GPS-SLAM baseline | 2.0755±0.0040 cm | 2.8645±0.0008 cm | invalid (177.1 cm diagnostic) | not reportable | 222.82 / 124.91 / invalid |
| Proposed, tracking isolation | 2.0760±0.0029 cm | 2.7813±0.0032 cm | 4.2655±0.0799 cm | 3.0409 cm | 208.02 / 130.68 / 82.10 |
| Proposed, integrated Gaussians | 2.0746±0.0019 cm | 2.7794±0.0007 cm | 3.6578±0.0953 cm | 2.8372 cm | 79.38 / 42.67 / 47.10 |

On the two valid paired baseline scenes, tracking-isolation ATE improves from
2.4700 to 2.4286 cm (1.68%). The proposed method also completes `fr3/office`
with 0.20% raw failed frames and applies an accepted loop correction; the
faithful baseline does not provide a valid paired result there. The published
GPS-SLAM three-scene mean is 3.08 cm; the aligned tracking-isolation result is
3.0409 cm (0.0391 cm, or 1.27%, lower). This is a numerical reference, not a
significance claim because the paper provides no repeat variance and used
different hardware. The integrated result is supporting end-to-end evidence
rather than the primary paired accuracy comparison.

### Replica reconstruction quality

Each scene uses all 2,000 frames at 1200×680, frame step 1, ground-truth poses,
the original L1 loss and 20-iteration/10-frame Gaussian schedule, and complete
2,000-image metric coverage. The proposed value is the read-only final-map
render of online state; it performs no terminal optimization, reintegration,
refinement, or Gaussian insertion.

| Arm | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ | Gaussians |
|---|---:|---:|---:|---:|---:|
| Faithful local baseline | 37.6253 | 0.96139 | 0.10265 | 33.46 | 303,322 |
| Proposed submap system | 37.1495 | 0.95691 | 0.11141 | 51.21 | 182,257 |
| GPS-SLAM paper reference | 37.24 | 0.960 | 0.103 | 252.64 (desktop) | — |

The proposed system is 53.1% faster than the local faithful baseline on mean
Thor FPS and uses 39.9% fewer Gaussians. Its quality delta is −0.476 dB PSNR,
−0.00448 SSIM, and +0.00875 LPIPS. Its mean PSNR is 0.091 dB below the paper
reference; the corresponding paper-reference deltas are −0.00309 SSIM and
+0.00841 LPIPS. Paper FPS was measured on a desktop RTX 4090 and is contextual
only, not a cross-hardware efficiency ranking. Proposed per-scene Replica
throughput ranges from 39.24 to 72.70 FPS; the minimum integrated TUM scene
mean is 42.67 FPS.

Replica process peak RSS averages 38.13 GiB for the baseline and 40.27 GiB for
the proposed method; the proposed maximum is 40.84 GiB. The maximum integrated
TUM process peak is 29.02 GiB. These values are below Thor's 122.8 GiB shared
memory but are not comparable to dedicated desktop VRAM. The locked parity
protocol deliberately preloads Replica input outside the timer; it is a fair
paired timing policy, not the minimum-memory deployment mode.

## Optimization Investigation

The opt-in sliced Gaussian scheduler remains implemented and regression-tested,
but is not promoted. The predeclared 30 Hz FIFO screen processed every frame
and improved throughput/tails, yet `fr2/xyz` exceeded the 500 ms backlog limit,
`fr2` and `fr3` exceeded one-second recovery, and three-scene online ATE exceeded
the promotion threshold. Immediate scheduling therefore remains locked.

A final low-risk quality candidate was tested without modifying source. A
320-frame full-resolution perceptual profile completed at 23.72 FPS with
35.663 dB PSNR, 0.95735 SSIM, and 0.11384 LPIPS. A less invasive 2,000-frame
room0 candidate retained the locked Gaussian work but used `ssim_weight: 0.2`
and on-demand embedded residency. Its complete online trace measured 23.46 FPS,
226.5 ms p95 service time, 211,218 Gaussians, and zero missing frames. It was
rejected before promotion because it failed the 30 FPS gate. Relative to the
frozen room0 trace, mean fusion increased from 11.29 to 21.90 ms on the
on-demand path and mean Gaussian work increased from 13.67 to 20.17 ms with
DSSIM. This accounts for the slowdown without indicating a tracking, submap, or
loop-closure regression. Moving/deeper-buffering the prefetch or optimizing the
DSSIM kernel would be a new implementation revision. It would invalidate the
freeze and require repeating the headline matrix; the evidence does not justify
that risk before external comparison. Because the 2,000-frame candidate was
already ineligible, its subsequent image-metric stage was stopped; its trace is
engineering timing evidence, not a complete publication result.

## Release Validation

`bash scripts/thor_container.sh regressions` passes the tracking policy,
validation, metrics, benchmark freeze, protocol/manifest, real-time trace,
promotion, and pose-graph solver suites. Freeze verification reports source and
binaries clean and the container image matched.

An independent artifact audit found zero coverage or metric-status issues in
all 16 Replica rows. All 45 TUM rows contain the expected number of timing,
tracking, estimated-pose, and matched-pose records; 40 pass acceptance and the
only five rejections are the declared faithful-baseline `fr3/office` failures.

The R2 headline commands accumulated 8.30 hours of telemetry-bracketed
execution. All clocks were locked, MAXN and fan PWM 255 were maintained, all
over-current counter deltas were zero, and no CPU/GPU/SoC thermal-throttle state
was observed. Peak junction temperature was 52.812 °C and peak instantaneous
VIN power was 122.134 W.

The canonical continuous soak then launched the exact locked integrated
`fr3/office` config from a clean workspace 24 consecutive times. The command
ran for 42.98 minutes and returned zero; `set -e` made any failed repetition
fatal. All 59,640 frames completed. Throughput averaged 47.454 FPS over the 24
maps (46.990–47.936 FPS); the final pass was 47.427 FPS, 0.93% above the first.
The late-half mean was 0.28 FPS above the early-half mean. Gaussian count
averaged approximately 326,095 (324,990–327,802); the final count was 0.035%
below the first. The rolling workspace bounded disk use without reusing map
state—each trainer process cleared the prior workspace and reconstructed the
sequence from frame zero. Per-repeat FPS and Gaussian counts are retained in
[`docs/paper_soak_r2_20260825.csv`](docs/paper_soak_r2_20260825.csv).

Across 5,127 soak telemetry samples, mean/peak instantaneous VIN was
47.48/73.41 W, mean/maximum junction temperature was 42.28/47.72 °C, maximum
GPU temperature was 44.0 °C, and mean/maximum used system memory was
38,239/49,541 MB. The final process peak RSS was 28.88 GiB. MAXN, locked clocks,
and fan PWM 255 remained asserted; every over-current counter delta and every
CPU/GPU/SoC thermal-throttle state was zero. The authoritative telemetry is
`../thor_runs/paper_soak_r2_20260825/telemetry_continuous/`.

## Rules for External Comparisons

- Run official external implementations on the same Thor, power mode, clocks,
  fan state, container/JetPack stack where possible, datasets, frame ranges,
  resolutions, and sensor rate. Report any unavoidable porting changes.
- Compare TUM ATE only with estimated-pose RGB-D runs using the same sequences,
  alignment, aggregation, and complete-frame policy. Do not compare Replica GT
  pose quality with tracking accuracy.
- Compare Replica quality only at 1200×680 over all 2,000 matched views using
  the same PSNR, SSIM, and VGG-LPIPS implementation and split.
- Report tracking-only FPS, integrated algorithmic FPS, terminal work, metric
  rendering, and input loading separately. Do not relabel average throughput as
  hard or sustained real time.
- Preserve failed runs and acceptance blockers. Do not exclude scenes because
  a baseline loses tracking, and do not turn the resulting diagnostic ATE into
  a percentage-improvement claim.

## Canonical Artifacts

- Freeze: `BENCHMARK_FREEZE.json`
- Protocol: `BENCHMARK_PROTOCOL.md`
- Full R2 matrix: `../thor_runs/headline_r2_2026-08-24/`
- Rejected deployment screen: `THOR_DEPLOYMENT_SCREEN_2026-08-19.md`
- Rejected quality screen: `../thor_runs/r3_quality_screen_20260825/`
- Continuous soak: `../thor_runs/paper_soak_r2_20260825/`

External runs must start with `bash scripts/thor_container.sh source-check`, a
machine-local freeze verification as described in `REPRODUCIBILITY.md`, and
`bash scripts/thor_container.sh regressions`. Any functional edit requires a
new reviewed freeze revision and invalidates direct use of the R2 results.
