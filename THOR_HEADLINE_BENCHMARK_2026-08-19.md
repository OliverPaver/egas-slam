# Thor Headline Benchmark Report — 2026-08-19

> **Historical pre-freeze result.** These runs predate comparison freeze
> `thor-paper-20260819-r2` and used the binary hashes recorded below. They are
> retained as engineering evidence, but the final paper matrix must be rerun
> with the frozen binaries and five TUM repetitions; do not relabel these as
> `r2` results.

## Executive result

The locked benchmark matrix completed successfully: 29/29 run manifests report `completed/OK`. On Replica `office0` at 1200×680, the proposed submap system ran at 61.16 FPS, 28.03% faster than the faithful local GPS-SLAM baseline, while its read-only final-map render was 0.49 dB below that baseline and 0.11 dB below the GPS-SLAM paper's `office0` PSNR. On the three paper TUM sequences, the proposed tracking-only protocol achieved 3.034±0.024 cm ATE RMSE, compared with 3.08 cm published by GPS-SLAM. The full integrated submap, loop-closure, SDF, and Gaussian pipeline remained above 43 FPS on every TUM sequence by mean terminal-inclusive throughput.

The defensible conclusion is:

> On NVIDIA AGX Thor, the proposed submap and loop-closure extensions improve trajectory accuracy over a faithful GPS-SLAM baseline on the valid paired sequences, recover robust tracking where that baseline fails, and maintain near-paper rendering quality while operating above 30 FPS on average at the tested resolutions.

This is a mean-throughput result, not a hard real-time deadline guarantee.

## Test environment and provenance

| Item | Locked value |
|---|---|
| Device | NVIDIA Jetson AGX Thor, aarch64, 122.8 GiB unified memory |
| Host | JetPack 7.1 / Jetson Linux 38.4 / CUDA 13.0 driver stack |
| Container | `gps-slam-thor:pytorch-26.07-jp7.1`, image ID `sha256:45cde99a...afd71b0` |
| Runtime | PyTorch 2.13.0a0 NVIDIA 26.07, CUDA 13.3, native SM 11.0 |
| Performance state | MAXN; CPU 2.601 GHz, GPU GPC 1.575 GHz and NVD 1.692 GHz locked |
| Baseline binary | `build_legacy/slam_trainer`, SHA-256 `33c7872c...ca231`, `legacy_large` |
| Proposed binary | `build_thor/slam_trainer`, SHA-256 `2706e55e...86463b`, `medium_128k` |

The host/container CUDA combination emitted NVIDIA's minor-version-compatibility warning, but native SM 11.0 compilation, CUDA kernels, Torch, and all preflight checks passed. NVIDIA identifies Jetson Linux 38.4 as the JetPack 7.1 Thor release with CUDA 13.0 support in its [official release archive](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.1).

## Replica `office0`: full-resolution reconstruction

This locked `gps_slam_replica_paper_v1` protocol uses all 2,000 frames at 1200×680, ground-truth poses, 25% Gaussian sampling, SH degree 3, 20 optimization iterations every 10 frames, L1 loss, seed 42, and image preloading outside the timed frame loop. Values are one run per system; image-metric coverage is 2,000/2,000 in every case.

| System/output | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ | Frame p95 / p99 | Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| GPS-SLAM paper, `office0` | 41.15 | — | — | 380.72 | — | 137,200 |
| Local faithful baseline | 41.528 | 0.97566 | 0.06998 | 47.772 | 155.41 / 204.21 ms | 118,497 |
| Proposed, sequential output | 40.913 | 0.97428 | 0.07839 | 61.160 | 91.55 / 137.40 ms | 165,178 |
| Proposed, saved final map | 41.040 | 0.97448 | 0.07773 | *same timed run* | *not timed* | 165,178 |

Against the local baseline, the proposed system is +28.03% in FPS, −41.09% at p95, and −32.72% at p99. Final-map quality changes are −0.489 dB PSNR, −0.00118 SSIM, and +0.00774 LPIPS. It uses 39.39% more Gaussians. The saved-map render improves its own sequential output by 0.126 dB without further training.

| Mean timed component | Baseline | Proposed | Change |
|---|---:|---:|---:|
| SDF fusion | 5.383 ms | 6.439 ms | +19.63% |
| Local Gaussian optimization | 10.776 ms | 6.002 ms | −44.30% |
| Gaussian operations | 15.541 ms | 9.901 ms | −36.29% |

Both systems produced zero invalid poses. Their near-zero Replica ATE (0.000054 cm) is only a pose-export sanity check because this quality protocol deliberately uses ground-truth poses; it is not a tracking result.

## TUM tracking accuracy

The locked `gps_slam_tum_tracking_v1` protocol uses original 640×480 RGB-D frames, every frame, estimated poses, the same InfiniTAM tracker and failure policy, and three repetitions. Reported uncertainty is sample standard deviation across `N=3`. FPS includes the terminal pose-graph stage; corrected ATE includes accepted loop correction.

| Sequence (frames) | Baseline ATE | Proposed raw → corrected ATE | Corrected Δ vs baseline | Baseline / proposed FPS |
|---|---:|---:|---:|---:|
| `fr1/desk` (573) | 2.076±0.001 cm | 2.074±0.005 → same | −0.118% | 219.33±3.72 / 205.75±3.19 |
| `fr2/xyz` (3,615) | 2.863±0.000 cm | 2.782±0.003 → same | −2.845% | 118.85±1.25 / 129.64±1.00 |
| `fr3/office` (2,485) | **invalid**; 181.24±11.99 cm diagnostic | 7.751±0.052 → **4.246±0.081 cm** | robustness recovery; no valid percentage | invalid / 80.07±0.39 |

The mean over the two valid paired baseline scenes improves from 2.470 to 2.428 cm (−1.70%). The baseline lost tracking on 77.7–80.8% of `fr3/office` frames, so presenting a percentage improvement from its 181 cm diagnostic ATE would be misleading. The proposed system had five failed frames (0.20%) and 92 non-good frames (3.70%), then accepted one loop and reduced `fr3/office` ATE by about 45% within the run.

Corrected proposed translation/rotation RPE RMSE was 0.684±0.005 cm / 0.886±0.002° on `fr1`, 0.138±0.000 cm / 0.232±0.000° on `fr2`, and 0.251±0.002 cm / 0.354±0.000° on `fr3`.

| Sequence | Baseline p95 / p99 | Proposed p95 / p99 | Proposed terminal stage |
|---|---:|---:|---:|
| `fr1/desk` | 8.62 / 9.60 ms | 8.88 / 10.48 ms | 2.66±0.02 ms |
| `fr2/xyz` | 12.35 / 13.08 ms | 10.65 / 12.55 ms | 5.24±1.07 ms |
| `fr3/office` | invalid run; 10.53 / 12.14 ms | 17.65 / 25.41 ms | 361.02±7.32 ms (1.16% of sequence time) |

The proposed three-scene, repeat-paired mean is **3.034±0.024 cm**, 0.046 cm (1.50%) below the GPS-SLAM paper's 3.08 cm. This numerical comparison is useful but not a significance claim: the paper reports no variance and used different hardware.

## TUM integrated full-system performance

The locked `gps_slam_tum_integrated_v1` runs tracking, multi-SDF submaps, ORB+ICP loop closure, Gaussian construction, and Gaussian optimization together. All nine runs passed.

| Sequence | Corrected ATE | Terminal-inclusive FPS | p95 / p99 | Fusion / local opt / Gaussian ops | Submaps; loops |
|---|---:|---:|---:|---:|---:|
| `fr1/desk` | 2.078±0.005 cm | 79.70±0.35 | 71.46 / 82.32 ms | 5.61 / 3.97 / 6.78 ms | 1; 0 |
| `fr2/xyz` | 2.779±0.001 cm | 43.28±1.07 | 157.07 / 225.99 ms | 8.54 / 10.93 / 14.47 ms | 2; 0 |
| `fr3/office` | 3.621±0.132 cm | 46.66±0.08 | 71.02 / 96.39 ms | 11.76 / 3.70 / 5.91 ms | 28; 1 |

The `fr3` final stage cost 341.43±8.64 ms (0.64% of sequence time) and reduced raw ATE from 7.717±0.033 to 3.621±0.132 cm. Corrected translation/rotation RPE was 0.683±0.004 cm / 0.885±0.002°, 0.138±0.000 cm / 0.232±0.000°, and 0.252±0.003 cm / 0.354±0.000° respectively. The repeat-paired three-scene ATE mean is 2.826±0.043 cm; this is operational consistency evidence, not the primary paper comparison, because there is no matching integrated baseline arm.

All means exceed 30 FPS, but `fr2/xyz` has 157 ms p95 and 226 ms p99 frames. The system therefore meets average 30 Hz throughput, not an every-frame 33.3 ms deadline.

## Comparison with the GPS-SLAM paper

The paper used an RTX 4090 24 GB and AMD 9950X3D, reports 20 optimization iterations every 10 frames, and ran competing implementations on that same desktop. Its exact `office0` Table 2 result is 41.15 dB, 137,200 Gaussians, and 380.72 FPS; the proposed Thor result is 41.040 dB, 165,178 Gaussians, and 61.16 FPS. Quality is 0.11 dB lower, Gaussian count is 20.39% higher, and Thor throughput is 6.22× lower. These are cross-hardware observations, not an efficiency ranking. See the [GPS-SLAM paper and Tables 1–3](https://arxiv.org/html/2509.11574v1).

The paper's Replica-wide averages—37.24 dB PSNR, 0.960 SSIM, 0.103 LPIPS, and 252.64 FPS—aggregate multiple scenes and must not be compared directly with this single-scene `office0` run. Its TUM value is directly aligned at the scene-set/metric level: unweighted mean ATE RMSE over `fr1/desk`, `fr2/xyz`, and `fr3/office`.

## Fairness and calculation rules

- The protocol registry locks dataset, resolution, frame count, pose source, timing boundary, and role-specific feature state. Each manifest hashes the generated YAML, binary, dataset inventory, evaluation artifacts, container image, and clock state.
- ATE is actual translational RMSE after rigid SE(3) alignment—rotation and translation, no scale—with complete one-to-one pose coverage. RPE uses one-frame relative transforms after the same rigid alignment.
- TUM reports mean±sample SD across three runs. All repeats use seed 42, so they measure execution/lifecycle repeatability, not seed sensitivity.
- Input preloading is outside the frame timer for both arms. Online pose-graph work is already inside per-frame timings; the explicit end-of-sequence candidate scan is added to terminal-inclusive FPS. It is not hidden offline work.
- The final Replica map is rendered read-only from the state built sequentially online. No quality finalization, frozen-map refinement, Gaussian graduation, or offline model optimization is enabled. PSNR, SSIM, and VGG-LPIPS use all 2,000 filename-matched RGB frames with cached weights and no depth mask.
- The TUM tracking arm intentionally omits Gaussians for both baseline and proposed systems. The paper states that Gaussian reconstruction does not affect SDF reconstruction, so this isolates trajectory behavior. The integrated arm separately verifies end-to-end speed and accuracy with Gaussians enabled.
- The local baseline is a faithful legacy executable, not a one-variable ablation. Its `legacy_large` SDF hash profile differs from the proposed `medium_128k` multi-map profile. Claims should therefore be framed as system-level baseline versus proposed system, not as causal attribution to one feature.

For comparisons with other papers, only put values in the same table when sensor modality, sequence set, resolution, pose source, split, metric alignment, and aggregation match. Cross-hardware FPS should be labeled by device and treated as contextual. A fair same-platform speed ranking requires running each official implementation on Thor; this benchmark does not claim such a ranking.

## Power, memory, and validation notes

One over-current event was observed during the separate Replica final-map renderer, after the timed SLAM loop and sequential quality evaluation. Temperatures were approximately 59–61°C with no thermal-trip flags. Fan duty was then locked to 255/255; all TUM matrices were bracketed by unchanged over-current counters. Electrical current limiting is distinct from thermal throttling, so maximum fan does not prevent it. It can affect timing, but not the deterministic rendered images; no affected final-map rendering time is reported.

The logged “GPU memory usage” is device-wide unified-memory occupancy on Thor, not per-process dedicated VRAM or a peak allocator measurement. It ranged from about 53–81 GB in tracking runs and 55–113 GB in integrated runs, increasing with run order/page cache. It is unsuitable for comparison with the paper's discrete-GPU memory values. Future memory claims need isolated process RSS plus CUDA allocator peaks under a cold-cache protocol.

Post-run regressions passed: tracking policy 12/12, validation hardening, metric hardening 9/9, benchmark hardening 5/5, and the robust pose-graph solver regression. Benchmark preparation changed validation/protocol wiring only; it did not tune the SLAM algorithm during these runs.

## Artifacts

- Output root: `../thor_runs/headline_2026-08-19/`
- Replica summaries: `replica_baseline/canonical_summary_*.csv`, `replica_modified/canonical_summary_*.csv`
- TUM summaries: `tum_baseline/headline_20260819_summary.csv`, `tum_modified/headline_20260819_summary.csv`, `tum_integrated/headline_20260819_summary.csv`
- Repeat statistics: each TUM directory's `*_stability.csv`, `*_stability_groups.csv`, and `*_stability.json`
- Reproducibility record: every run directory contains `benchmark_manifest.json`; generated locked YAML files are retained under each TUM `configs/` directory and `configs/experiments/` for Replica.
