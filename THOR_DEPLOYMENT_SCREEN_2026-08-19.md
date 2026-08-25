# Thor Deployment Hardening Screen — 2026-08-19

> **Unpromoted optimization screen.** This document records a rejected
> deployment candidate, not a publication-locked comparison. The corresponding
> `*_deployed_v1` protocols now require an explicit override and are not
> paper-eligible under freeze `thor-paper-20260819-r2`.

## Decision

The minimum paper-hardening implementation is complete and validated, but no
tested deployment optimization passes every locked promotion gate. Immediate
Gaussian scheduling therefore remains the default, and the current defensible
runtime claim is average throughput rather than sustained 30 Hz operation.

This is a one-pass optimization screen, not the final five-repeat paper matrix.
All runs processed every input frame and exported a complete pre-terminal
`online_end` trajectory.

## TUM 30 Hz FIFO Results

| Configuration | Sequence | FPS | Service p95 (ms) | Max backlog (ms) | Recovery (ms) | Online ATE (cm) | Gate |
|---|---|---:|---:|---:|---:|---:|:---:|
| Unsliced reference | fr1/desk | 55.11 | 76.45 | 333.89 | 225.20 | 2.1347 | Pass |
| Unsliced reference | fr2/xyz | 38.43 | 147.99 | 944.90 | 11652.57 | 2.8956 | Fail |
| Unsliced reference | fr3/office | 42.10 | 71.84 | 1872.26 | 6189.44 | 3.5017 | Fail |
| Streaming config, immediate | fr1/desk | 55.62 | 75.63 | 325.47 | 221.56 | 2.1318 | Pass |
| Streaming config, immediate | fr2/xyz | 38.59 | 148.23 | 873.09 | 10196.21 | 2.8951 | Fail |
| Streaming config, immediate | fr3/office | 41.33 | 74.99 | 396.03 | 3312.38 | 5.0491 | Fail |
| Streaming config, sliced | fr1/desk | 66.70 | 34.30 | 296.89 | 185.46 | 2.1392 | Pass |
| Streaming config, sliced | fr2/xyz | 41.65 | 51.56 | 737.18 | 6978.43 | 2.8937 | Fail |
| Streaming config, sliced | fr3/office | 43.64 | 51.94 | 416.74 | 3577.50 | 5.0663 | Fail |

The sliced scheduler materially reduced service-time tails and improved
throughput over the matched streaming-immediate configuration, but fr2 exceeded
the 500 ms backlog limit and fr2/fr3 exceeded the one-second recovery limit.
Its three-sequence mean online ATE was 3.3664 cm, above the 2.9446 cm limit.
The matched streaming-immediate mean was 3.3586 cm, showing that the accuracy
regression comes from the streaming loop profile, not from slicing itself.

## Integrity and Hardware Envelope

The corrected sliced screen completed 6,673/6,673 frames. It exercised the
fr2 warm-candidate transition that previously exposed an active-submap
ownership defect; sliced jobs are now drained before vector growth, paging,
handoff optimization, graduation, refinement, finalization, or owner retirement.
No work is discarded.

The sliced screen ran in MAXN with CPU/GPU clocks locked and fan PWM 255.
Across 320 telemetry samples, the peak junction temperature was 48.06 °C,
mean/peak instantaneous VIN power was 57.32/83.59 W, and all three over-current
counters had zero delta. No CPU, GPU, or SoC thermal-throttle state was observed.

## Promotion Consequence

Replica promotion runs are not justified for these candidates because the TUM
gate is conjunctive and already failed. Before a paper release, retain the
frozen parity results, run the final repetition matrix only for a candidate that
first passes this screen, and complete the 30-minute hardware soak. Do not label
the present implementation sustained or hard real time.

Artifacts:

- `../thor_runs/deployment_screen_20260819/`
- `../thor_runs/deployment_screen_sliced_retry_20260819/`
- `../thor_runs/telemetry/deployment_screen_sliced_retry_20260819/`
