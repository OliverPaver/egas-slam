# Thor Paper Hardening Protocol

## Scope

Paper-parity protocols remain behaviorally unchanged. Thor deployment features are opt-in through `gps_slam_tum_deployed_v1` and `gps_slam_replica_deployed_quality_v1`; they do not reduce resolution, skip frames, cap Gaussians, add offline refinement, or alter the asynchronous architecture.

## Implemented Runtime Contract

`timing.csv` now records complete algorithmic service time, pose-ready latency, online pose-graph latency, and sliced-Gaussian job state. The deployed TUM protocol exports `pose_graph_live/online_end_poses/` before terminal pose-graph work. Only this trajectory is eligible for deployed accuracy claims.

`scripts/analyze_realtime_trace.py` deterministically replays service times against original TUM RGB timestamps and normalized 30 Hz arrivals. It writes `realtime_trace.csv` and `realtime_summary.json` with throughput, latency percentiles, deadline misses, queue depth, backlog, recovery, continuity, and expected-frame completeness. The sustained gate permits individual deadline misses; it requires at least 30 FPS, all frames, final backlog ≤33.3 ms, maximum backlog ≤500 ms, and recovery ≤1 s.

The opt-in scheduler is:

```yaml
PIPE:
  gaussian_schedule:
    mode: sliced
    iterations_per_frame: 2
```

It preserves 20 deterministic optimization steps per 10-frame interval. Inputs and camera order are retained for the job, cleanup follows iteration 20, and lifecycle/optimizer conflicts force a drain or error. `immediate` remains the default.

## Thor Measurement

Run headline commands through `scripts/thor_container.sh telemetry-run`. It records process RSS, tegrastats power/temperature/RAM, MAXN, clock locks, fan PWM, over-current deltas, and sampled thermal cooling/throttle states. With the default clean requirement, an otherwise successful command exits nonzero if the hardware envelope is invalid.

```bash
GPS_SLAM_TELEMETRY_DIR=/absolute/host/output \
  bash scripts/thor_container.sh telemetry-run COMMAND...
```

For the one-pass three-configuration screen of every prepared TUM sequence:

```bash
bash scripts/thor_container.sh telemetry-run python3 run/thor_deployment_screen.py \
  --tum-root /workspace/transfer/datasets/tum_rgbd/processed \
  --output-root /workspace/runs/deployment_screen \
  --trainer build_thor/slam_trainer --run --quiet-run-logs
```

The screen fixes: unsliced robust reference; streaming loop/paging with immediate optimization; and streaming loop/paging with sliced optimization. Use `--screens combined_sliced_streaming` (or another screen name) to rerun only one arm without replacing completed results.

## Promotion and Validation

Run `bash scripts/thor_container.sh regressions`, then evaluate completed TUM, Replica, and telemetry summaries with `scripts/evaluate_deployment_promotion.py`. It enforces the approved ATE/RPE, 30 Hz queue, Replica quality, Gaussian-count, coverage, and clean-hardware thresholds. A candidate is rejected on missing or non-finite inputs.

The final 35-frame CUDA smoke completed 3/3 sliced jobs (60/60 iterations), exported 35/35 online poses, and reported 39.58 processing FPS, 297.06 ms maximum backlog, zero final backlog, and 191.09 ms recovery. Its matched immediate smoke reported 37.22 FPS and 329.00 ms maximum backlog. Final Gaussian counts differed by 0.36%, and the largest elementwise difference between exported 4×4 pose matrices was 0.000213. MAXN, locked clocks, maximum fan, thermal state, and over-current counters remained clean. This is implementation validation, not a paper benchmark or promotion result.

The full one-pass TUM screen is recorded in `THOR_DEPLOYMENT_SCREEN_2026-08-19.md`. Slicing improved throughput and service-time tails, but no arm passed every latency and accuracy gate; immediate scheduling therefore remains the default and no optimization was promoted. The frozen results in `THOR_HEADLINE_BENCHMARK_2026-08-19.md` predate these deployment protocols. Five-repeat TUM matrices, eight-scene Replica matrices, and the 30-minute soak remain required before changing the paper claim.
