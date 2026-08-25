# Scripts

This directory contains dataset preparation, evaluation metrics, and offline
diagnostics. Metric scripts are intentionally kept separate from the timed SLAM
run unless a runner explicitly invokes them after training.

## Core Metrics

| Script | Purpose |
| --- | --- |
| `metric_general.py` | Full image-quality metrics: PSNR, SSIM, LPIPS. |
| `metric_sampled_eval.py` | Sampled image-quality metrics for long or memory-sensitive validation. |
| `ate_general.py` | Absolute trajectory error with separately labelled mean and true RMSE values. |
| `rpe_general.py` | Relative pose error for loop-closure and drift analysis. |
| `geo_general.py` | Geometry accuracy, completion, and F1 against a ground-truth mesh. |

## Dataset Preparation

| Script | Purpose |
| --- | --- |
| `replica_preprocess.py` | Original hardcoded Replica conversion helper. |
| `replica_prepare_scene.py` | Repeatable single-scene Replica preparation, including optional half-resolution output. |
| `tum_preprocess.py` | Download, associate, and convert TUM RGB-D sequences into GPS-SLAM layout without unfair filtering. |

## Diagnostics

| Script | Purpose |
| --- | --- |
| `analyze_hybrid_submap_run.py` | Summarize submap events, render diagnostics, budgets, and timing for one run. |
| `compare_hybrid_submap_runs.py` | Diff two hybrid runs by config, events, stats, and metric rows. |
| `analyze_submap_handoff.py` | Relate accepted SDF/Gaussian handoffs to trajectory drift. |
| `analyze_tum_loop_opportunities.py` | Offline GT-only estimate of loop opportunities for benchmark planning. Not used by runtime loop detection. |
| `tum_loop_gt_audit.py` | Offline GT revisit summary for TUM scene selection. |
| `compare_tum_run_stability.py` | Compare repeated TUM loop-closure runs for deterministic behavior. |
| `summarize_tum_runs.py` | Aggregate TUM run outputs into compact comparison tables. |
| `pose_graph_solver_regression.py` | Build and run a small MiniSlamGraph regression for robust loop-edge handling. |
| `metrics_hardening_regression.py` | Dependency-light checks for metric pairing, device selection, pose validity, deterministic geometry sampling, and failure artifacts. |
| `analyze_realtime_trace.py` | Replay per-frame service times against TUM timestamps and normalized 30 Hz FIFO arrivals. |
| `thor_telemetry.py` | Wrap a command with Thor power, temperature, fan, clock, memory, and throttle capture. |
| `evaluate_deployment_promotion.py` | Enforce the locked TUM, Replica, Gaussian-count, and hardware promotion gates. |
| `benchmark_freeze.py` | Create or verify the immutable functional-source, binary, build-cache, architecture, and container envelope used by paper comparisons. |
| `benchmark_freeze_regression.py` | Check that freeze verification detects modified, missing, and added functional files. |
| `realtime_hardening_regression.py` | Check FIFO, recovery, completeness, malformed timing, and telemetry parsing. |
| `promotion_hardening_regression.py` | Check acceptance and rejection at deployment gate boundaries. |

## Evaluation Contracts

Image scorers use `--device auto` by default and accept `cpu`, `cuda`, or an
explicit `cuda:N`. Render frames must have all required references; pass
`--allow-partial` only for an intentional intersection-only audit. Successful
image runs record pairing coverage and the selected device in `results.json`.
Failures remove stale results and write `metric_error.json` with a nonzero exit.

ATE and RPE similarly reject duplicate, malformed, non-finite, or unmatched
estimated poses by default and write `pose_eval.json` or `rpe_eval.json`.
Geometry sampling defaults to deterministic seed `0`; record any `--seed`
override alongside results. It requires SciPy and the pinned pure-Python
`trimesh` package; Open3D is not required. Run the focused contract checks with:

```bash
python3 scripts/metrics_hardening_regression.py
```

Run the full lightweight set, including native pose-graph linkage, in the Thor
container with `bash scripts/thor_container.sh regressions`.

## Fairness Rules

Replica rendering-quality rows should use GT poses when measuring image quality.
TUM ATE/RPE rows should use the standard raw/preprocessed associations. Main
loop-closure numbers must come from online timed runs; offline diagnostics here
may explain a result but should not be fed back into the reported runtime path.
