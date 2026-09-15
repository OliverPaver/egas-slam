# Measurement interfaces

This repository distributes final GPS-SLAM software and its input, output and evaluation interfaces. Thesis campaigns and the comparator harness are maintained separately.

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for final configurations and [measurement documentation](docs/BENCHMARK_HARDENING.md) for timing, completeness and metrics. Training and saved-map evaluation are separate. Full input processing and fixed500-view evaluation are different amounts of work.

Keep rigid SE(3) RMSE distinct from mean error and similarity-aligned diagnostics; supplied-pose reconstruction distinct from estimated-pose mapping; process completion distinct from uninterrupted tracking; and processing throughput distinct from whole-command/offline-evaluation rates. Memory statistics also retain their distinct scopes.

General config-generation helpers retain older named protocol modes. Their identifiers describe their own rules and are not aliases for the revised multi-system benchmark. Current software presets are in `configs/final/`, with identity in `SOURCE_MANIFEST.json`. New work needs its own output directory and provenance; never overwrite completed evidence.
