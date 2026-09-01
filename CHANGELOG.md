# Changelog

All notable project changes are documented here. Dates use ISO 8601.

## Unreleased

- Make the header-only tracking-failure regression compile on CPU-only GitHub-hosted runners without requiring CUDA headers.

## 1.0.0-paper — 2026-08-25

- Preserve the frozen `thor-paper-20260819-r2` submap and loop-closure implementation and its 61-run comparison evidence.
- Add native Jetson AGX Thor/SM 11.0 container build, preflight, telemetry, benchmark provenance, and regression tooling.
- Add bounded multi-SDF and Gaussian submaps, loop-candidate retrieval and verification, online pose-graph correction, tracking-failure containment, and embedded paging controls.
- Add opt-in deployment scheduler/trace analysis while retaining immediate Gaussian scheduling as the paper default.
- Make container, dataset, weight, run, and experiment paths clone-independent.
- Vendor pinned build dependencies as source and add licensing/provenance documentation.
- Add release hygiene, CI, reproducibility, and compact-artifact tooling.

The algorithmic snapshot predates the release-only portability/documentation commit. See [RELEASE_EQUIVALENCE.md](RELEASE_EQUIVALENCE.md) for the exact boundary.
