# Provenance

## Upstream lineage

This repository preserves the Git history of [MisEty/GPS-SLAM](https://github.com/MisEty/GPS-SLAM). The imported upstream `release` tip is commit `057ad6a122f5ef7f8d0269125741cf4d39b86ffc`; the `upstream` remote points to that repository.

Oliver Paver developed the submap, loop-closure, robustness, benchmark-hardening, and Jetson AGX Thor changes as part of postgraduate research in the Department of Computer Science at The University of Manchester, supervised by Prof. Mikel Luján.

## Snapshot chain

1. The transferred R2 implementation was frozen as `thor-paper-20260819-r2` and recorded in `BENCHMARK_FREEZE.json`.
2. Every allowlisted functional source/configuration file was checked against that manifest before import. There were 3,911 matching files and no missing or modified entries after excluding generated `ThirdLibs/install` products.
3. The exact source import was committed once and annotated as `thor-paper-20260819-r2-source`.
4. Release hardening was performed only in the separate staging repository. The locked working copy and its 38 MB private archival bundle were not modified.

The original archival bundle remains private because it contains generated binaries and did not include all third-party licence material. The public source release instead vendors the pinned source dependencies listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Evidence and traceability

- `BENCHMARK_FREEZE.json` identifies the historical R2 source, binaries, build profiles, container, and dataset protocol.
- [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) states metric, timing, completeness, and failure-handling rules.
- [PAPER_READINESS_LOCK_2026-08-25.md](PAPER_READINESS_LOCK_2026-08-25.md) records the final 61-run matrix and soak decision.
- [RELEASE_EQUIVALENCE.md](RELEASE_EQUIVALENCE.md) defines and verifies the unchanged algorithm boundary.

The compact paper artifact is built separately from the raw run store. Its manifest maps each retained file to a SHA-256 digest; datasets, checkpoints, full image sequences, models, meshes, and caches are excluded.
