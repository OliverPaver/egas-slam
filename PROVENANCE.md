# Source provenance

This is Oliver Paver's final GPS-SLAM research implementation for NVIDIA Jetson AGX Thor, developed at The University of Manchester under the supervision of Prof. Mikel Luján. It derives from [MisEty/GPS-SLAM](https://github.com/MisEty/GPS-SLAM), preserving upstream Git history and component attribution. The imported upstream release tip is `057ad6a122f5ef7f8d0269125741cf4d39b86ffc`.

The final source combines the GPS-SLAM native implementation frozen for `thor_repair_v3_20260910` with the separately accepted `saved_overlap_v2` evaluation correction. The correction restores recorded overlap membership during saved-map evaluation; it does not replace the training path. The original native and evaluation programs were built separately. This unified software release is not the exact binary used for both historical stages.

Assembly preserves scientific source while adapting wrappers, explicit build presets, configuration paths, release checks and documentation. Numeric settings in `configs/final/` correspond to final training/evaluation templates; only locations and campaign-only labels are adapted. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records exact source/configuration/overlay digests. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) binds the entire shipped tree, except its own self entry.

Earlier R2 tags and commits remain intact as history. They are not the identity of the current tree. Benchmark outputs, exploratory sweeps, thesis documents, maps, datasets, weights and builds are excluded.

Bundled components retain their copyright and licence material; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), `LICENSES/` and [RELEASE_EQUIVALENCE.md](RELEASE_EQUIVALENCE.md). The combined distribution is subject to those restrictions. This private source upload does not change repository visibility or resolve separate public-distribution permissions.
