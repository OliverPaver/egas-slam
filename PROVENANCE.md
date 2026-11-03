# Source provenance

EGAS-SLAM is Oliver Paver's research extension of GPS-SLAM, developed at The University of Manchester under the supervision of Prof. Mikel Luján. It derives from [MisEty/GPS-SLAM](https://github.com/MisEty/GPS-SLAM), preserving upstream Git history and component attribution. The imported upstream release tip is [`057ad6a122f3391b297382798cb50ee2e33df08e`](https://github.com/MisEty/GPS-SLAM/commit/057ad6a122f3391b297382798cb50ee2e33df08e).

The Thor source release at `c944954d7000d2c905975c784f30a99ff7f3ff04` combines the GPS-SLAM native implementation frozen for `thor_repair_v3_20260910` with the separately accepted `saved_overlap_v2` evaluation correction. The correction restores recorded overlap membership during saved-map evaluation; it does not replace the training path. The original native and evaluation programs were built separately. This unified software release is not the exact binary used for both historical stages.

Release assembly preserves scientific source while adapting wrappers, explicit build presets, configuration paths, release checks and documentation. Numeric settings in `configs/final/` correspond to final training/evaluation templates; only locations and campaign-only labels are adapted. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records exact source/configuration/overlay digests. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) is the immutable manifest of that 1.1.0 release, excluding its own self entry; it does not describe subsequent development commits.

The current repository also contains desktop build support and demonstration instrumentation added after the Thor release. These changes and their validation are described in the [desktop development record](docs/development/desktop_setup_record.md) and [demonstration guide](DEMO_GUIDE.md). [Reported benchmark results](docs/results.md) retain their frozen Thor scope and are not measurements of the later desktop checkout.

Earlier R2 tags and commits remain intact as history. They are not the identity of the current tree. Benchmark outputs, exploratory sweeps, thesis documents, maps, datasets, weights and builds are excluded.

Bundled components retain their copyright and licence material; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), `LICENSES/` and [RELEASE_EQUIVALENCE.md](RELEASE_EQUIVALENCE.md). The combined distribution is subject to those restrictions, including the applicable non-commercial component terms. Repository visibility does not alter these terms or establish separate ownership permissions.
