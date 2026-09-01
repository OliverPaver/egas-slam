# Release Equivalence

## Equivalence claim

`thor-paper-20260819-r2-source` is the exact imported source snapshot used by the locked R2 evaluation. The `v1.0.0-paper` release changes repository presentation and execution paths only. It does not change SLAM behavior, numerical defaults, scheduler policy, tracking, submap construction, paging, loop closure, pose-graph optimization, Gaussian optimization, or CUDA kernels.

The protected algorithm tree is:

```text
slam/
src/
include/
gsplat/
InfiniTAM/
slam_trainer.cpp
CMakeLists.txt
configs/release/
```

Run the release check from any Git clone:

```bash
python3 tools/release_hygiene.py \
  --algorithm-equivalence thor-paper-20260819-r2-source
```

The check hashes Git blobs, rejects any added, deleted, mode-changed, or content-changed protected file, and therefore does not depend on filesystem timestamps.

## Post-R2 changes

- `scripts/thor_container.sh`: changed only clone-independent mounts, environment variables, build paths, and the source-equivalence command.
- `run/hybrid_validation.py`: resolves Replica data and the default run root only from `GPS_SLAM_DATA_ROOT`/`GPS_SLAM_RUN_ROOT`; benchmark configuration values are unchanged.
- `run/tum_loop_validation.py`: replaces removable-drive input/output fallbacks with the same explicit data/run roots; benchmark profiles and generated algorithm settings are unchanged.
- `run/batch_experiments.sh`: derives the repository and data/build/output roots instead of using a personal home directory; argument quoting and missing-binary diagnostics were hardened.
- `scripts/tum_loop_gt_audit.py`: obtains its optional dataset root from `GPS_SLAM_DATA_ROOT` instead of personal/removable-drive fallbacks.
- `scripts/tum_preprocess.py`: writes to the explicit data root instead of selecting a personal removable drive according to free space.
- `scripts/benchmark_freeze.py`: accepts `GPS_SLAM_BENCHMARK_FREEZE` so independently built binaries can use a separate exact local manifest without replacing the historical R2 record.
- `scripts/benchmark_hardening_regression.py`: uses a synthetic clean-freeze fixture for its manifest round-trip, allowing the test to run without R2 binaries; production manifest behavior is not mocked.
- `scripts/tracking_failure_policy_regression.py`: compiles its header-only policy fixture with InfiniTAM's `COMPILE_WITHOUT_CUDA` guard so the hardware-independent check also runs on CPU-only CI hosts; the tested policy and production build are unchanged.
- Repository metadata, curated documentation, licensing material, GitHub Actions, release hygiene, and artifact packaging were added.
- File modes and line-ending policy were normalized without modifying R2 algorithm contents.

Container paths changing from `/workspace/transfer/GPS-SLAM` to `/workspace/project` do not alter compiled code. Docker on Thor remains native AArch64/SM 11.0 execution, not emulation. Builds from another Git commit can differ byte-for-byte because the commit identifier is embedded in the executable; algorithm equivalence is consequently established from the protected Git tree, while each benchmark manifest records its exact binary digest.
