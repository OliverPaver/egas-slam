# GPS-SLAM submaps for NVIDIA Jetson AGX Thor

This repository contains the final implementation of a research fork of
[GPS-SLAM](https://github.com/MisEty/GPS-SLAM). It combines RGB-D tracking and
TSDF fusion with Gaussian reconstruction, local SDF/Gaussian submaps, map
residency controls and online pose-graph loop closure. The original BasicEngine
and unified Gaussian path remains available alongside the proposed submap path.

The supported deployment target is native AArch64 on Jetson AGX Thor with
JetPack 7.1 and CUDA SM 11.0. Source, configuration templates, build support and
evaluation tools are included. Datasets, weights, compiled libraries, map
checkpoints and historical benchmark results are maintained separately.

## Source and evaluation scope

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) identifies the shipped source.
[PROVENANCE.md](PROVENANCE.md) explains its relationship to the revised native
implementation and the accepted saved-map evaluation correction.

The evaluation correction restores a camera's missing overlap-submap metadata
from the assignment already saved in its checkpoint. It is confined to saved-map
evaluation and leaves populated live-camera overlap lists unchanged. A binary
built from this combined source has its own identity; this repository does not
claim that the combined release binary was used for historical native timing.

## Build and run

Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for prerequisites, external data
and weight layouts, build commands and one-scene execution. Build on the Thor's
NVMe/ext4 filesystem in the supplied NVIDIA container environment.

The proposed build uses `medium_128k` and writes `build_thor/slam_trainer`.
The original-algorithm build uses `legacy_pool_2m` and writes
`build_original/slam_trainer`. Configuration templates are under
[configs/final/](configs/final/). Both TUM and Replica proposed templates enable
the pipeline's online pose-graph path; the separate internal MultiEngine loop
closure switch remains disabled.

Check the source snapshot without launching SLAM:

```bash
python3 tools/release_hygiene.py --source-manifest SOURCE_MANIFEST.json
```

Set `GPS_SLAM_DATA_ROOT`, `GPS_SLAM_WEIGHT_ROOT` and `GPS_SLAM_RUN_ROOT` as
described in the reproducibility guide. The wrapper exposes data at
`/workspace/data` and results at `/workspace/runs`. Give each training run a
dedicated output directory: startup can clear the configured `workspace_dir`.

## Code and documentation

| Path | Purpose |
| --- | --- |
| `slam_trainer.cpp` | Training, reconstruction and saved-map evaluation entrypoint. |
| `slam/`, `src/`, `include/` | Tracking integration, submaps, Gaussian models and dataset handling. |
| `InfiniTAM/`, `gsplat/` | SDF/tracking implementation and CUDA rasterization. |
| `ThirdLibs/` | Bundled build dependencies and their original notices. |
| `configs/final/` | Explicit scene templates and separate evaluation settings. |
| `scripts/`, `containers/` | Native build, runtime checks, preprocessing and metrics. |
| [docs/](docs/README.md) | Architecture, configuration and measurement guidance. |

## Attribution and terms

Keep [LICENSE.txt](LICENSE.txt), the complete [LICENSES/](LICENSES/) directory
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with the project. Bundled
components have distinct terms, including non-commercial research conditions;
the combined project must not be described as solely AGPL-3.0. See
[CITATION.cff](CITATION.cff) for software attribution and the original GPS-SLAM
paper reference. This repository remains private.
