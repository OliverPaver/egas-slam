# Architecture

GPS-SLAM couples an InfiniTAM SDF representation with a Gaussian model. The SDF
engine tracks and integrates RGB-D frames and provides raycast geometry and
colour. Gaussian optimization adds image detail using that geometric reference.
EGAS-SLAM partitions both representations into submaps while retaining
the original single-map implementation.

## Runtime flow

`slam_trainer.cpp` accepts one YAML configuration. It constructs a
`DatasetReader`, creates the TSDF engine through `TsdfEngineFacade`, configures
`SLAMPipeline` and dispatches the selected work mode.

| Work mode | Main work |
| --- | --- |
| `train` | Track and fuse RGB-D input, perform configured Gaussian optimization and export the map. |
| `recon` | Track and reconstruct the SDF without Gaussian training. |
| `eval` | Load saved map state and poses, select evaluation cameras and render images. |

For estimated-pose construction, the entrypoint forces the reader's pose source
to `identity`. It does not read future ground-truth poses to initialize tracking
or derive the Gaussian learning-rate scene scale. Ground-truth construction is
a separate explicit `PIPE.TSDF.use_gt_pose` mode.

The final training templates load input on demand, use the direct tensor bridge
and execute Gaussian work with the `immediate` schedule. Inputs are processed in
order. The pipeline owns tracking/fusion, map lifecycle, Gaussian work and the
configured online pose-graph checks. Reaching the final input is not by itself
proof that all native work or exports have completed.

## Map ownership and coordinates

`PIPE.TSDF.engine: basic` selects the original BasicEngine. With Gaussian submaps
disabled, this uses a unified Gaussian model. `engine: multi` selects MultiEngine;
the final EGAS-SLAM templates pair it with Gaussian submaps using
`ownership: tsdf` and `coordinate_frame: local`.

Each Gaussian submap belongs to an SDF submap. SDF voxels and Gaussian parameters
remain in their local coordinate systems. Rigid submap transforms place them in
the global frame. Loop correction updates those transforms and synchronizes the
rendering state instead of treating the submaps as one concatenated global
Gaussian tensor.

The active-map manager distinguishes a primary map, overlapping candidates and
passive maps. Capacity and tracking conditions can request a candidate. Promotion
uses configured overlap, visibility and pose-continuity checks. A raw tracker
failure, a candidate request and a successful primary-map handoff are different
events and have separate diagnostics.

## Residency and rendering

MultiEngine paging controls which SDF maps occupy device memory and retains
passive state in the configured backing storage. Snapshot queues and CPU cache
limits are separate from the compile-time per-map allocation profile. Gaussian
residency and insertion limits are controlled independently.

Rendering selects relevant submaps and composes their SDF/Gaussian contributions
using the configured owner, overlap and visibility policies. Camera frame IDs,
map ownership, submap transforms and overlap assignments are therefore part of
the saved state required for consistent evaluation. A saved Gaussian tensor
alone does not reconstruct the full hybrid map.

## Online loop correction

`PIPE.pose_graph` controls the pipeline's online loop path. It uses stored
observations, ORB matching and depth/ICP verification to propose constraints.
The pose-graph solver combines sequential and accepted loop constraints, while
support, residual, observation and tracking-health checks govern application.
Detecting a candidate does not guarantee that a correction will be applied.

The final EGAS-SLAM TUM and Replica templates enable this pipeline path. The
separate `PIPE.TSDF.multi.loop_closure` and `global_adjustment` switches remain
false; they are not aliases for `PIPE.pose_graph.online_updates`.

## Saving and evaluating a map

Training can save Gaussian state, the TSDF engine and estimated poses. When
corrected poses exist, the entrypoint also saves them. In `eval` mode it loads
the saved engine and Gaussian state, prefers a complete corrected trajectory
where available, and otherwise loads the saved estimated trajectory. Missing
required estimated poses are an error.

The included `saved_overlap_v2` correction fills an empty evaluation camera's
overlap list from its saved checkpoint assignment. It preserves a populated
live-camera list and does not change SLAM construction. This matters when a
render selection policy uses temporal overlap metadata, including a fallback
from bounds-based selection. Its source relationship is recorded in
[PROVENANCE.md](../PROVENANCE.md).

An evaluation run should use a writable copy of the required saved state and a
separate output location. Keep the source checkpoint, native timing and native
trajectory intact. An evaluation image or metric is a derivative of that native
observation, not another native SLAM run.

## Source map

| File or directory | Responsibility |
| --- | --- |
| `slam_trainer.cpp` | Entrypoint, work modes, checkpoint loading and camera selection. |
| `src/dataset_reader.cpp` | RGB-D input, intrinsics, frame IDs and pose I/O. |
| `slam/slam_pipeline.cpp` | Frame processing, Gaussian submaps, loop integration, save/load and rendering. |
| `slam/InfiniTAM_tools.cpp` | BasicEngine/MultiEngine facade and configuration validation. |
| `slam/TsdfFusion/CLIEngine.cpp` | Frame delivery to the SDF engine. |
| `InfiniTAM/ITMLib/Core/ITMMultiEngine.tpp` | MultiEngine processing, paging and combined raycasts. |
| `InfiniTAM/ITMLib/Engines/MultiScene/ITMActiveMapManager.cpp` | Candidate lifecycle and handoff decisions. |
| `InfiniTAM/MiniSlamGraphLib/` | Rigid pose-graph optimization. |
| `src/raw_gs_model.cpp`, `slam/slam_gs_model.cpp` | Gaussian optimization, insertion and model state. |
| `gsplat/` | C++/CUDA rasterization interface and kernels. |

See [Configuration](CONFIGURATION.md) for the supplied settings and
[Measurement and validation](BENCHMARK_HARDENING.md) for output interpretation.
