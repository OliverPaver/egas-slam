# Configuration

Use the explicit templates under [configs/final/](../configs/final/) as the
starting point. Their training values come from the revised external GPS-SLAM
scene configurations, with portable input and output paths. Separate
`*_eval.yaml` templates describe saved-map rendering. Changing a parameter makes
a new configuration; retain the exact YAML with its outputs.

## Scene templates and input

| Template group | Scenes | Input image shape | Native depth divisor |
| --- | --- | --- | --- |
| `configs/final/tum/` | `fr1_desk`, `fr2_xyz`, `fr3_office` | 640 by 480 | 5000 |
| `configs/final/replica/` | `office0` through `office4`, `room0` through `room2` | 1200 by 680 | 6553.5 |

The TUM templates retain their own scene intrinsics. These GPS-SLAM calibration
values must not be replaced with another method's configuration. Depth is
converted using the configured divisor; image dimensions, intrinsics, prepared
depth encoding and any resizing must agree.

The container maps external inputs under `/workspace/data` and run outputs under
`/workspace/runs`. `READER.input_dir` selects a prepared scene. Its `image_path`,
`depth_path`, `pose_path`, prefixes and extensions describe the files below that
scene. `start_frame` and `end_frame` are inclusive, and `frame_step` controls
input sampling. Template frame ranges and intrinsics are explicit.

Use a dedicated `workspace_dir` for each training/reconstruction invocation;
startup can clear it. For saved-map evaluation, work from a copy of the required
checkpoint and direct derived output to its own location. Follow the concrete
setup and commands in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Build profile and runtime settings

| Setting | Final EGAS-SLAM configuration | Meaning |
| --- | --- | --- |
| Build hash profile | `medium_128k` | Compile-time SDF capacity profile for the EGAS-SLAM engine. |
| `PIPE.TSDF.engine` | `multi` | Use SDF submaps. |
| `PIPE.TSDF.multi.hash_profile` | `medium_128k` | Runtime configuration must match the binary's profile. |
| `PIPE.gaussian_submaps.enable` | `true` | Pair Gaussian submaps with the SDF maps. |
| Gaussian ownership/frame | `tsdf` / `local` | Preserve SDF ownership and local parameter coordinates. |
| `PIPE.gaussian_schedule.mode` | `immediate` | Execute scheduled Gaussian work without distributing it across later inputs. |
| `PIPE.TSDF.input_mode` | `on_demand` | Decode input as it is consumed. |
| `PIPE.TSDF.input_prefetch` | `false` | Use the recorded synchronous input setting. |
| `PIPE.TSDF.input_tensor_bridge` | `direct` | Convert CPU tensors into reusable SDF input buffers. |
| `PIPE.TSDF.tracking_profile` | `legacy` | Preserve the selected ICP tracker recipe. |
| `PIPE.TSDF.tracking_failure_mode` | `legacy_ignore` | Preserve this recipe's fusion policy after raw tracking failure. |
| `PIPE.TSDF.reject_nonfinite_tracking_pose` | `true` | Reject invalid numerical tracker output. |
| `PIPE.TSDF.use_gt_pose` | `false` in training | Estimate poses from RGB-D input. |
| `PIPE.eval_after_train` | `false` | Keep image evaluation separate from native construction. |
| `PIPE.weight_configs.ssim_weight` / `depth_weight` | `0.0` / `0.0` | Retain the selected RGB loss recipe. |

The original-algorithm build uses `legacy_pool_2m`, BasicEngine and a unified
Gaussian model. It is a separate configuration and binary, not a runtime toggle
on a `medium_128k` build. The presets use `build_thor/` for the EGAS-SLAM binary
and `build_original/` for the original-algorithm binary.

BasicEngine capacity instrumentation is compiled OFF for the EGAS-SLAM preset
and ON for the original preset. That switch controls the added BasicEngine
capacity instrumentation; it is distinct from the runtime's full measurement
profile. Preserve those settings and the runtime environment when comparing
measurements. Do not infer diagnostic scope from a build directory name alone.

## Loop closure and map transitions

Both EGAS-SLAM dataset groups use `PIPE.pose_graph.candidate_detection: true`,
`online_updates: true` and `apply_mode: loop_candidates`. Internal MultiEngine
`loop_closure` and `global_adjustment` remain false. Disabling one internal
switch does not disable the pipeline's pose-graph path.

The templates retain explicit candidate verification, support, consistency,
pose-change and observation-health thresholds. Keep these together when using
the final loop recipe. Candidate detection, graph solving and applying a
correction are separate operations. The `dry_run` and `solve_only` fields do not
replace the explicit `apply_mode` and `online_updates` controls.

SDF transition settings are grouped under `PIPE.TSDF.multi`: capacity thresholds,
candidate visibility, overlap/trial duration, primary-failure handling and
pose-continuity policy. Tracking-boundary requests do not automatically authorize
a handoff. The supplied `validated_candidate` handoff policy works with the
remaining acceptance checks.

## Residency and Gaussian work

SDF paging uses CPU backing storage in these templates. `resident_gpu_maps`,
`gpu_reserve_mb`, `cpu_page_cache_limit_mb` and the pending-snapshot settings
control different parts of the memory budget. Per-map SDF capacity is not a
whole-process memory cap. The effective depth frustum is 0.2 to 10 metres.

Gaussian insertion and residency have their own controls under
`PIPE.gaussian_submaps`. The training templates preserve candidate warming and
handoff settings while disabling graduation optimization, frozen refinement and
sequence-end quality finalization. Enabling those branches changes the work
performed; it is not a storage-only adjustment.

## Saved-map evaluation

Use the separate evaluation templates instead of treating a training render
policy as interchangeable with final-map rendering. The final evaluation recipe
uses Gaussian `owner_neighbor_fill` with `sdf_bounds` selection and a maximum of
eight visible submaps. It sets
`PIPE.gaussian_submaps.render_recompose_with_global_sdf_base: true`,
`PIPE.TSDF.multi.render_compositor: owner_priority` and
`PIPE.TSDF.multi.render_order_policy: adjacent_first`. Preserve the saved map's
owner/overlap assignments and poses.

`PIPE.eval_split` chooses the camera split. Explicit `eval_frame_indices` select
sorted, unique nonnegative frame IDs and cannot be combined with stride,
offset or maximum-count sampling. The alternative controls are `eval_stride`,
`eval_start_offset` and `eval_max_frames`. An explicit list is required to
reproduce a particular fixed view set; an arbitrary stride is not equivalent.
The renderer records its selection in `eval_sampling.json`.

Lossless PNG output and the recorded quantisation convention support subsequent
image scoring. `scripts/metric_general.py` converts RGB inputs from [0,1] to
[-1,1] for the bundled LPIPS implementation. Preserve that convention and the
weight identity when comparing scores. More output and timing guidance is in
[Measurement and validation](BENCHMARK_HARDENING.md).
