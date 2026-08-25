# Hybrid MultiEngine SDF/Gaussian Submap Config Guide

This guide documents the non-base GPS-SLAM configuration toggles added for the
MultiEngine SDF submap, Gaussian submap, rendering, residency, and online loop
closure work. The original BasicEngine path remains the baseline and should be
kept working for every comparison.

## Baseline Modes

Use these switches to choose the major runtime path.

| Key | Values | Default | Meaning |
| --- | --- | --- | --- |
| `PIPE.TSDF.engine` | `basic`, `multi` | `basic` | Selects the original single SDF engine or the MultiEngine submap backend. |
| `PIPE.TSDF.input_mode` | `preload`, `on_demand` | `preload` in base configs | Controls whether frames are preloaded or streamed on demand. Use `on_demand` for long runs and embedded-style memory pressure. |
| `PIPE.TSDF.input_prefetch` | `true`, `false` | `false` in the release baseline | Uses one persistent worker to load the next on-demand frame while the current frame runs. Enable for Thor streaming; disable for a synchronous A/B. |
| `PIPE.gaussian_schedule.mode` | `immediate`, `sliced` | `immediate` | `sliced` spreads one fixed local optimization job over subsequent frames. Paper-parity runs remain `immediate`. |
| `PIPE.gaussian_schedule.iterations_per_frame` | positive integer | `1` | Slice size. Thor deployed protocols use `2`, preserving 20 iterations over a 10-frame interval. The config is rejected if the job cannot finish before the next trigger. |
| `PIPE.export_online_end_trajectory` | `true`, `false` | `false` | Exports the causal corrected trajectory before terminal scanning under `pose_graph_live/online_end_poses/`. Required for deployed real-time accuracy. |
| `PIPE.TSDF.input_tensor_bridge` | `direct`, `opencv_legacy` | `direct` | `direct` fills reusable InfiniTAM buffers from CPU Torch tensors without owning OpenCV intermediates. `opencv_legacy` preserves the earlier conversion path. |
| `PIPE.TSDF.input_tensor_bridge_validate` | `true`, `false` | `false` | Executes both input bridges and aborts on any per-pixel difference. Requires direct on-demand input and should be disabled for timing. |
| `PIPE.submap_stats_interval_frames` | integer ≥ 0 | `1` | Samples ordinary submap-stat rows every N frames; `0` writes only end state. Use `10` for embedded throughput runs. Event/transition logs remain enabled. |
| `PIPE.TSDF.tracking_failure_mode` | `legacy_ignore`, `contain_failed`, `stop_integration`, `relocalise` | `legacy_ignore` | `legacy_ignore` exactly preserves the old baseline, including fusion after raw failure. `contain_failed` is the recommended mode: raw `FAILED` rolls back and cannot fuse, while recoverable `POOR` frames retain legacy behavior. `stop_integration` also suppresses `POOR`. `relocalise` is BasicEngine-only. |
| `PIPE.TSDF.tracking_profile` | `legacy`, `efficient`, `robust_rpe` | `legacy` | `legacy` preserves InfiniTAM's 20/50-iteration tracker. `efficient` caps both ends at 20 iterations. `robust_rpe` also lowers the mature-voxel gate from 20 to 10; it is the best measured RPE/failure profile, but should remain explicit because short-sequence ATE can shift slightly. |
| `PIPE.TSDF.use_bilateral_filter` | `true`, `false` | `false` | Applies InfiniTAM's five-pass edge-aware depth filter before tracking and fusion. Test accuracy and latency together; filtering changes the TSDF input as well as ICP. |
| `PIPE.TSDF.use_approximate_raycast` | `true`, `false` | `false` | Forward-renders a recent tracking point cloud until it is stale or the camera moves about 2.2 cm, avoiding some full TSDF raycasts. Keep disabled for the strict baseline. |
| `PIPE.TSDF.reject_nonfinite_tracking_pose` | `true`, `false` | `false` | Rejects non-finite tracker output before policy handling. Robust profiles set this to `true`; the baseline remains selectable. |
| `PIPE.gaussian_submaps.enable` | `true`, `false` | `false` | Enables one Gaussian model per SDF submap. When false, the original unified Gaussian model is used. |
| `PIPE.gaussian_submaps.ownership` | `legacy`, `tsdf` | `legacy` | `tsdf` ties Gaussian ownership to MultiEngine SDF submap IDs. |
| `PIPE.gaussian_submaps.coordinate_frame` | `global`, `local` | `global` | `local` stores Gaussian means/cameras in the owner SDF submap frame. Required for future pose-graph correction. |

Recommended clean baselines:

- Original SDF only: `PIPE.TSDF.engine: basic`, Gaussian training disabled.
- Original unified Gaussian: `PIPE.TSDF.engine: basic`,
  `PIPE.gaussian_submaps.enable: false`.
- Hybrid submaps without loop closure: `PIPE.TSDF.engine: multi`,
  `PIPE.gaussian_submaps.enable: true`, `ownership: tsdf`,
  `coordinate_frame: local`.

## MultiEngine SDF Submaps

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.TSDF.multi.loop_closure` | `false` | Keeps non-sequential loop closure disabled unless explicitly testing pose graph apply. |
| `PIPE.TSDF.multi.global_adjustment` | `false` | Prevents global TSDF reintegration. Pose graph corrections update submap transforms only. |
| `PIPE.TSDF.multi.console_diagnostics` | `false` | Enables verbose MultiEngine event prints to stdout. Keep `false` for Thor/long runs; structured CSV diagnostics are still written. |
| `PIPE.TSDF.multi.boundary_policy` | `visibility` | Creates candidate maps when the active map no longer explains enough of the incoming view. |
| `PIPE.TSDF.multi.new_map_visible_ratio` | `0.25` | Visibility threshold for starting/accepting transitions. Lower creates fewer submaps; higher creates more. |
| `PIPE.TSDF.multi.max_frames` | `750` | Safety limit on frames per SDF submap. Used to bound map size even when visibility stays high. |
| `PIPE.TSDF.multi.max_block_occupancy` | `0.80` | Creates a new map before the hash volume silently saturates. |
| `PIPE.TSDF.multi.occupancy_threshold_policy` | `fixed` | Selects the effective capacity thresholds. `fixed` preserves the configured `new_map_block_occupancy` and `max_block_occupancy`. `adaptive_longspan` is experimental and may relax thresholds only when raw tracking is healthy. |
| `PIPE.TSDF.multi.adaptive_new_map_block_occupancy` | profile-dependent | Relaxed new-map trigger used only by `adaptive_longspan`. Keep unset or equal to the fixed threshold for baseline-comparable runs. |
| `PIPE.TSDF.multi.adaptive_max_block_occupancy` | profile-dependent | Relaxed fusion stop threshold used only by `adaptive_longspan`. Values above about `0.93` can increase non-good tracking and near-full-map risk on TUM prefixes. |
| `PIPE.TSDF.multi.adaptive_occupancy_min_primary_frames` | profile-dependent | Minimum current-primary age before adaptive occupancy thresholds can activate. |
| `PIPE.TSDF.multi.adaptive_occupancy_min_processed_frames` | profile-dependent | Minimum sequence age before adaptive occupancy thresholds can activate. |
| `PIPE.TSDF.multi.adaptive_occupancy_max_raw_weak_streak` | profile-dependent | Maximum current weak raw-tracking streak allowed while using adaptive thresholds. |
| `PIPE.TSDF.multi.adaptive_occupancy_max_raw_failed_streak` | profile-dependent | Maximum current failed raw-tracking streak allowed while using adaptive thresholds. |
| `PIPE.TSDF.multi.hash_profile` | `medium_128k` | TSDF hash capacity profile. `legacy_large` is for BasicEngine baselines; `medium_128k` is the main quality profile; `compact_64k` is embedded stress. |
| `PIPE.TSDF.multi.sampled_original_blocks` | `2000` | Number of active-map blocks sampled when checking new-map visibility. |
| `PIPE.TSDF.multi.overlap_frames` | `20` | Frames where old and candidate SDF maps coexist during transition. |
| `PIPE.TSDF.multi.trial_frames` | `30` | Candidate evaluation duration before accepting/rejecting a new submap. |
| `PIPE.TSDF.multi.handoff_pose_policy` | `candidate`, `preserve_global`, `validated_candidate` | `validated_candidate` retains a fresh candidate pose after a normal validated-overlap handoff, but preserves the last finite primary global pose during emergency failure recovery. It is the paper-parity recommendation. |
| `PIPE.TSDF.multi.tracking_boundary_policy` | `raw_weak_streak` or `disabled` | Allows weak/failed raw tracking streaks to request a candidate submap. |
| `PIPE.TSDF.multi.tracking_boundary_handoff_policy` | `weak_or_failed`, `failed_only`, `never` | Controls whether tracking-boundary requests may also force primary promotion. Use `failed_only` for paper-parity runs so weak tracking warms candidates but does not cut the map. |
| `PIPE.TSDF.multi.raw_weak_boundary_frames` | profile-dependent | Consecutive non-good raw tracking frames before requesting a candidate. |
| `PIPE.TSDF.multi.raw_failed_boundary_frames` | profile-dependent | Consecutive hard raw tracking failures before requesting or promoting via the tracking-boundary path. |
| `PIPE.TSDF.multi.primary_failure_handoff_policy` | `disabled`, `promote_visible_candidate` | Emergency promotion path for a candidate that is already visible and mature while the primary raw tracker is failing. |
| `PIPE.TSDF.multi.primary_failure_handoff_min_raw_failed_frames` | profile-dependent | Consecutive hard raw tracking failures required before the primary-failure path can promote a candidate. |
| `PIPE.TSDF.multi.primary_failure_handoff_raw_gate` | `failed_only`, `failed_or_weak` | Selects whether the primary-failure path is gated only by hard raw failures or may also use non-good weak tracking. Use `failed_only` for paper-parity ATE runs. |
| `PIPE.TSDF.multi.primary_failure_handoff_min_raw_weak_frames` | `0` for paper parity | Consecutive raw non-good frames required only when `primary_failure_handoff_raw_gate: failed_or_weak`. This is opt-in legacy/stress behavior. |
| `PIPE.TSDF.multi.primary_failure_handoff_require_pose_continuity` | `true` for robust profiles | Refuses emergency promotion unless `preserve_global` or `validated_candidate` can anchor the candidate to the last valid primary camera pose. Validation rejects `true` with plain `handoff_pose_policy: candidate`. |
| `PIPE.TSDF.multi.tracking_boundary_min_primary_frames` | profile-dependent | Grace period after a handoff before weak/failure tracking can request another structural transition. |
| `PIPE.TSDF.multi.tracking_boundary_cooldown_frames` | profile-dependent | Cooldown between tracking-boundary candidate requests. |
| `PIPE.TSDF.multi.occupancy_growth_lookahead_frames` | `0` for parity | Predictive capacity lookahead. Useful for stress profiles, but disabled in paper-parity runs to avoid cutting easy sequences early. |
| `PIPE.TSDF.multi.render_compositor` | `owner_priority` | TSDF overlap compositor. `owner_priority` is the current safe default for final-map rendering. |
| `PIPE.TSDF.multi.render_order_policy` | `adjacent_first` | Orders candidate TSDF maps for combined raycast. Used with owner-priority composition. |

Notes:

- The CUDA 32-map limit is treated as a render-batch limit, not a logical
  submap limit.
- MultiEngine no-loop ATE must be compared against BasicEngine on matching
  frames before loop-closure results are trusted.
- Weak raw tracking should warm a candidate submap, not immediately promote it
  in paper-parity profiles. `primary_failure_handoff_raw_gate: failed_only`
  keeps that separation explicit; `failed_or_weak` exists only for controlled
  recovery/stress ablations.
- Hard failure bypasses only the candidate-request cooldown: it can escalate an
  existing candidate immediately, but promotion still has to pass candidate
  quality, visibility, block-count, and pose-continuity gates.

## TSDF Residency

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.TSDF.multi.paging` | `true` | Enables bounded SDF residency. |
| `PIPE.TSDF.multi.paging_storage` | `cpu` | Keeps passive maps in host memory for PC/Thor-like validation. Disk paging remains available for stress tests. |
| `PIPE.TSDF.multi.resident_gpu_maps` | `8` | Number of SDF maps allowed resident on GPU. Keep primary/candidate/recent maps resident. |
| `PIPE.TSDF.multi.async_page_out` | `true` | Snapshots passive maps asynchronously. Dirty maps must not be evicted before snapshot completion. |
| `PIPE.TSDF.multi.max_pending_page_snapshots` | `2` | Bounds background page-out work. |
| `PIPE.TSDF.multi.max_pending_page_snapshot_mb` | `1200` | Bounds queued page-out memory pressure. |
| `PIPE.TSDF.multi.async_page_in` | `true` | Allows async prefetch when rendering/evaluating passive maps. |
| `PIPE.TSDF.multi.max_pending_page_ins` | `1` | Bounds background page-in work. |
| `PIPE.TSDF.multi.max_pending_page_in_mb` | `512` | Bounds queued page-in memory pressure. |

For Thor, use bounded resident maps and CPU paging coordination. Do not fill
shared memory with passive TSDF capacities.

Before moving a generated config to Thor, run:

```bash
python scripts/thor_preflight.py path/to/config.yaml
```

The preflight is intentionally conservative. It checks for long-run hazards such
as preloaded images, disabled prefetch, legacy/validation bridge overhead,
unbounded MultiEngine residency, verbose or per-frame diagnostics, unsafe
Gaussian-submap ownership, and missing loop-apply health gates. It does not
modify the config or benchmark the system.

## Gaussian Submaps

Image-loss controls are independent of the submap toggle:

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.weight_configs.ssim_weight` | `0.0` baseline, `0.2` perceptual | Blends RGB L1 with D-SSIM. Values outside `[0,1]` are rejected. `0.2` improved PSNR, SSIM, and LPIPS on all eight matched Replica scenes at an average 8.5% throughput cost. |
| `PIPE.weight_configs.depth_weight` | `0.0` | Adds rendered-depth L1. Keep disabled: the tested `0.1` setting reduced RGB quality, coverage, and throughput. |

`run/hybrid_validation.py --gaussian-profile
thor_quality_reconciled_perceptual` selects the validated `0.2` SSIM objective.
The baseline profile remains `thor_quality_reconciled`; passing
`--ssim-weight 0` also explicitly restores L1-only training. Empty RGB/depth
masks produce differentiable zero losses rather than NaNs.

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.gaussian_submaps.render_selection_policy` | `sdf_bounds` | Selects visible Gaussian submaps using projected SDF bounds. |
| `PIPE.gaussian_submaps.render_max_visible_submaps` | `6` to `8` | Caps submaps rendered per frame. Owner is selected first. |
| `PIPE.gaussian_submaps.render_max_visible_gaussians` | `280000` | Caps non-owner Gaussian load for half-res desktop validation. Owner may be oversized. |
| `PIPE.gaussian_submaps.render_selection_hysteresis_frames` | `5` | Adds short stickiness to recently selected submaps to reduce flicker. |
| `PIPE.gaussian_submaps.render_overlap_depth_gate` | `true` | Rejects overlapping neighbor contributions that disagree with owner/SDF depth. |
| `PIPE.gaussian_submaps.render_overlap_depth_tolerance_mode` | `scaled_max` | Uses depth/voxel-scaled tolerance instead of a fixed additive threshold. |
| `PIPE.gaussian_submaps.cross_submap_novelty_gate` | `true` | Suppresses new Gaussians that are already represented by appropriate neighboring submaps. |
| `PIPE.gaussian_submaps.cross_novelty_reference_policy` | `asymmetric_handoff` | Active-map insertion references frozen ready maps plus outgoing handoff; candidate insertion references frozen accepted maps only. |
| `PIPE.gaussian_submaps.gaussian_insert_selection` | `tile_error_topk` or `adaptive_tile_error_topk` | Chooses candidate pixels for new Gaussians. Tile policies avoid concentrating all insertions in one image region. |
| `PIPE.gaussian_submaps.gaussian_insert_tile_size` | `64` | Tile size for tiled insertion policies. |

The current stable quality rows use Gaussian submaps in local coordinates and
do not concatenate tensors into a fake global model during normal rendering.

## Gaussian Budgeting

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.gaussian_submaps.gaussian_budget_policy` | `sdf_block_density` | Sets per-submap Gaussian caps from SDF block count. |
| `PIPE.gaussian_submaps.gaussian_budget_base` | `8000` | Base Gaussian allowance per submap. |
| `PIPE.gaussian_submaps.gaussian_budget_blocks_density` | `0.40` | Adds Gaussians proportional to allocated SDF blocks. |
| `PIPE.gaussian_submaps.gaussian_budget_min` | `8000` | Minimum cap. |
| `PIPE.gaussian_submaps.gaussian_budget_max` | `90000` | Maximum cap. |
| `PIPE.gaussian_submaps.gaussian_budget_pressure_ratio` | `0.90` | Starts redundant-Gaussian pruning before hard-cap insertion skips. |
| `PIPE.gaussian_submaps.gaussian_budget_prune_interval_frames` | `10` | Minimum interval between pressure-prune passes. |

Budget pressure must never block TSDF tracking or submap lifecycle. At hard cap,
new Gaussian insertions are skipped and logged.

## Graduation And Readiness

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.gaussian_submaps.graduation_optimize` | `true` in hybrid quality profiles | Runs bounded final optimization before freezing a submap. |
| `PIPE.gaussian_submaps.graduation_opt_iters` | profile-dependent | Iterations per graduation pass. |
| `PIPE.gaussian_submaps.graduation_max_opt_passes` | `2` | Caps graduation passes. |
| `PIPE.gaussian_submaps.graduation_readiness_enable` | `true` | Measures whether hybrid rendering improves over owner SDF on held-out observations. |
| `PIPE.gaussian_submaps.graduation_readiness_sample_cameras` | `5` | Held-out cameras used for readiness. |
| `PIPE.gaussian_submaps.graduation_readiness_skip_gain` | `0.0015` | If readiness is already this good, skip extra graduation work. |
| `PIPE.gaussian_submaps.graduation_readiness_min_abs_drop` | `0.0001` | Minimum improvement required to justify another pass. |

Readiness is measured as:

```text
readiness_delta = owner_sdf_rgb_l1 - hybrid_rgb_l1
```

Positive values mean Gaussians improve over the owner SDF base.

## Frozen-Submap Refinement

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.gaussian_submaps.frozen_refinement_enable` | `false` stable, `true` ablation | Re-optimizes already frozen accepted Gaussian submaps using stored boundary/cross-boundary observations. |
| `PIPE.gaussian_submaps.frozen_refinement_interval_frames` | `30` | Minimum frame interval between online frozen-refinement checks. |
| `PIPE.gaussian_submaps.frozen_refinement_keyframes` | `8` | Stored observations selected for each refinement attempt. |
| `PIPE.gaussian_submaps.frozen_refinement_max_passes_per_submap` | `2` | Hard cap on completed refinement attempts per submap. Rejected acceptance attempts still count so the policy cannot retry indefinitely. |
| `PIPE.gaussian_submaps.frozen_refinement_acceptance_enable` | `true` in `xboundary_accept` | Snapshots Gaussian tensors and rolls back the refinement attempt when held-out readiness does not improve. |
| `PIPE.gaussian_submaps.frozen_refinement_acceptance_min_delta` | `0.0` | Required readiness improvement to keep a frozen-refinement attempt. Positive thresholds are experimental and can add overhead by forcing more rollbacks. |
| `PIPE.gaussian_submaps.frozen_refinement_observation_policy` | `temporal_spread` stable | Selects stored observations for frozen refinement. Supported values are `temporal_spread`, `readiness_temporal_mix`, and `sdf_error_temporal_mix`. |
| `PIPE.gaussian_submaps.frozen_refinement_observation_selection_max_eval` | `8-12` in diagnostics | Caps how many stored observations are scored before selecting refinement cameras. `0` evaluates all stored observations. |
| `PIPE.gaussian_submaps.cross_boundary_refinement_enable` | `true` in `xboundary*` ablations | Stores depth-gated observations from neighboring accepted submaps for possible frozen refinement. |

Frozen-submap refinement is currently a controlled diagnostic path, not the
main stable quality profile. It can help `room0` slightly, but the same policy
does not beat the base reconciled profile on `office2`.

The `readiness_temporal_mix` selector scores candidate observations by rendering
the frozen Gaussian submap and selecting frames where hybrid RGB is worst
relative to the owner SDF. It is informative but expensive. The
`sdf_error_temporal_mix` selector uses only owner-SDF RGB error plus temporal
coverage, avoiding the extra Gaussian render during selection. Both remain
diagnostic policies until they improve quality without unacceptable P99
latency.

## Sequence-End Quality Finalization

| Key | Typical Value | Meaning |
| --- | ---: | --- |
| `PIPE.gaussian_submaps.quality_finalize_enable` | `true` in reconciled profiles | Runs a capped final check on weak/short submaps. |
| `PIPE.gaussian_submaps.quality_finalize_resident_only` | `true` | Avoids paging in passive maps just for finalization. |
| `PIPE.gaussian_submaps.quality_finalize_max_submaps` | `2` to `5` | Maximum submaps finalized at sequence end. |
| `PIPE.gaussian_submaps.quality_finalize_max_observed_frames` | `256` or `0` | Finalize short-observation submaps; `0` disables this criterion. |
| `PIPE.gaussian_submaps.quality_finalize_min_keyframes` | `8` | Finalize submaps with fewer keyframes. |
| `PIPE.gaussian_submaps.quality_finalize_max_hybrid_l1` | `0.010` | Finalize submaps whose cached hybrid RGB L1 is above threshold. |
| `PIPE.gaussian_submaps.quality_finalize_insert_enable` | `false` stable, `true` ablation | Attempts sequence-end Gaussian top-up. Keep disabled for stable rows until proven helpful. |
| `PIPE.gaussian_submaps.quality_finalize_insert_include_frozen` | `false` stable, `true` ablation | Allows top-up into frozen submaps. |
| `PIPE.gaussian_submaps.quality_finalize_insert_frames` | `3` to `8` | Number of stored observations used for top-up. |
| `PIPE.gaussian_submaps.quality_finalize_insert_max_new_gaussians_per_frame` | `1500` to `2000` | Temporary insertion cap used only during top-up. |
| `PIPE.gaussian_submaps.quality_finalize_insert_acceptance_enable` | `true` | Rolls back top-up if held-out readiness regresses. |
| `PIPE.gaussian_submaps.quality_finalize_insert_acceptance_cameras` | `5` | Held-out validation cameras used by the top-up acceptance guard. Set to `0` to make the guard inactive. |
| `PIPE.gaussian_submaps.quality_finalize_insert_min_readiness_delta` | `0.0` | Required readiness improvement to keep inserted Gaussians. |

Current evidence says unguarded sequence-end top-up is harmful on corrected
`room0`. Guarded top-up is useful as a safety mechanism but should remain an
ablation, not the stable default.

## Pose Graph And Loop Closure

| Key | Values | Meaning |
| --- | --- | --- |
| `PIPE.pose_graph.apply_mode` | `disabled`, `sequential_noop`, `synthetic`, `loop_candidates` | Controls whether pose graph corrections are only checked or applied. |
| `PIPE.pose_graph.dry_run` | `true`, `false` | Detects/verifies loop candidates without changing transforms. |
| `PIPE.pose_graph.solve_only` | `true`, `false` | Builds and solves the graph without applying corrections. |
| `PIPE.pose_graph.apply_correction_scale` | `0.75` in `recommended_longspan_robust` | Applies a fixed fraction of each accepted rigid-submap correction. The validated 0.75 setting reduces repeated-update over-correction without an extra solve or observation-render pass. |
| `PIPE.gaussian_submaps.pose_graph_overlap_roundtrip_check` | `true`, `false` | Read-only safety check for sequential submap transforms. It verifies each accepted overlap edge equals `inverse(to.local_to_global) * from.local_to_global` and that the edge inverse round-trips to identity. |
| `PIPE.gaussian_submaps.pose_graph_overlap_roundtrip_tolerance` | `0.0001` typical | Maximum transform max-absolute error allowed by the overlap round-trip check. |
| `PIPE.pose_graph.candidate_interval_frames` | `30` typical | Loop-candidate scheduling interval. |
| `PIPE.pose_graph.loop_min_submap_separation` | scene/profile-dependent | Rejects recent/sequential neighbors as loop candidates. |
| `PIPE.pose_graph.max_candidate_pairs_per_check` | `16` | CPU cap for candidate checks. |
| `PIPE.pose_graph.max_verified_loops_per_check` | `4` | CPU cap for accepted loop constraints per check. |
| `PIPE.pose_graph.information_policy` | `scalar`, `normalized_diagonal`, `confidence_diagonal` | Controls pose-graph residual weighting. `confidence_diagonal` is the current robust loop-profile choice. |
| `PIPE.pose_graph.loop_robust_kernel` | `none`, `huber`, `cauchy` | Robust loop-edge loss. `cauchy` protects against high-confidence but wrong loop edges. |
| `PIPE.pose_graph.loop_robust_scale` | `1.0` typical | Scale parameter for the robust loop kernel. |
| `PIPE.pose_graph.apply_health_gate` | `false` baseline, `true` robust loop profiles | Turns loop apply into observe-only when frontend raw tracking health is poor. Detection and verification still log. |
| `PIPE.pose_graph.apply_health_max_raw_failed_ratio` | `0.02` | Maximum hard raw tracking failure ratio before loop apply is blocked. |
| `PIPE.pose_graph.apply_health_max_raw_non_good_ratio` | `0.10` | Maximum non-good raw tracking ratio before loop apply is blocked. |
| `PIPE.pose_graph.apply_health_max_raw_failed_streak` | `5` | Maximum observed hard-failure streak before loop apply is blocked. |
| `PIPE.pose_graph.apply_health_min_frames` | `30` | Minimum frames before the run-health gate can block apply. |
| `PIPE.pose_graph.apply_observation_gate` | `true` in robust loop profiles | Evaluates selected held-out loop observations before mutating submap transforms. This catches corrections that improve graph error but damage local rendered/depth evidence. |
| `PIPE.pose_graph.apply_observation_defer_nonresident` | `true` in robust loop profiles | If an observation-gate sample needs a nonresident SDF submap, defer apply and request prefetch instead of synchronously paging during the critical apply path. |
| `PIPE.pose_graph.apply_observation_defer_retry_frames` | `120` | Number of frames a deferred apply remains eligible for retry after its endpoint maps are pinned/prefetched. |
| `PIPE.pose_graph.apply_observation_pin_deferred_maps` | `true` in robust loop profiles | Pins loop endpoint maps while a deferred observation retry is pending. |
| `PIPE.pose_graph.apply_observation_prefetch_loop_endpoints` | `true` in robust loop profiles | Prefetches SDF maps touched by accepted/exported loop constraints before observation-gated apply needs them. |
| `PIPE.pose_graph.apply_observation_prefetch_max_maps` | `4` | Maximum number of loop endpoint maps requested by the prefetch path. Larger values reduce apply latency risk but reserve more residency budget. |
| `PIPE.pose_graph.apply_observation_prefetch_pin_ttl_frames` | `120` in robust loop profiles | Keeps recently prefetched loop endpoint maps protected from residency and CPU-cache spill for a bounded number of frames. This avoids re-paging the same old endpoint during later retry/terminal loop apply without permanently pinning all loop maps. |
| `PIPE.pose_graph.apply_observation_skip_resident_trim` | `true` in robust loop profiles | Avoids immediately trimming maps that are resident specifically because the observation gate or deferred retry needs them. |
| `PIPE.pose_graph.apply_observation_terminal_sync_page_in` | `true` in robust loop profiles | Allows terminal end-of-run apply to synchronously page in endpoint maps if a deferred retry cannot complete before the sequence ends. Keep bounded by the prefetch map cap. |

Pose graph correction must update only shared submap transforms. TSDF voxels and
Gaussian tensors remain in local submap coordinates. Export both raw and
corrected trajectories when loop apply is enabled.

## TUM Accuracy Runner Profiles

`run/tum_loop_validation.py` is the reproducible runner for TUM ATE/RPE and
loop-closure checks.

Useful lifecycle/profile switches:

- `--tsdf-input-mode on_demand --tsdf-input-prefetch on
  --tsdf-input-tensor-bridge direct`: selects the clean streaming path.
- `--submap-stats-interval-frames 10`: bounds stats output while keeping
  actionable timing history.
- `--tsdf-lifecycle-profile paper_parity`: capacity-driven TUM paper-style
  candidate creation with larger frame/occupancy ceilings, disabled occupancy
  lookahead, weak tracking as candidate warming, and
  `tracking_boundary_handoff_policy: failed_only`. It still allows validated
  visibility-based primary switches by default because full `fr2_xyz` testing
  showed that disabling them can hurt ATE. Use
  `--allow-visibility-handoff false` only for capacity-only lifecycle ablations.
- `--loop-profile recommended_longspan_robust`: the current helper-remediation
  loop profile, based on `recommended_longspan` plus
  `information_policy: confidence_diagonal`, `loop_robust_kernel: cauchy`, the
  run-health apply gate, loop-observation endpoint prefetch, nonresident
  observation deferral, and deferred endpoint pinning.
- `--allow-visibility-handoff true|false`: controls whether a tracked
  candidate may become primary only because it explains more of the current
  view than the existing primary. Defaults to legacy `true`; disabling is useful
  to test strict capacity-only handoff, but is not the current paper-parity
  default.
- `--basic-debug-tracking-start-frame` and
  `--basic-debug-tracking-end-frame`: write bounded BasicEngine tracker/raycast
  diagnostics to `basic_tracking_diagnostics.csv`.
- `--basic-nonfinite-recovery`: opt-in diagnostic recovery that rejects
  non-finite BasicEngine tracker poses and restores the previous finite pose.

## Replica Quality Runner Metric Modes

`run/hybrid_validation.py` records `metric_mode` in each quality summary.

For uncontaminated live throughput measurement, also pass
`--gaussian-submaps-debug off --gaussian-insert-diagnostics off
--disable-render-overlap-diagnostics`. Quality/final-map runs may deliberately
enable some of these reductions to produce compositor evidence; do not compare
that timing directly with a clean throughput run.

- `dataset_depth`: the historical Replica path. The runner uses
  `scripts/metric_general.py` with the dataset root's `camera/` and `depth/`
  folders. This mode is preferred whenever the rendered image dimensions match
  the dataset RGB images.
- `sampled_eval`: the resized/sampled eval path. The runner uses
  `scripts/metric_sampled_eval.py` against the eval-produced `val/gt` images.
  This is used when a full-resolution prepared scene is evaluated at
  half-resolution, so the raw dataset images cannot be compared directly to the
  half-resolution renders.

The runner removes stale `results.json` and `per_view.json` files before
scoring. This prevents old metric outputs from being reused after a failed
metric run.
  Leave this off for untouched legacy baseline rows.

Helper audit scripts:

- `scripts/summarize_submap_transitions.py`: summarizes transition triggers,
  handoff moves, sequential transform deltas, and weak/failed counters at moves.
- `scripts/rescore_saved_trajectories.py`: recomputes legacy mean ATE and actual
  ATE RMSE for saved `val/pose` and `val/corrected_pose` folders without
  modifying old outputs.

## Replica Quality Runner Profiles

`run/hybrid_validation.py` is the main reproducible runner for Replica quality
and hybrid submap checks.

Useful labels:

- `basic_sdf`: BasicEngine SDF-only baseline.
- `basic_unified_gaussian`: BasicEngine unified Gaussian baseline.
- `multi_hybrid_warm_fill`: MultiEngine SDF plus local Gaussian submaps.

Useful Gaussian profiles:

- `thor_quality_reconciled`: unchanged L1-only baseline.
- `thor_quality_reconciled_perceptual`: validated quality profile; adds a 0.2
  D-SSIM term and improves all three image metrics on all eight Replica scenes.
- `thor_quality_reconciled_spatial64` and
  `thor_quality_reconciled_adaptive_spatial64`: tiled insertion ablations. They
  improved some perceptual rows but were not retained because of their cost.
- `thor_quality_reconciled_finalize`: older finalization profile.
- `thor_quality_reconciled_frozen_refine`,
  `thor_quality_reconciled_xboundary`,
  `thor_quality_reconciled_xboundary_accept`,
  `thor_quality_reconciled_xboundary_readiness_accept`,
  `thor_quality_reconciled_xboundary_sdferror_accept`: controlled ablations,
  not stable defaults. `xboundary_accept` keeps the rollback safety guard;
  the readiness and SDF-error variants additionally change frozen-refinement
  observation selection.

Always check the summary CSV fields `input_dir` and `gt_dir` before trusting a
cross-scene result. Runs generated before the resolver fix may have scene names
that do not match their actual input path.

## Required Reporting Rules

- Replica rendering-quality runs should use GT poses for fair paper-style
  quality evaluation.
- TUM ATE/RPE runs should use standard raw/preprocessed associations without
  unfair filtering or offline refinement.
- Main loop-closure results must be online timed runs: detection, verification,
  graph solve, transform apply, and rendering all happen during the run.
- Post-run mesh/image export is allowed only as deterministic export from
  already live-corrected submaps, and should be labeled as export-only.
