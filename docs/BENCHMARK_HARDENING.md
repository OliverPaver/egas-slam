# Measurement and validation

This guide explains how to interpret software outputs and preserve a useful
record of a run. It does not publish numerical results or make the standalone
project a replacement for the separately retained benchmark campaign harness.

## Source and run identity

Verify [SOURCE_MANIFEST.json](../SOURCE_MANIFEST.json) with the release hygiene
tool before building. Keep the exact configuration, source identity, compiler
options, dependency/container identity and executable hash with new outputs.
The manifest identifies source; it does not certify a different binary or
reproduce the timing of a prior build.

Record runtime environment overrides as well as YAML. CUDA precision settings,
instrumentation, map capacity and the input-loading policy can affect execution.
The proposed and original presets have distinct BasicEngine capacity
instrumentation settings. A full runtime measurement profile does not mean every
optional debug log is enabled, or establish that instrumentation has no effect
on asynchronous work.

## Completion and timing

Distinguish input processing from terminal native work, model serialization,
offline rendering and metric calculation. A frame-loop duration, a complete
native processing duration and a whole-command wall duration answer different
questions. Use the actual emitted field definitions; do not relabel one as
another or add a stage that was already included.

GPU work must have reached the declared completion boundary before assigning
its duration. Host return from an asynchronous CUDA submission alone is not GPU
completion. Throughput is also distinct from per-frame latency or the ability
to keep up with sensor arrivals.

Keep process exit status, expected and processed frame counts, finite-state
checks, terminal work and required exports together. A nonzero exit, missing
output or unclosed worker cannot become a successful observation just because a
metric file exists. Preserve failure evidence before any separately authorized
recovery or rerun; a new native invocation is a new attempt.

## Tracking and pose support

Raw ICP status, the configured failure policy, fusion suppression and a saved
pose are separate facts. The final native training templates retain
`tracking_failure_mode: legacy_ignore` while rejecting nonfinite poses. Do not
describe them as suppressing every raw tracking failure.

Keep frame IDs and timestamps with trajectories. Estimated-pose construction
does not consume future ground-truth poses. The offline pose tools perform
explicit pairing and report coverage, rigid SE(3) translation error and relative
pose error. Mean translation error and translation RMSE are different fields;
retain their units and alignment labels.

Missing poses must be reported under an explicit policy. Dropping unmatched
frames, interpolating them or holding a preceding pose changes the question the
metric answers. Such a diagnostic must retain its own support and must not
silently replace a complete-frame result. Saved corrected trajectories also
have different semantics from uncorrected online tracker output.

## Image evaluation

Keep construction pose mode, saved-map identity, rendering pose source,
resolution, frame selection and metric implementation explicit. The final
training templates use estimated poses. Supplying ground-truth construction
poses is a different study. Evaluating observed training frames does not become
held-out novel-view evaluation by using a smaller frame list.

Saved-map evaluation uses the separate templates and records the selected frame
IDs. Retain aggregate metrics, per-view values, coverage and evaluation
configuration together. Do not replace failed or absent views with an
unlabelled subset. A new rendering derivative preserves the original map's
native run identity and timing.

The image scorer applies the required [-1,1] LPIPS input conversion and uses
external model weights. Record the weights and software versions. PSNR, SSIM and
LPIPS are distinct quantities; their image count is not the number of independent
native repetitions. Nonfinite or unavailable values need explicit handling
rather than substitution with zero.

## Resource and repeated-run summaries

Preserve resource labels: allocator use, RSS, container memory, host availability,
map bytes and sampled platform power cannot be treated as interchangeable
measurements. Sampling has finite temporal resolution. Platform power does not
isolate GPU power or provide idle-subtracted energy without additional evidence.

When aggregating repeated runs, report actual completed and failed counts,
metric-specific support and sample standard deviation. An equal-scene mean is
different from pooling all frames. Paired differences require the same paired
observations for each metric; a difference between two unequal marginal means
is not a paired estimate.

Use the maintained source, configuration and CPU checks in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Native build and scene validation
remain separate checks. Keep new measurements separate from the historical
results archive and from the source release's packaging validation.
