# Desktop development profiles

These overlays use the latest final EGAS-SLAM settings from `configs/final`.
They default to Replica at **600 × 340** and retain TUM at **640 × 480**. This is a
new desktop environment, not a rerun of the Thor timing benchmark.

`train --resolution full` selects native **1200 × 680** Replica images and
intrinsics `[600, 600, 599.5, 339.5]`, with `downscale_factor: 1`. Prepared local
full-resolution data must exist or be supplied with `--input-dir`. A complete
room0 run has been tested on this RTX 3070; this is not a guarantee for every
scene or larger future map. `--capture-every N` enables the optional visual
observer described in [DEMO_GUIDE.md](../../DEMO_GUIDE.md). Capture adds substantial
work and its timings must not be used as benchmark rates.

Office0 and room0 are already prepared locally. From the repository root:

```bash
source scripts/desktop_env.sh
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene office0 --frames 60
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene office0
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset tum --scene fr1_desk
```

The launcher prints each new run directory. A run contains:

- `record/config.yaml`: complete effective configuration, outside the trainer's
  clearable map directory.
- `record/run.json`: configuration and binary hashes, source identity/status,
  input support, environment, command, exit status and whole-command wall time.
- `record/native.log`: all native output, also streamed to the terminal.
- `record/telemetry.csv`: asynchronous per-second process RSS and whole-device GPU
  memory, utilisation, temperature and power. Observed peaks are recorded with
  their scopes in `run.json`; sampling can miss brief peaks.
- `map/`: the saved hybrid map, trajectory and native timing/diagnostic outputs.

Use `--frames N` for the first N frames, `--no-save` to skip large checkpoints,
`--name NAME` for a memorable unique run name, and `--dry-run` to validate inputs
and write a configuration without launching the GPU executable. Existing run
names are refused. A dry run reserves its name; choose another for execution.

**Training can clear its configured workspace.** Always use the launcher for a
new run. Do not manually rerun a saved training configuration if its map matters.
Evaluation reads the saved map and writes a newly named image directory:

```bash
"$EGAS_PYTHON" scripts/desktop_run.py eval \
  --run "$HOME/egas_workspace/runs/NAME_PRINTED_BY_TRAINING" --views 20
```

This uses the accepted final saved-map renderer settings, preserves the actual
training frame range and calibration, and selects uniformly spaced observed
views. `--views 0` renders all recorded views. It does not run training again.
The convenience sample is not the historical 500-view Thor benchmark selection.

The default workspace is `$HOME/egas_workspace`; `--workspace-root` changes it.
All inputs, executable paths and output paths must remain on local storage.
External mount paths and output symlink escapes are rejected. TUM scenes reuse
existing locally prepared data. Replica preparation reuses existing full-resolution
prepared files or converts local raw files; it then creates a dedicated half copy
without downloading or touching the external backup:

```bash
"$EGAS_PYTHON" scripts/desktop_run.py prepare --scene room0
"$EGAS_PYTHON" scripts/desktop_run.py train --dataset replica --scene room0
```

Preparation writes `$HOME/egas_workspace/data/replica_half/SCENE` and refuses
to overwrite it. Newly required full-resolution intermediates are kept in
`data/replica_prepared`. Use `--raw-root` for a different local raw Replica root or
training's `--input-dir` for a different local prepared scene matching the
selected Replica resolution (native resolution for TUM).

Replica uses physically resized images, calibration `[300, 300, 299.75, 169.75]`,
image shape `[600, 340]` and downscale factor `1`. This keeps both the TSDF tracker
and the Gaussian camera calibration consistent: the frozen TSDF constructor
uses reader-level metadata, which it does not scale for on-the-fly resizing.
The inherited preprocessing helper uses area interpolation and JPEG re-encoding
for colour, nearest-neighbour interpolation for depth, and unchanged depth units.
Each newly prepared Replica scene has a `desktop_preparation.json` receipt. Do not halve the calibration
again or use the old `configs/release/*half*` presets as final EGAS settings.

All final algorithm parameters are preserved: estimated tracking, immediate
Gaussian updates, coupled local submaps, pipeline loop closure and CPU-backed
SDF paging. No ground-truth camera poses are used for tracking. `--no-save` only
removes final map export, which is outside native processing FPS. The launcher
records whole-command wall time separately from the native processing trace.

Telemetry is sampled without CUDA synchronisation. Its monitoring cost has
not been independently bounded; recorded desktop measurements include it. GPU
memory is whole-device usage (including desktop applications), whereas RSS is
only the native process. These are not the thesis's cgroup memory measurements.
