# EGAS-SLAM demonstration

[Watch the 65-second GitHub film](assets/demo/egas-slam.mp4), or use the
[animated preview](assets/demo/preview.gif) and [poster](assets/demo/poster.png).
The supplied media can be viewed without a CUDA installation or datasets.

## What the video shows

1. **Build local maps:** full-resolution Replica room0 input, native hybrid
   renderings, reconstructed SDF surfaces and two overlapping local maps.
2. **Recognise a revisit:** an earlier TUM office image beside the moving camera
   image. The shared desk and objects make the return recognisable. Colour and
   depth verify a constraint between submaps 0 and 18.
3. **Correct the map:** identical captured local geometry at its actual before
   and after placements. Both paths use the same historical camera samples;
   the dashed orange path is the previous estimate and teal is the correction.

The recordings use estimated camera poses on the desktop. Room0 contains no
applied loop correction in this recording; the office supplies that example.
Playback timing is chosen for presentation. The input video moves continuously,
while native renderings and map surfaces update at capture intervals. The final
inspection camera moves around recorded states; submap transforms are not
interpolated or exaggerated. This is a demonstration, not a speed benchmark.

[Media provenance](assets/demo/README.md) records the source and encoding.
The [technical edit record](docs/development/github_demo_v3.md) explains the
matched images, actual correction and validation in detail.

## Capture your own run

First follow the [desktop setup guide](docs/desktop_setup.md) and prepare the
selected scenes. Use a new run name each time:

```bash
# Replica room0: native 1200 × 680, with calibrated full-resolution input.
./egas train --dataset replica --scene room0 --resolution full \
  --capture-every 10 --no-save --name my-room-demo

# TUM office: native 640 × 480.
./egas train --dataset tum --scene fr3_office \
  --capture-every 20 --no-save --name my-office-demo
```

Full-resolution room0 has completed on the tested 8 GB RTX 3070. Other scenes
and longer maps can have different requirements. Omit `--resolution full` for
the default half-resolution Replica profile. `--no-save` retains the capture
and trajectory but skips the large reloadable training checkpoint.

Capture is disabled by default. It observes completed frames, resident maps,
owner changes and transform updates. It adds GPU work, synchronisation and I/O;
its timings should not be compared with the thesis benchmark. It does not
trigger additional loop solving, Gaussian optimisation or SDF page reloads.
A complete pair of demonstration captures can require approximately 14 GB.

## Open the browser viewer

Set `EGAS_WORKSPACE_ROOT` consistently with training, or use the exact run paths
printed by the launcher:

```bash
./egas viewer "$EGAS_WORKSPACE_ROOT/runs/my-room-demo" \
  "$EGAS_WORKSPACE_ROOT/runs/my-office-demo" --open
```

The local server binds to `127.0.0.1`. Stop it with Ctrl+C; `--port 0` chooses an
available port. Three.js and its licence are included, so playback needs no
internet connection.

- **Play, slider and speed:** replay or seek recorded processing states.
- **Scene selector:** switch between the supplied runs.
- **Panel headings:** enlarge a panel, then return to the four-panel overview.
- **Drag and scroll:** rotate and zoom map views; **Reset view** restores framing.
- **Gaussian detail:** toggle sampled owner-coloured Gaussian ellipsoids.
- **Before / corrected placement:** compare recorded placements at one viewpoint.
- **Follow live:** observe newly completed snapshots while a run is executing.

The four panels show the camera/native hybrid image, reconstructed SDF map,
overlapping local maps and loop correction. Surface geometry gives context to
the sampled Gaussian ellipsoids: the hybrid system does not represent the
entire room with Gaussians alone. A virtual cutaway makes interiors visible.
Passive-map display assets may predate their final finishing step; capture
filenames retain their frame identity. The viewer is not a quantitative
novel-view or geometry evaluator.

## Validate a recording

```bash
source scripts/desktop_env.sh
"$EGAS_PYTHON" scripts/demo_validate.py \
  "$EGAS_WORKSPACE_ROOT/runs/my-room-demo" \
  "$EGAS_WORKSPACE_ROOT/runs/my-office-demo"
```

Validation checks completed runs, frame coverage, configuration hashes, asset
shapes and finite values, mesh indices, rigid transforms and correction
consistency. It writes receipts and asset hashes under each run's `record/`.
A recording without accepted corrections is valid, but cannot supply a genuine
loop-correction example for the film.

## Exporting the authored videos

Export requires FFmpeg/FFprobe, Python Playwright and a compatible installed
Chrome/Chromium browser. The video-export environment used Playwright 1.56.0;
the default executable is `/usr/bin/google-chrome`. `--chrome` can select another
compatible browser path.

```bash
python -m pip install playwright==1.56.0
sudo apt install ffmpeg
```

`scripts/github_export.py` contains the three-stage edit used for the included
film. Its default source names are `demo_replica_room0_full` and
`demo_tum_office` under `$HOME/egas_workspace/runs`. The original raw RGB inputs
must remain available at the paths recorded in those runs. Captures and data
are not bundled with the repository.

```bash
source scripts/desktop_env.sh
"$EGAS_PYTHON" scripts/github_export.py --prepare-only
./egas viewer --port 8766 \
  --presentation "$HOME/egas_workspace/presentation/github_v3"

# In a second terminal, with the same Python environment:
python scripts/github_export.py
```

The exporter validates source identity, uses only map states captured at or
before the displayed input frame, streams screenshots to FFmpeg and checks
complete media decoding. It writes source hashes, edit decisions and validation
beside the output files. `--help` lists input/output overrides; `--draft`
creates a separate lower-frame-rate review copy.

This edit deliberately selects known matching views and a verified correction
from the original captures. A new run can produce different submap IDs or event
frames. Review and adapt the selection before exporting a different recording;
do not assume the same fixed frames contain an accepted loop.

The earlier interview film and four-panel edit remain reproducible through
`./egas demo-export`. Their timings and deliberate holds are documented in the
[previous edit record](docs/development/demo_revision_v2.md); the GitHub film
uses its separate continuous-playback exporter.
