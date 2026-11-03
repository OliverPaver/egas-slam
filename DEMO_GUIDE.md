# EGAS-SLAM visual demonstration

The demonstration has three outputs: a narrated-by-captions interview film, a
short GitHub preview, and an interactive four-panel viewer. All images and map
states come from actual EGAS-SLAM runs on the local desktop. No reference mesh,
ground-truth camera tracking or synthetic loop correction is used.

## Open the viewer

```bash
cd /home/oliver/EGAS-SLAM
./egas viewer --open
```

The default scenes are the two prepared recordings in
`/home/oliver/egas_workspace/runs/demo_replica_room0_full` and
`/home/oliver/egas_workspace/runs/demo_tum_office`. The server binds only to
`127.0.0.1`. Use `--port 0` to choose an available port automatically. Stop it
with Ctrl+C. No internet connection is needed; Three.js and its MIT licence are
included under `viewer/vendor/`.

- **Play / slider:** replay or seek through recorded processing states.
- **Scene selector:** switch between synthetic Replica room0 and real TUM office.
- **Drag / scroll over a 3D panel:** rotate and zoom its view. Scene viewpoints
  are restored at each newly displayed state to keep replay views stable.
- **Gaussian detail:** show or hide sampled, owner-coloured Gaussian ellipsoids.
- **Follow live:** follow newly completed capture snapshots from an ongoing run.
  This is periodic observation with capture latency, not a hard real-time GUI.

The four panels show:

1. **From camera to reconstruction:** a matched RGB input and the native
   owner-local SDF/Gaussian hybrid image. This is the real hybrid renderer with
   all Gaussians belonging to that camera's owner, not a mesh screenshot or the
   separately configured final saved-map compositor.
2. **The map grows:** reconstructed SDF surfaces, corrected past camera path and
   current camera. Display cutaways remove obstructing walls/ceilings; they do
   not change the map or reconstruction results.
3. **Overlapping local maps:** two adjacent owners shown separately, from an
   identical world viewpoint, with consistent colours. Surface geometry gives
   context to the Gaussian ellipsoids. Gaussian detail is sparse in this hybrid
   system; these are not complete standalone Gaussian room reconstructions.
4. **Loop closure:** actual recorded owner placements. During office correction,
   submap 4 is shown in detail with its previous placement in white. The
   translation annotation comes from its recorded transform. Other panels
   retain the wider map context. Orange edges join verified submap graph nodes.

Room0 has no applied loop correction in this recording. Office supplies the real
correction example. The scene change is explicit in both videos.

## Record a new run

```bash
# Native 1200 × 680 Replica, using local full-resolution prepared inputs.
./egas train --dataset replica --scene room0 --resolution full \
  --capture-every 10 --no-save --name my_room_demo

# Native 640 × 480 TUM office, with all 2485 frames.
./egas train --dataset tum --scene fr3_office \
  --capture-every 20 --no-save --name my_office_demo

# In a second terminal, follow one of those runs while it is still executing.
./egas viewer /home/oliver/egas_workspace/runs/my_room_demo --open
```

Use a fresh run name. `--no-save` skips the large final training checkpoint;
it preserves the demo capture, trajectory and logs. Omit it when a reloadable
training checkpoint is needed. Half-resolution Replica remains the default.
Full-resolution selection changes both input calibration and engine size; it
does not simply enlarge exported images. Only the tested room0 sequence has
complete-run full-resolution feasibility evidence on this 8 GB RTX 3070.

Capture is **disabled by default**. The launcher writes `PIPE.demo_capture` with
`enabled`, `interval`, `scene` and `dataset`. Snapshots are also taken on owner
changes and owner-transform epoch changes, so correction batches are retained
even when the regular interval is larger. The observer does not invoke graph
solving, optimiser updates, final-map saving or SDF page reloads. It meshes
resident maps and uses its own raycast buffers. It adds GPU work, synchronisation
and disk IO, so these runs must not be used as throughput benchmarks. Adding
capture can alter execution timing; bitwise equality with an uncaptured run is
not claimed.

The completed recordings occupy roughly 14 GB together. A 1 cm vertex-clustering
step reduces display-mesh storage, while native hybrid images remain unchanged.
Only a deterministic sample of Gaussian ellipsoids is exported (up to 8000 per
owner); the viewer displays at most about 2500 per owner at their real 2-sigma
scales. This sampling is for representation inspection, not image-quality
evaluation. Offline map views retain the last captured surface for a passive
owner; metadata records its capture frame. Capturing does not page old maps back
onto the GPU merely to refresh a display asset.

## Export the two videos

```bash
# Terminal 1
./egas viewer --port 8766

# Terminal 2
./egas demo-export
```

The exporter requires FFmpeg/FFprobe, system Chrome and Python Playwright. This
desktop is already configured. The Python dependency can be installed into the
environment selected by `scripts/desktop_env.sh` with
`python -m pip install playwright==1.56.0`; the exporter uses system Chrome rather
than downloading a second browser. `--help` lists alternate scene, output and
server paths.

Outputs are written to
`/home/oliver/egas_workspace/deliverables/EGAS_SLAM_Demo_2026-11-03/`:

- `EGAS_SLAM_Interview.mp4`: 1080p, approximately 70 seconds, captions, no audio.
- `EGAS_SLAM_GitHub_Preview.mp4`: 1080p, 20 seconds.
- `EGAS_SLAM_GitHub_Preview.gif`: 960 × 540 animated README preview.
- `EGAS_SLAM_Poster.png`: 1080p still for a video link or project header.
- `edit_decisions.json`: exact scene, capture index, display options and hold time.
- `media_validation.json`: dimensions, durations, decoding checks and file hashes.

Recorded states are compressed in time and correction steps deliberately held.
The final orbit moves a virtual inspection camera around the completed SDF
surface; it is not additional SLAM tracking or a quantitative novel-view test.
The accompanying native image pair is labelled with its original recorded frame.
The video footer states that playback timing is adjusted. No timing or accuracy
claim from the thesis is transferred to this desktop demonstration.

The exporter caches PNG frames under `rendered_frames/` and keys them by display
state and viewer-source hash. It can be rerun after an interrupted export. Those
intermediate PNGs are disposable after the final videos have been checked.

## Verify a capture

```bash
source scripts/desktop_env.sh
"$EGAS_PYTHON" scripts/demo_validate.py \
  /home/oliver/egas_workspace/runs/demo_replica_room0_full \
  /home/oliver/egas_workspace/runs/demo_tum_office
```

Validation checks completed native runs, configuration hashes, full-sequence
coverage, asset dimensions and finite values, mesh indices, rigid transforms,
causal mesh timestamps and agreement between logged corrections and committed
SDF/Gaussian placements. Each run receives a validation summary and asset-hash
manifest under `record/`. The capture-image PSNR in that engineering report is
a sanity check on these particular JPEG snapshots, not a thesis result.

## Implementation map

- `slam/demo_capture.cpp`: optional observer, native hybrid snapshots, mesh and
  Gaussian metadata, atomic snapshot publication.
- `slam/slam_pipeline.cpp/.h`: completed-frame capture hooks and actual owner
  transform event hooks.
- `InfiniTAM/ITMLib/Core/ITMMultiEngine.h/.tpp`: independent resident-only meshing
  and raycasting resources; compact indexed display meshes.
- `scripts/desktop_run.py`: correct full-resolution calibration and capture flags.
- `scripts/demo_viewer.py`: read-only local HTTP server for complete/live captures.
- `viewer/`: editable Three.js interface, colour legends and scene viewpoints.
- `scripts/demo_export.py`: deterministic edit decisions and browser/FFmpeg export.
- `scripts/demo_validate.py`: causal asset and correction checks.

## Visual references

The design uses the clear separation of input, reconstruction and global map
seen in the [GPS-SLAM project demonstration](https://gapszju.github.io/GPS-SLAM/),
the representation views on the [MonoGS project page](https://rmurai.co.uk/projects/GaussianSplattingSLAM/),
and the distinction between online growth and completed-map inspection on the
[SplaTAM project page](https://spla-tam.github.io/). The
[LoopSplat project page](https://loopsplat.github.io/) also informed the use of
stable overview cameras. All visual assets in these outputs are newly captured
EGAS-SLAM states; imagery from those systems is not reused.

The completed checks and deliverable details are recorded in [DEMO_VALIDATION_2026-11-03.md](DEMO_VALIDATION_2026-11-03.md).
