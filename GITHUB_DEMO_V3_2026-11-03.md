# GitHub demonstration, revision 3

This revision makes the GitHub film a continuous explanation of EGAS-SLAM:
build overlapping local maps, recognise a revisit, then correct the map. It
uses the existing verified desktop recordings. No SLAM algorithm, native run
configuration or capture asset is changed. The earlier interview film and
interactive viewer are preserved.

## Edit and presentation

| Output time | Source | What the viewer sees |
|---|---|---|
| 0:00–0:46.667 | Replica room0, frames 600–1999 | Continuous RGB input, a growing SDF reconstruction, captured native hybrid images and neighbouring submaps with sampled Gaussian ellipsoids |
| 0:46.667–0:55.200 | TUM office, frames 1800–2040 | The current camera returns to the desk shown in an earlier image; the recorded verified revisit appears in the map |
| 0:55.200–1:05.200 | TUM office, placements at frames 1919 and 1924 | Simultaneous before/corrected geometry and camera paths, viewed with a slow inspection-camera motion |

There are no opening, handoff or closing freeze holds. The room excerpt covers
all three handoffs at source frames 650, 924 and 1798. Its 30-source-frame/s
pace is chosen for presentation, not inferred sensor timing or measured SLAM
performance. The office excerpt spans 8.506578 seconds of recorded sensor time;
30-fps output quantises it to 8.533333 seconds. Original RGB frames are sampled
at the nearest timestamp. Map captures always come from the latest recorded
online state at or before the displayed source frame.

The continuous input and captured hybrid image have separate frame labels.
Map updates were recorded every ten room frames and twenty office frames,
with additional captures at handoffs and corrections. Sparse recorded map
updates are not replaced with invented intermediate geometry.

## What demonstrates loop closure

The accepted office revisit links submaps **0 and 18**. The earlier image is
frame **1**; the accepted correction's supporting revisit image is frame
**1920**. The red toy aeroplane, plates, bottles and desk make the relationship
visible. The first captured record of the verified pair appears at frame
1919. The orange connection is a constraint between submap coordinate frames,
not an exported set of pixel feature matches.

Actual transform updates are applied over frames **1920–1924**. The ending
shows the retained pre-correction local SDF meshes at their actual before and
after placements. Both sides therefore use identical local geometry, including
the active map's retained pre-correction mesh. Only the owner placement differs;
new mapping between those frames is excluded from this comparison.

Both camera paths use the same **118** recorded camera-to-owner samples through
frame 1919. They are placed using the respective native owner transforms. The
right panel shows the corrected path in teal and the previous path in dashed
orange, at the same scale. This illustrates correction; it is not a
ground-truth error plot. No map-transform interpolation, exaggerated movement
or fabricated snap of the camera endpoint onto the start is used.

The local-map panel earlier in the film shows sampled Gaussian ellipsoids with
SDF surfaces for context. The correction panel uses SDF meshes to keep the
before/after comparison readable. The method's coupled Gaussian/SDF placement
is described in its caption; the comparison is not a full Gaussian rerender.
The meshes use the existing 1 cm display clustering and cutaway bounds. Retained
passive-map assets can predate final finishing, as recorded in their filenames.

## Reproduction and provenance

Follow the commands under **Current GitHub film** in [DEMO_GUIDE.md](DEMO_GUIDE.md).
The new components are:

- `scripts/github_export.py`: prepares original RGB assets, validates the edit,
  renders reproducible video segments and checks the final media.
- `viewer/github.html`, `github.css`, `github.js`: the dedicated three-stage
  presentation, separate from the interactive viewer.
- `scripts/demo_viewer.py --presentation <directory>`: serves prepared assets
  within the explicitly supplied local directory.

Prepared source material lives at
`/home/oliver/egas_workspace/presentation/github_v3/`. It contains local
hardlinks/copies, `story.json`, the identical historical path comparison and
SHA-256 provenance for all original RGB files. Native capture identity reuses
the completed-run receipts, validated asset manifests, configuration hashes
and snapshot metadata. The exporter never writes into the native recordings.

Outputs live at
`/home/oliver/egas_workspace/deliverables/EGAS_SLAM_GitHub_v3_2026-11-03/`:

- `EGAS_SLAM_GitHub_Preview.mp4`: 65.2 seconds, 1920 × 1080, 30 fps, silent.
- `EGAS_SLAM_GitHub_Preview.gif`: 960 × 540 animated preview at 10 fps.
- `EGAS_SLAM_Poster.png`: 1920 × 1080 still.
- `README_GitHub.md`: concise presentation copy and embedding examples.
- `edit_decisions.json`: every output frame's source frame, capture index,
  presentation time and inspection-camera angle.
- `source_provenance.json`: native-capture identities and original RGB hashes.
- `media_validation.json`: dimensions, durations, decoding checks, source and
  renderer identities, and SHA-256 hashes of the finished media.

Per-phase MP4 segments are cached with source and renderer hashes. Screenshots
stream directly into FFmpeg, avoiding thousands of intermediate PNG files.
Final outputs replace temporary files only after decoding and duration checks.

## Validation

Preparation and browser smoke checks passed: all expected source images exist,
every displayed map capture is causal, original frame endpoints are retained,
and the 118 generated comparison samples exactly match the independent
technical audit. Chrome successfully rendered mapping, all tested handoffs,
the revisit and both comparison endpoints without JavaScript errors.

The complete exporter additionally checks every output-frame response, fully
decodes the MP4 and GIF, verifies durations and dimensions, and writes the
finished-media results to `media_validation.json`. Independent visual inspection
of the final encoded video remains part of the release check; the report must
be read alongside that inspection, rather than treating decoding alone as a
visual-quality test.

## Completed release checks

The final export passed in Chrome with no JavaScript errors. FFmpeg decoded
the complete 65.2-second, 1956-frame H.264 video and the animated GIF without
errors. The MP4 is 1920 × 1080 at 30 fps; the GIF is 960 × 540 at 10 fps.
Decoded RGB and 3D regions were checked frame by frame, excluding text and the
progress indicator: no identical consecutive images occurred in those tested
regions. Original input timing and every causal map-state selection also passed.
Representative encoded frames were inspected at the start, late submap growth,
revisit, correction and ending. The final local-map camera framing has clearance
for the late submaps. Earlier v2 media hashes remain unchanged.

Validation commands: `python scripts/github_export.py`, `python -m py_compile
scripts/github_export.py scripts/demo_viewer.py`, and `git diff --check`. The
additional decoded-image motion check is retained locally in
`/home/oliver/egas_workspace/validation/github_v3/check_encoded.py`; its output
is `motion_validation.json`, tied to the final video hash.

The shareable package is copied to
`/home/oliver/Downloads/EGAS_SLAM_GitHub_v3_2026-11-03/`. No pending fixes remain
from this pass. Publishing the files is separate from this local media update.
