# Demonstration revision 2 — 3 November 2026

This revision improves framing, pacing and explanation of the existing local
recordings. Native capture code, SLAM algorithms, run configurations and captured
map contents are unchanged. The original videos remain in
`/home/oliver/Downloads/EGAS_SLAM_Demo_2026-11-03/`.

## Presentation changes

The interview film is 144 seconds, increased from 70 seconds. Its sequence is:

| Time | View and purpose |
|---|---|
| 0:00–0:04 | Reconstructed-room introduction |
| 0:04–1:04 | Chronological room reconstruction, with pauses around each local-map handoff |
| 1:04–1:12 | Enlarged, identically framed views of two overlapping submaps |
| 1:12–1:37 | Office reconstruction approaching the recorded revisit |
| 1:37–1:42 | Enlarged office detail before correction |
| 1:42–1:54 | Five actual recorded correction batches, held for 2.4 seconds each |
| 1:54–2:01 | Corrected placement, then a labelled before/after comparison |
| 2:01–2:19 | Slow nine-degree inspection orbit of the completed room |
| 2:19–2:24 | Closing hold |

The GitHub preview is 40 seconds, increased from 20 seconds. It also includes
submap-growth holds, an enlarged submap comparison and a slower correction
sequence. Both films remain silent, with explanatory captions.

Room cameras now look across the sofa, table and chairs from a higher angle.
The full extent of the late submap remains visible through the completed-map
orbit. Office correction has a separate close camera. Its final framing was
checked in the small panel and enlarged view; the camera was pulled back and
recentered to retain bottom clearance. The correction annotation now sits below
the 3D viewport, so it cannot cover map geometry.

The office's side-by-side local-map camera was fitted against the displayed mesh
vertices across all 166 captured states, after the existing cutaway bounds, with
a 16% margin. This prevents a later pair from extending beyond the panel edges.
Small office fragments consequently appear smaller in the overview; clicking
the panel heading provides a larger inspection view.

## What the correction shows

The office revisit is between maps **0 and 18**. Map **4** is selected to make
the resulting placement correction visible; it is not one of those two revisit
endpoints. The selected event spans source frames 1920–1924 and updates 19 map
owners. Its largest owner-origin translation is 0.1560966848 m, shown as
**15.6 cm**. This is displacement of a submap coordinate origin, not trajectory
error reduction.

Map 4 uses exactly the same captured mesh and Gaussian asset at frames 1919 and
1924. Both assets were last captured at frame 441, before the final finishing
step: the retained Gaussian file contains 7,804 primitives, while later frozen
model metadata reports 7,737. The viewer displays a sample of the retained file,
not an exact refresh of the final frozen model. The comparison shows identical
previously captured local contents at their actual recorded before and after
placements. There is no invented deformation, interpolation
of the correction or exaggerated displacement. Before/after replay is labelled
and uses the same camera. During the recorded update sequence, white wireframe
shows the old placement and colour shows the current placement.

The viewer distinguishes a verified revisit awaiting application from an
applied correction. The completed room0 recording explicitly says that it has
no loop correction. Owner states are labelled from recorded metadata; a new
owner does not by itself imply that the preceding owner's appearance is frozen.

## Interactive viewer

Click a panel heading to enlarge it; use **Four-panel overview** to return.
**Slow**, **Normal** and **Fast** set delays of 600, 300 and 100 ms between
recorded snapshots, subject to rendering time. These are replay settings,
not processing-speed claims. Mouse camera adjustments now persist as playback
advances; **Reset view** restores the prepared camera. The before/corrected
placement button pauses playback and shows the labelled comparison.

State rendering is queued to prevent overlapping scene/seek updates. Seeking
pauses playback and resets its button label. Export renders are cached against
the viewer source, camera settings and capture identity. Temporary images and
media are promoted only after successful rendering or decoding.

## Validation

- Python compilation and `git diff --check` passed for the changed files.
- `./egas demo-export --timeline-only` verified the 144/40-second edit plans,
  actual handoff frames and recorded correction range.
- Independent Chrome/Playwright regression passed 36 checks with no JavaScript
  errors. Checks covered native before/after transforms, asset identity,
  correction labels, annotation clearance, all four expanded views, scene
  selection, all playback speeds, seeking, the Gaussian toggle, persistent
  camera interaction and reset.
- Final correction framing was additionally checked at 1920 × 1080,
  1440 × 900 and 1280 × 720, including both recording and interactive layouts.
  The small panel naturally provides less detail at 720p; enlarging it is useful.
- The original complete-run capture validation remains applicable: 2,000 native
  1200 × 680 Replica frames and 2,485 native 640 × 480 TUM frames, using estimated
  poses. See [the original validation report](DEMO_VALIDATION_2026-11-03.md).

The independent checks, camera screenshots and inspected encoded frames are in
`/home/oliver/egas_workspace/validation/demo_v2/`. Exact edit decisions and media
validation are stored alongside the revised authoritative exports in
`/home/oliver/egas_workspace/deliverables/EGAS_SLAM_Demo_v2_2026-11-03/`.

Display meshes still use 1 cm vertex clustering and existing cutaway bounds;
Gaussian ellipsoids are sampled. Native input/hybrid images are unchanged.
Offline views can retain earlier captured surfaces and Gaussian samples of a
passive map, as identified by asset filenames and mesh timestamps. Playback timing is adjusted for explanation and does not
measure real-time performance. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for commands,
controls and capture boundaries.

## Final media and handoff

| File | Verified output |
|---|---|
| `EGAS_SLAM_Interview.mp4` | 144.00 s, 1920 × 1080, H.264 at 30 fps, 14.4 MB |
| `EGAS_SLAM_GitHub_Preview.mp4` | 40.00 s, 1920 × 1080, H.264 at 30 fps, 4.5 MB |
| `EGAS_SLAM_GitHub_Preview.gif` | 40.01 s, 960 × 540, 4.1 MB |
| `EGAS_SLAM_Poster.png` | 1920 × 1080 |

FFmpeg decoded both complete MP4 files and the complete GIF without errors.
FFprobe verified their dimensions, codecs and durations. Encoded frames were
inspected at opening, growth, overlapping submaps, office approach, correction,
the explicit before/after replay, completed-map orbit and ending. The final
caption says that the separate local maps are placed together in one global map.
No JavaScript errors occurred during the complete export.

The share folder is
`/home/oliver/Downloads/EGAS_SLAM_Demo_v2_2026-11-03/`. It contains the four media
files, an executable local viewer launcher, quick-start instructions, a GitHub
embedding example and the validation records. Copied media were SHA-256 checked
against the authoritative exports. No required task remains open.

Validation commands used for the revision:

```bash
python -m py_compile scripts/demo_export.py
git diff --check
./egas demo-export --timeline-only
./egas demo-export
```

The complete export includes automated Chrome rendering, FFprobe inspection,
full FFmpeg decoding and media hashing. The additional independent browser and
camera checks are recorded under the validation directory listed above.

## Changed source files

| Path | Change |
|---|---|
| `viewer/scenes/replica_room0.json` | Room overview and matching local-map cameras |
| `viewer/scenes/tum_fr3_office.json` | Office overview, local-map and correction cameras |
| `viewer/index.html` | Separate correction annotation strip and new replay controls |
| `viewer/style.css` | Enlarged panels, annotation layout and before/after labels |
| `viewer/viewer.js` | Comparison placement, accurate state labels, persistent cameras and queued updates |
| `scripts/demo_export.py` | Longer edit plans, focused shots, capture-aware caching and checked atomic exports |
| `DEMO_GUIDE.md` | Updated controls, reproduction commands and presentation boundaries |
| `DEMO_REVISION_V2_2026-11-03.md` | Revision and validation record |
