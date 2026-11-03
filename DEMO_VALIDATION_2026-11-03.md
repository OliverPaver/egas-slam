# Demonstration validation — 3 November 2026

The three deliverables are complete and stored locally. The demo changes extend
the desktop development checkout based on final Thor release
`c944954d7000d2c905975c784f30a99ff7f3ff04` and desktop parent
`150503df256a3bb389e136c2bd0f1f1f6b731f25`. Historical benchmark files and final
configuration templates were not modified.

## Native execution and capture

The build passed with `./egas build`. The executable used for both complete
captures has SHA-256:

```text
7a9654397750f542a3fc17ff656040e3e49f399b80c9cf8183f91bf9279933ce
```

| Check | Observed result |
|---|---|
| Full-resolution Replica feasibility | Complete 2000-frame room0 run completed on the RTX 3070 |
| Indexed capture smoke | 60-frame native office0 capture completed; eight snapshots verified |
| Capture disabled | 60-frame native office0 run completed; no demo directory created |
| Replica room0 demonstration | 2000 frames, 1200 × 680, 204 snapshots, exit 0 |
| TUM office demonstration | 2485 frames, 640 × 480, 166 snapshots, exit 0 |
| Corrected trajectory exports | All 2000 / 2485 expected pose files present |
| Room0 applied transforms | Zero; the video makes no room0 correction claim |
| Office applied transforms | 45 recorded owner-transform updates across two correction bursts |
| Selected correction | Frames 1920–1924, 19 owners; maximum owner-origin translation 0.156097 m |

Both recordings use estimated poses. `scripts/demo_validate.py` checked complete
run receipts and configuration hashes, finite asset contents, mesh indices,
image dimensions, full-sequence coverage, rigid placements and mesh timestamps.
For office, newly logged owner transforms were also compared with the actual
committed SDF and Gaussian placements in the following captured state. These
checks passed. Learned Gaussian quaternion parameters are normalised when
rendered; the validator checks that they are finite and nonzero, without
incorrectly requiring the raw learned parameters to be unit quaternions.

`build_desktop/desktop_filesystem_regression` still passed both recursive-removal
overloads with the host `libstdc++.so.6` provider. Python syntax checks and
`git diff HEAD^ HEAD --check -- . ':!viewer/vendor'` passed for project-owned
changes. The unmodified Three.js vendor file retains one upstream mixed-indent
whitespace warning; its source was preserved verbatim.

The captures are visualisation evidence, not timing or accuracy benchmarks.
Capture adds meshing, native rendering, transfers, synchronisation and disk IO.
No bitwise-equivalence claim is made between captured and uncaptured executions.

## Viewer and media

Playwright with system Chrome verified live-follow advancement during the actual
room0 run, playback advancement, seeking, scene selection, mouse orbit and the
Gaussian-detail toggle. No JavaScript page errors or viewer error banner were
observed. The selected correction annotation showed 15.6 cm, matching the
recorded owner transform. A fresh `./egas viewer --port 0` launch loaded both
default recordings through the local API.

| Deliverable | Verified media properties |
|---|---|
| Interview MP4 | 70.00 s, 1920 × 1080, H.264, 30 fps, approximately 11.8 MB |
| GitHub preview MP4 | 20.00 s, 1920 × 1080, H.264, 30 fps, approximately 4.2 MB |
| GitHub GIF | 20.01 s, 960 × 540, approximately 3.4 MB |
| Poster PNG | 1920 × 1080 |

FFprobe inspected the encoded streams; FFmpeg decoded each complete MP4 and GIF
without errors. Encoded frames were inspected at opening, map growth, submap
comparison, before/after correction, completed-map orbit and ending. The final
preview starts with a reconstructed-room view. Videos are silent with captions;
their footer explicitly identifies adjusted replay timing. Source frame indices,
hold times and inspection-camera changes are recorded in `edit_decisions.json`.
The final orbit's native image pair is labelled with its original online frame.

The demonstration uses 1 cm clustered display meshes, sampled Gaussian
ellipsoids and view cutaways. The office correction panel isolates submap 4 for
clarity. Native hybrid image pixels are separately captured using all current
owner Gaussians. The scene change between Replica and TUM is labelled.

## Locations and reproduction

- Share folder: `/home/oliver/Downloads/EGAS_SLAM_Demo_2026-11-03/`.
- Authoritative exports, edit decisions and media hashes:
  `/home/oliver/egas_workspace/deliverables/EGAS_SLAM_Demo_2026-11-03/`.
- Capture runs: `/home/oliver/egas_workspace/runs/demo_replica_room0_full/` and
  `/home/oliver/egas_workspace/runs/demo_tum_office/`.
- Browser checks and inspected frames:
  `/home/oliver/egas_workspace/validation/demo/`.
- Build, capture and export logs: `/home/oliver/egas_workspace/logs/demo_*.log`.
- Editable implementation and reproduction commands: [DEMO_GUIDE.md](DEMO_GUIDE.md).

The copies in Downloads were hash-checked against the verified original exports.
The viewer depends on the retained local capture folders; the videos, GIF and
poster can be shared independently. No required validation task remains open.
