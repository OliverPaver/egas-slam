# Demonstration assets

- `egas-slam.mp4`: the verified 65.2-second 1920 × 1080 H.264 film at 30 fps.
- `preview.gif`: a reduced-size animated README preview of the same complete
  film, with unchanged playback timing.
- `poster.png`: a still from the same film.
- `manifest.json`: file identities and encoding metadata.

The film contains estimated-pose desktop recordings of Replica room0 and TUM
Freiburg 3 office. Its loop example uses a verified revisit between submaps 0
and 18. The comparison places the same captured local surfaces and 118 historical
camera samples using native transforms recorded at frames 1919 and 1924.
The prior path is shown with an orange dashed line beside the corrected teal path.

The recordings are genuine processing outputs, replayed with chosen presentation
timing and inspection viewpoints. They do not establish a processing rate,
every-frame real-time performance or a new quantitative accuracy result.
The [demonstration guide](../../DEMO_GUIDE.md) explains capture and reproduction.

The source datasets retain their original terms: [Replica](https://github.com/facebookresearch/Replica-Dataset)
and [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset). The application
and viewer retain the notices in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
and `viewer/vendor/`. No dataset archive or model checkpoint is included here.
