# Thor benchmark results

EGAS-SLAM was evaluated on NVIDIA Jetson AGX Thor against Photo-SLAM, MemGS and
DiskChunGS in the final `thor_repair_v3_20260910` campaign. These results describe
the frozen, instrumented Thor implementations. The current desktop port and
demonstration recordings are separate development work and do not replace these
measurements.

## Replica: reconstruction with estimated poses

Each system estimated its own camera poses and processed all 2,000 RGB-D frames
of each scene at 1200 × 680. Image quality was measured on the same 500
temporally stratified **observed views** per scene. Values below give equal
weight to the eight scenes (`room0`–`room2`, `office0`–`office4`), with five
independent runs per scene and system.

| System | PSNR ↑ (dB) | SSIM ↑ | VGG-LPIPS ↓ | Processing FPS ↑ | Full-program FPS ↑ | Mean peak container memory ↓ (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EGAS-SLAM** | **37.49** | **0.9634** | **0.1577** | **24.06** | **22.73** | **3.62** |
| Photo-SLAM | 31.67 | 0.9071 | 0.3211 | 18.02 | 17.11 | 6.40 |
| MemGS | 31.93 | 0.9085 | 0.3248 | 18.04 | 17.13 | 6.65 |
| DiskChunGS | 29.95 | 0.8978 | 0.2996 | 17.66 | 16.63 | 4.54 |

EGAS-SLAM has the best mean image scores and processing rate on each of these
eight scenes under this protocol. The scores evaluate completed-map rendering
from estimated poses at observed views; they do not measure held-out novel-view
quality or dense-surface accuracy.

## TUM RGB-D: final-corrected trajectory accuracy

The tracking comparison uses `fr1_desk`, `fr2_xyz` and `fr3_office` at 640 × 480,
with 573, 3,615 and 2,485 inputs respectively. Each system has five runs per
sequence. ATE is translation RMSE after rigid SE(3) alignment without scale,
using the complete final-corrected trajectory. Aggregate values weight the three
sequences equally.

| System | Final-corrected ATE ↓ (cm) | Processing FPS ↑ | Full-program FPS ↑ |
| --- | ---: | ---: | ---: |
| **EGAS-SLAM** | **2.89** | **35.36** | **30.94** |
| Photo-SLAM | 4.90 | 22.92 | 20.64 |
| MemGS | 3.61 | 22.47 | 20.34 |
| DiskChunGS | 3.38 | 22.11 | 19.78 |

EGAS-SLAM's aggregate accuracy advantage comes from more consistent desk runs.
All three alternatives have lower mean ATE on XYZ and office. Three Photo-SLAM
desk runs, one MemGS run and one DiskChunGS run contain 84 missing poses each;
the declared evaluation holds the preceding available pose for those inputs
and retains every run. Complete EGAS-SLAM exports also include frames reported
as failures by its raw tracker. These scores describe final outputs, not the
pose accuracy available at each online frame.

## Measurement boundaries

- **Operating conditions:** the comparisons use instrumented, unpaced local
  implementations on the same Thor device. They are not exact reproductions
  of every published execution recipe. Photo-SLAM's released TUM example is
  paced; its effect on the observed tracking losses was not isolated.
  Instrumentation overhead was not demonstrated to stay below the original
  2% bound. Prepared TUM depth uses 5,000 units/metre; Photo-SLAM and MemGS
  retain their released XYZ setting of 5,208.
- **Processing FPS:** includes decoding, tracking/mapping, completion of
  accepted work, applicable sequence-end correction and GPU completion.
  **Full-program FPS** also includes native initialisation and output writing.
  Offline compression, saved-map rendering and metric computation are separate.
  Every estimated-pose run contains frame calls longer than 33.3 ms; the average
  rates do not establish every-frame 30 Hz compliance or hard real time.
- **Memory:** each value averages per-run cgroup peaks across the complete
  program, then weights scenes equally. Peaks include applicable file-cache
  pages and are not private GPU allocations. GiB means 2³⁰ bytes. Frozen-Gaussian
  paging is disabled in these estimated-pose runs. This result does not
  establish a fixed total-map memory bound; in the separate original-system
  TUM office comparison, EGAS-SLAM uses more memory and storage than the
  capacity-augmented original control.

The completed campaign contains 265 fresh estimated-pose runs and 72 separately
admitted historical supplied-pose runs. The tables above use only the fresh
external comparison: 160 Replica and 60 TUM runs. Historical supplied-pose
Replica results isolate reconstruction under supplied camera poses and are not
tracking evidence. They are not pooled with these tables.

## Result provenance

The values are reproduced from the accepted final report
`production_20260913_attempt04` and the thesis's Chapter 8 tables
`tab:thor-replica-final`, `tab:external-tum` and `tab:resource-comparison`.
Trajectory and image-quality/rate means come from the report's
`study_statistics.json` macro summaries; memory comes from the retained
`resource_timing_scene_summaries.csv` cgroup summaries. Their SHA-256 identities
are:

```text
study_statistics.json
0a5dde0a63811c4dc96e5313c8efc77de62a8e5264c12af6f4d913f06cdb4058
resource_timing_scene_summaries.csv
d809790849b11bee67a06642756d3f09d8e32b45e90f6374b00f2f00bc59710c
```

Raw benchmark data, complete statistical reports and map checkpoints are
retained separately and are not shipped in this repository. The
[source provenance](../PROVENANCE.md), [source/configuration identities](../SOURCE_PROVENANCE.json)
and [final configuration templates](../configs/final/) document the software
release's relationship to the native benchmark and its separately built
`saved_overlap_v2` evaluator. A current build has its own identity; the combined
release is not asserted to be the historical binary for both stages.
