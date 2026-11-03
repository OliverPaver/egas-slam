# Desktop installation and first run

This guide covers the native Linux x86-64 **RTX 3070 / SM 86** profile. The
tested workstation has 8 GB GPU memory and 32 GB system memory. The desktop
launcher defaults to calibrated 600 × 340 Replica input; TUM uses 640 × 480.
For Jetson AGX Thor, use [the Thor reproduction guide](../REPRODUCIBILITY.md).

## 1. Install the toolchain

The tested software combination is Ubuntu 24.04-compatible Linux (Linux Mint
22.1), GCC 13.3, CMake 3.28, CUDA Toolkit 12.8 and Python 3.12. Install an NVIDIA
driver compatible with CUDA 12.8 and the **development toolkit**, including
`nvcc`, using [NVIDIA's CUDA 12.8 installation instructions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)
and [download archive](https://developer.nvidia.com/cuda-12-8-0-download-archive).
The CUDA runtime in a PyTorch wheel does not provide the project’s compiler.

```bash
sudo apt update
sudo apt install git build-essential cmake ninja-build pkg-config \
  python3-venv libopencv-dev libeigen3-dev libprotobuf-dev protobuf-compiler
nvidia-smi
/usr/local/cuda/bin/nvcc --version
```

OpenCV's native development package is separate from Python's `opencv-python`.
Protobuf remains a build dependency because the bundled dependency builder
compiles `tensorboard_logger`, even though logging is disabled in the trainer.
The documented headless build needs neither Pangolin nor a display server.

## 2. Clone and install Python dependencies

```bash
git clone https://github.com/OliverPaver/gps-slam-submaps-thor.git EGAS-SLAM
cd EGAS-SLAM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-desktop.txt
source scripts/desktop_env.sh
```

This Torch/torchvision pairing and CUDA index follow the
[official PyTorch 2.10 installation instructions](https://pytorch.org/get-started/previous-versions/#v2100).
Torch also supplies the LibTorch headers and libraries used by the C++ build;
a separate LibTorch download is unnecessary. The pinned desktop requirements
cover dataset preparation, launch, trajectory summaries and image scoring.

Keep this environment active for subsequent commands. `./egas` selects
`python3` from the active environment. To choose an interpreter explicitly:

```bash
export EGAS_PYTHON="$PWD/.venv/bin/python"
source scripts/desktop_env.sh
"$EGAS_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

The expected GPU capability is `(8, 6)`. This build script checks that capability;
other GPU architectures require a separately validated build profile.

## 3. Build

```bash
./egas build
```

This builds the bundled dependencies and `build_desktop/slam_trainer`, then runs
the native filesystem regression. Four compiler workers are used by default;
`EGAS_BUILD_JOBS=2 ./egas build` reduces peak build memory. The script finds
`nvcc` on `PATH`, falling back to `/usr/local/cuda/bin/nvcc`; set
`CUDACXX=/path/to/cuda/bin/nvcc` for another toolkit location. Use CUDA 12.8 with
the documented Torch wheel.

The build retains the `medium_128k` SDF capacity, fast math OFF, TensorBoard OFF,
legacy UI OFF and the host C++ runtime link-order correction. The correction
prevents a demonstrated filesystem cleanup crash with this Torch wheel and is
checked by `desktop_filesystem_regression`. The environment also sets
`NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## 4. Choose local storage

The default workspace is `$HOME/egas_workspace`, containing `data/` and `runs/`.
To use a different dedicated directory, set it before preparing or running:

```bash
export EGAS_WORKSPACE_ROOT="$HOME/egas_workspace"
```

`--workspace-root /local/path` overrides this variable for one `prepare`,
`train` or `eval` command. `--input-dir /local/prepared/scene` selects an input
outside its default data directory. Use the same workspace selection when
evaluating a run. The desktop launcher derives its paths from this workspace;
the Thor container's independent `GPS_SLAM_DATA_ROOT` and `GPS_SLAM_RUN_ROOT`
variables do not override it.

Source, datasets, build products and outputs must be on local storage. The
desktop tools reject paths under `/media`, `/mnt` and `/run/media`, including
resolved symbolic links. Use a dedicated workspace: native training clears its
configured map directory. The launcher reserves a new run name and keeps its
receipts outside that clearable directory.

## 5. First run: TUM RGB-D

This command downloads and prepares the Freiburg 1 desk sequence from the
[TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).
Review the dataset's terms before downloading. Preparation preserves the PNG
images and depth units and associates frames by timestamp.

```bash
"$EGAS_PYTHON" scripts/tum_preprocess.py \
  --data-root "$EGAS_WORKSPACE_ROOT/data/tum_rgbd" \
  --sequences rgbd_dataset_freiburg1_desk --download --copy

# Short startup check; skips the large final checkpoint.
./egas train --dataset tum --scene fr1_desk --frames 60 --no-save

# Complete sequence with saved map.
./egas train --dataset tum --scene fr1_desk
```

Preparation writes `data/tum_rgbd/processed/rgbd_dataset_freiburg1_desk`.
Run it once: rerunning preprocessing replaces that processed scene's images.
For the other supported scenes, use `--paper-only` instead of `--sequences ...`
to prepare all three, then select `fr2_xyz` or `fr3_office` in `./egas train`.
If raw archives have already been extracted into `data/tum_rgbd/raw/`, omit
`--download`.

## 6. Replica preparation and saved-map rendering

Obtain the rendered Replica RGB-D sequences distributed with
[NICE-SLAM](https://github.com/cvg/nice-slam#replica-1), following its
[dataset download instructions](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh)
and the applicable dataset terms. The preparation script requires this layout:

```text
Replica/
  office0/
    traj.txt
    results/
      frame000000.jpg
      depth000000.png
      ...
```

The final profiles use 2,000 frames per scene. Original Replica mesh assets
alone do not supply these rendered sequences. Point `--raw-root` at the local
directory containing `office0`, `room0`, and the other scene directories:

```bash
./egas prepare --scene office0 --raw-root /local/path/to/Replica
./egas train --dataset replica --scene office0 --frames 60 --no-save
./egas train --dataset replica --scene office0
```

Preparation creates `data/replica_prepared/office0` at 1200 × 680 and
`data/replica_half/office0` at 600 × 340. It refuses to overwrite the half copy.
The half-resolution calibration is `[300, 300, 299.75, 169.75]`, with reader
downscale factor `1`: both TSDF and Gaussian cameras receive the same physical
image size. Colour uses area interpolation and JPEG re-encoding; depth uses
nearest-neighbour interpolation without changing depth units.

`train --resolution full` selects the prepared full-resolution copy. A complete
room0 run has been tested at this resolution on the RTX 3070; larger scenes and
future map growth may have different memory requirements.

The launcher prints the new run directory. Substitute it below after a saved
Replica run completes:

```bash
./egas eval --run "$EGAS_WORKSPACE_ROOT/runs/RUN_NAME" --views 20
./egas summary "$EGAS_WORKSPACE_ROOT/runs/RUN_NAME"
```

Evaluation reads the saved hybrid map and renders uniformly spaced observed
views into a new directory under the map. `--views 0` renders all recorded
views. This sample differs from the historical 500-view Thor evaluation.
Runs made with `--no-save` cannot be rendered later. The convenience evaluation
launcher supports Replica; TUM trajectory outputs can be summarised directly.

## Outputs and optional image metrics

Every training run contains `record/config.yaml`, `record/run.json`,
`record/native.log`, sampled `record/telemetry.csv` and the `map/` outputs.
`--name NAME` gives a run a memorable unique name; an existing name is refused.
`--dry-run` validates input and writes its exact configuration without starting
SLAM. A dry run reserves its name, so choose a new name for execution.

Training and saved-map rendering need no pretrained network weights. Optional
VGG-LPIPS scoring downloads `vgg16-397923af.pth` and `vgg.pth` on first use to
`${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints`. Keep that cache on local
storage. Select the actual evaluation directory printed by the renderer:

```bash
"$EGAS_PYTHON" scripts/metric_general.py \
  --gt_path "$EGAS_WORKSPACE_ROOT/data/replica_half/office0" \
  --render_path "$EGAS_WORKSPACE_ROOT/runs/RUN_NAME/map/EVAL_NAME/render"
```

Use the full-resolution reference folder for full-resolution runs. The scorer
reports PSNR, SSIM and VGG-LPIPS separately from processing time; depth masking
is opt-in. See [the script guide](../scripts/README.md) for metric contracts.

Desktop measurements include different hardware, image preparation and Torch
versions from the final Thor studies. Native processing FPS, command wall time
and sampled process/device memory cover different scopes. An average throughput
does not demonstrate every-frame deadline compliance. Optional capture and the
local browser viewer are documented in [the demo guide](../DEMO_GUIDE.md).
