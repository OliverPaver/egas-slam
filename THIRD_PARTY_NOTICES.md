# Third-Party Notices

This file is an engineering inventory, not legal advice. The repository combines works under different terms; the most restrictive applicable term may control a particular use. Complete licence texts are retained in `LICENSES/` and original source notices must not be removed.

## Core research components

| Component | Repository paths | Origin | Terms retained here |
|---|---|---|---|
| GPS-SLAM | project code inherited from upstream | [MisEty/GPS-SLAM](https://github.com/MisEty/GPS-SLAM) | AGPL-3.0, [LICENSE.txt](LICENSE.txt), [LICENSES/GPS-SLAM-AGPL-3.0.txt](LICENSES/GPS-SLAM-AGPL-3.0.txt) |
| InfiniTAM.3 and ORUtils/MiniSlamGraph | `InfiniTAM/`, `slam/TsdfFusion/` | [victorprad/InfiniTAM](https://github.com/victorprad/InfiniTAM) | Non-commercial InfiniTAM licence, [LICENSES/InfiniTAM.txt](LICENSES/InfiniTAM.txt) |
| gsplat rasterization | `gsplat/rasterizer/` except the Graphdeco files below | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0, [LICENSES/gsplat-apache-2.0.txt](LICENSES/gsplat-apache-2.0.txt) |
| Graphdeco Gaussian splatting | `gsplat/rasterizer/simple_knn.*`, `scripts/utils/` files carrying Inria notices | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Non-commercial research licence, [LICENSES/Gaussian-Splatting.txt](LICENSES/Gaussian-Splatting.txt) |
| lpips-pytorch | `scripts/lpipsPyTorch/` | [S-aiueo32/lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch) | BSD-2-Clause, [LICENSES/lpips-pytorch.txt](LICENSES/lpips-pytorch.txt) |
| PlenOctree SH utilities | `scripts/utils/sh_utils.py` | PlenOctree authors | BSD-2-Clause, [LICENSES/PlenOctree-sh-utils.txt](LICENSES/PlenOctree-sh-utils.txt) |
| COLMAP model I/O helper | `scripts/utils/read_write_model.py` | ETH Zurich and UNC Chapel Hill | BSD-3-Clause, [LICENSES/COLMAP-read-write-model.txt](LICENSES/COLMAP-read-write-model.txt) |

InfiniTAM requires its copyright notice to appear prominently wherever it is used. Publications based on it must identify the software and its licence as described in the complete licence text. InfiniTAM and Graphdeco terms prohibit commercial use without separate permission.

## Pinned build dependencies

| Component | Version/snapshot | Path | Licence |
|---|---:|---|---|
| cxxopts | 3.2.0 | `ThirdLibs/fetchcontent/cxxopts/` | MIT, [LICENSES/cxxopts.txt](LICENSES/cxxopts.txt) |
| GLM | 1.0.1 | `ThirdLibs/fetchcontent/glm/` | Happy Bunny/MIT alternatives, [LICENSES/glm.txt](LICENSES/glm.txt) |
| nanoflann | 1.5.5 | `ThirdLibs/fetchcontent/nanoflann/` | BSD-2-Clause, [LICENSES/nanoflann.txt](LICENSES/nanoflann.txt) |
| nlohmann/json | 3.11.3 | `ThirdLibs/fetchcontent/nlohmann_json/`, `json.hpp` | MIT, [LICENSES/nlohmann-json.txt](LICENSES/nlohmann-json.txt) |
| indicators | 2.3.0 | `ThirdLibs/indicators/` | MIT; bundled termcolor BSD-3-Clause, [LICENSES/indicators.txt](LICENSES/indicators.txt), [LICENSES/termcolor.txt](LICENSES/termcolor.txt) |
| tensorboard_logger | transferred pinned source | `ThirdLibs/tensorboard_logger/` | MIT, [LICENSES/tensorboard-logger.txt](LICENSES/tensorboard-logger.txt) |
| tinyply | 2.3.4 | `ThirdLibs/tinyply/` | Public domain dedication/perpetual fallback; BSD-2 option, [LICENSES/tinyply.txt](LICENSES/tinyply.txt) |
| yaml-cpp | 0.8.0 | `ThirdLibs/yaml-cpp/` | MIT, [LICENSES/yaml-cpp.txt](LICENSES/yaml-cpp.txt) |

CUDA, NVIDIA PyTorch containers, OpenCV, Eigen, Protobuf, Torch/Torchvision, SciPy, and other system/runtime packages are not redistributed by this repository. They remain subject to their suppliers’ terms. Model weights and datasets are also excluded.

## Redistribution gate

Before changing the repository from private to public, confirm University ownership/embargo requirements and the redistribution conditions for every core research component. A release must include this notice and the entire `LICENSES/` directory. Do not describe the combined work as commercially reusable or solely AGPL-3.0.
