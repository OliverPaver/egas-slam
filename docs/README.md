# EGAS-SLAM documentation

Start with the installation guide for your hardware, or follow the architecture
from RGB-D input through local mapping and loop correction.

| Guide | Contents |
| --- | --- |
| [Desktop setup](desktop_setup.md) | Linux x86-64 dependencies, data preparation, first run and troubleshooting |
| [Thor reproduction](../REPRODUCIBILITY.md) | Pinned environment, final configurations, build and evaluation |
| [Architecture](ARCHITECTURE.md) | Runtime flow, ownership, lifecycle, paging, loop correction and source map |
| [Configuration](CONFIGURATION.md) | Final presets, build profiles and runtime controls |
| [Results](results.md) | Final Thor comparisons, metric definitions and evaluation boundaries |
| [Demonstration](../DEMO_GUIDE.md) | Video, optional capture, browser viewer and export |
| [Thor porting](THOR_PORTING.md) | Native CUDA/Torch compatibility and build rationale |
| [Measurement and validation](BENCHMARK_HARDENING.md) | Timing, metric support, completion and resource measurements |
| [Provenance](../PROVENANCE.md) | Source lineage and the evaluation-only overlap correction |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Component attribution and licence terms |

Historical desktop and visualisation checks are in
[development records](development/README.md). The results guide reports the
retained thesis measurements; raw benchmark payloads and dataset files are
maintained separately.
