# Configuration generators

Run `slam_trainer` with `configs/final/` for current presets; follow [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

`hybrid_validation.py` generates Replica reconstruction configurations. `tum_loop_validation.py` generates TUM tracking/loop configurations. Their complete interfaces retain compatibility profiles for explicitly configured custom experiments. Generator defaults are not the final multi-system benchmark recipe. Inspect generated YAML, build selection and paths before running.

Keep outputs outside the repository, and never use a dataset, source tree or retained results directory as a workspace. Exploratory sweep and archival-cleanup programs are not part of this final distribution.
