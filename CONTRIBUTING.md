# Contributing

Thank you for helping improve this research software. Open an issue before a substantial change so scope, evaluation cost, and licensing can be agreed first.

## Development rules

- Work from a topic branch and keep commits small, imperative, and scoped, for example `slam: reject incomplete loop jobs`.
- Use four-space indentation and nearby Allman-style C++/CUDA conventions. Python and YAML names use `snake_case`.
- Do not reformat vendored code or mix unrelated cleanup with behavior changes.
- Do not commit datasets, weights, builds, generated configs, workspaces, checkpoints, full renders, telemetry, or credentials.
- Keep experimental behavior disabled by default and validate configuration errors at startup.

Every source change must build on Thor, pass `bash scripts/thor_container.sh regressions`, and receive a representative scene smoke test. Algorithm changes also require exact configuration, seed, hardware/power state, complete-frame status, accuracy/quality, timing, memory, and failure results. A functional change requires an updated source manifest and new validation; it cannot inherit measurements from a previous build.

Pull requests must describe the motivation and behavior, link relevant issues, list commands/results, identify licensing effects, and include focused logs or renders for quality/performance changes. Confirm `python3 tools/release_hygiene.py --source-manifest SOURCE_MANIFEST.json` passes before submission.

By contributing, you confirm you have the right to submit the work under terms compatible with every affected component. Do not copy code with unclear provenance. Publication-related author or contributor credit is determined by documented scholarly contribution, not by Git activity alone.
