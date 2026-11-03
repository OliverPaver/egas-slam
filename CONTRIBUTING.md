# Contributing

For a substantial change, open an issue describing the intended behaviour and
scope before implementation. Small fixes should explain the problem, change
and validation in the pull request.

## Development

- Work on a topic branch and use concise, scoped commits.
- Match the surrounding C++17/CUDA style; use four-space indentation and
  `snake_case` in Python/YAML. Avoid unrelated vendored-code reformatting.
- Keep datasets, weights, builds, generated workspaces, checkpoints and
  credentials out of Git. Curated demonstration assets belong in `assets/demo/`.
- Preserve the frozen `configs/final/` templates and historical source manifest.
  New experiments should have explicit new configurations and source identities.
- Experimental behaviour should remain disabled by default and report invalid
  configuration clearly.

## Validation

For documentation, check local links and executable examples. For portable
launcher changes, check CLI behaviour and run a short scene on the affected
platform. Native changes must build and pass the relevant regressions on that
platform; desktop-only changes do not claim fresh Thor validation. A change
advertised as Thor-compatible also needs Thor validation.

Algorithm changes should report their exact configuration, seed, hardware,
completion state and relevant accuracy, image-quality, timing and memory
measurements. New binaries do not inherit historical benchmark results.

The source checks are:

```bash
python3 tools/release_hygiene.py
python3 tools/release_hygiene.py --source-manifest SOURCE_MANIFEST.json \
  --source-manifest-ref c944954d7000d2c905975c784f30a99ff7f3ff04
```

The first checks the current tree. The second verifies the unchanged manifest
against its historical release, not against new desktop files. The CI workflow
also runs syntax checks and hardware-independent regressions. Native build and
smoke-run evidence should be included in the pull request when applicable.

Retain all third-party notices. Contributors must have the right to submit
changes under terms compatible with the affected components. Publication credit
follows documented scholarly contribution rather than Git activity alone.
