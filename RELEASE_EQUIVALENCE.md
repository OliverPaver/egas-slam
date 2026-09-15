# Source correspondence and release identity

The final implementation intentionally differs from `thor-paper-20260819-r2-source`; R2 equivalence is not the current acceptance criterion.

It combines the revised native source with the separately accepted saved-map overlap-hydration correction. The latter restores a serialized evaluation-camera membership list before Gaussian selection. Native fixes include CUDA synchronization/prefix handling, capacity profiles and diagnostics, estimated-pose isolation, complete saved-pose loading, precise pose serialization, PNG evaluation and normalized VGG-LPIPS.

Release adaptations cover portable input/output paths, explicit build presets, local manifest paths, a portable regression fixture, source checks and documentation. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records imported digests and identifies adapted files; it contains no benchmark outcomes.

```bash
python3 tools/release_hygiene.py --source-manifest SOURCE_MANIFEST.json
```

This verifies every shipped file, including `.tpp`/`.tcu`, metric scripts, runtime wrappers, configurations and vendored sources. It detects missing/added/changed contents. Hygiene also checks generated-output contamination, unsafe links, accidental secrets and broken documentation links. The manifest excludes its own entry to avoid a circular digest; Git carries its identity.

Source correspondence is not bitwise binary or asynchronous-runtime equivalence. Compilers, libraries, flags and the embedded Git revision affect executables. New measurements need their own binary/config/environment records. Release builds and short checks establish functionality; they add no observations to the completed thesis campaign.
