# Publication Checklist

The local release may be prepared and validated while these items remain open. Do not make the repository public or enable release immutability until every blocking item is confirmed.

## Institutional and contributor approval

- [ ] The University of Manchester confirms whether it owns the modifications or requires an institutional notice.
- [ ] Publication of the project modifications is approved.
- [ ] Prof. Mikel Luján approves being named as supervisor contributor.
- [ ] Thesis/paper review and any embargo permit publication on the intended date.

## Third-party redistribution

- [ ] InfiniTAM non-commercial redistribution conditions are approved and its publication-notification obligation has been addressed.
- [ ] Inria/MPII Gaussian-splatting redistribution terms are satisfied.
- [ ] `THIRD_PARTY_NOTICES.md` and all files under `LICENSES/` receive a final human/legal review.
- [ ] The combined repository is described as source-available for non-commercial academic research where component restrictions apply—not as unconditionally AGPL-only.

## Technical release

- [ ] Fresh-clone build/check/regression/TUM/Replica acceptance passes.
- [ ] Algorithm equivalence to `thor-paper-20260819-r2-source` passes.
- [ ] Repository hygiene, documentation links, staged secret scan, size, modes, and Git status pass.
- [ ] Compact artifact manifest and SHA-256 verification pass in a second extraction directory.
- [ ] Private GitHub repository and authenticated remote are configured without exposing credentials.
- [ ] Draft release assets and metadata receive author review.
- [ ] Zenodo integration and ORCID linking are enabled; a stable thesis/paper identifier is added when available.
