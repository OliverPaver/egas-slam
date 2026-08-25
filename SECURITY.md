# Security Policy

## Supported version

Security fixes are considered for the latest tagged paper release. This is research software, not a safety-certified system, and it must not be used as the sole control or localization component in a safety-critical deployment.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose a host, dataset, credential, or network service. Email `oliver_paver@yahoo.com` with the affected revision, reproduction conditions, impact, and any proposed mitigation. Do not include secrets or sensitive datasets. You should receive acknowledgement within seven days.

Container execution grants the process GPU access and read/write access to explicitly mounted paths. Docker daemon access is effectively root-equivalent on the host. Review scripts before execution, mount datasets and weights read-only, use a dedicated run directory, and never point `workspace_dir` at source or shared data because run startup may clear it.
