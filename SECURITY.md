# Security

## Supported versions

Only the latest released version of `hone` receives security
updates. Older versions are not patched.

## Reporting a vulnerability

Open a GitHub issue with the `security` label or email the
maintainers. Do not disclose the vulnerability publicly until a
fix is released.

## Scope

`hone` runs training scripts and reads local JSONL files. Treat
input JSONL as untrusted: malformed records should not crash the
normalizer. The data-validation contract (see `docs/data.md`) is
the first line of defense against malformed input.
