# Security

## Supported versions

Only the latest released version of `hone` receives security
updates. Older versions are not patched.

## Reporting a vulnerability

Report security issues privately via
[GitHub Security Advisories](https://github.com/sachncs/hone/security/advisories/new).
Do not open a public issue for a suspected vulnerability; wait for
a fix to be released before disclosing.

## Scope

`hone` runs training scripts and reads local JSONL files. Treat
input JSONL as untrusted: malformed records should not crash the
normalizer. The data-validation contract is the first line of
defense against malformed input — see
[`hone.prepare.service._parse_chat`](hone/prepare/service.py) for
the canonical validator and [`tests/test_prepare.py`](tests/test_prepare.py)
for the regression tests that pin its behaviour.
