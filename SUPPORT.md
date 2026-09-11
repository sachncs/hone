# Getting support

`hone` is a small library; the maintainer runs the same `setup.sh` +
`train-soup.sh` workflow that the README documents. If something
does not work, the right channel depends on the kind of question:

| Kind of question | Where to ask |
|---|---|
| Bug or unexpected behaviour | Open a GitHub issue with the `bug` label (use `.github/ISSUE_TEMPLATE/bug.yml`) |
| Feature idea / improvement | Open a GitHub issue with the `enhancement` label (use `.github/ISSUE_TEMPLATE/feature_request.yml`) |
| "How do I ...?" / "Why does ...?" | Open a GitHub issue with the `question` label (use `.github/ISSUE_TEMPLATE/question.yml`) |
| Security disclosure | Use [GitHub Security Advisories](https://github.com/sachncs/hone/security/advisories/new) — do **not** file a public issue |

For general questions that do not fit a template, file the issue
with the `question` label and the maintainer will route it
appropriately.

## Before opening an issue

1. Read the relevant section of [`README.md`](README.md) and the
   prepare / Soup docs.
2. Search [open and closed issues](https://github.com/sachncs/hone/issues?q=is%3Aissue)
   to see if the topic has already been addressed.
3. Run `uv run pytest` against your checkout — a failing test
   is the fastest way to surface a real bug.