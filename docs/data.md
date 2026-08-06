# Data

## JSONL contract

Each line is a JSON object with a `messages` array. The array must
contain at least two messages and the last message must have role
`assistant`.

```jsonl
{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
```

`prompt`/`completion` is an alias for the two-message form:

```jsonl
{"prompt":"Q","completion":"A"}
```

Optional scalar metadata fields (string, int, float, bool, null)
are preserved through normalization and round-tripped by the
JSONL reader.

## SWE-bench format

`hone prepare swe` normalizes rows from `SWE-bench/SWE-bench`
(or compatible) into a prompt that contains the repository, the
version, and the issue statement; the assistant message is the
unified diff patch.

The `swe` subcommand **refuses non-train splits** because the
evaluation patches are the answer.

## Prepare subcommands

```text
hone prepare file       local JSONL → train/valid splits
hone prepare code       HF competitive programming dataset → JSONL
hone prepare swe        HF SWE-bench train split → JSONL
hone prepare all        full HF corpus dump (sft or codeforces-text)
hone prepare evaluate   LiveCodeBench prompts for evaluation only
```

## Splitter guarantees

`Splitter(ratio, seed)`:

- `ratio` must be in `(0, 1)` (exclusive).
- At least 2 examples are required.
- The output is deterministic for a given seed.
- The validation set has at least `MIN_VALID = 1` example.
- Train and valid are disjoint and preserve all elements.

`split_file(source, train_path, valid_path, ratio, seed)` extends
the same guarantees at file level: it streams a JSONL file into
disjoint, byte-identical `train`/`valid` files with memory bounded
by the validation size, deterministic for a given input order and
seed.

## Reservoir sampling (code)

`hone prepare code` streams a HuggingFace dataset and keeps a
deterministic reservoir of `max_samples` rows. The reservoir is
seeded; two runs with the same seed produce identical output.
