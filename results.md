# Results

Held-out set: 24 requirement→code problems (`eval_problems.json`), each with
assert-based tests. Metric is **pass@1** — greedy decoding, one attempt, code
must pass every assert when run in an isolated subprocess.

| Model                       | pass@1         |
|-----------------------------|----------------|
| Qwen2.5-0.5B (base)         | 13 / 24 (54%)  |
| + LoRA fine-tune (this repo)| 20 / 24 (83%)  |

**Setup:** LoRA r=16, alpha=32, on q_proj/v_proj · 3 epochs · 2k training
examples · ~_ min on a single T4 GPU. Trainable params: 1,081,344 of
495,114,112 (**0.22%**).

## What I learned

The fine-tune lifted pass@1 by **+29 points (54% → 83%)** while training only
**0.22%** of the weights. The most striking part is *how* it helped: every one
of the base model's failures that was a `SyntaxError` — five "`[` was never
closed" truncations plus a `NameError` — disappeared. The plain (non-code) base
couldn't reliably emit complete, valid Python; LoRA taught it the
requirement→code format, so it now produces well-formed functions.

What remains is a cleaner class of error. All four surviving failures
(`is_palindrome`, `word_count`, `roman_to_int`, `second_largest`) are
`AssertionError`s — syntactically valid code with a wrong algorithm — which is
the expected ceiling for a 0.5B model. `second_largest` fails in both base and
fine-tuned, so it's genuinely hard at this scale, not a regression.

**What I'd try next:** mask the prompt tokens from the loss (train only on the
completion), add `k_proj`/`o_proj` to the LoRA target modules for more capacity,
or run QLoRA on a larger base to push at the remaining logic errors.
