# Results

Held-out set: 8 requirement→code problems (`eval_problems.json`), each with
assert-based tests. Metric is **pass@1** — greedy decoding, one attempt, code
must pass every assert when run in an isolated subprocess.

> Fill this in after running `python evaluate.py --mode both`.

| Model                       | pass@1      |
|-----------------------------|-------------|
| Qwen2.5-Coder-0.5B (base)   | _ / 8 (_%)  |
| + LoRA fine-tune (this repo)| _ / 8 (_%)  |

**Setup:** LoRA r=16, alpha=32, on q_proj/v_proj · 3 epochs · 2k training
examples · ~_ min on a single _ GPU.

## What I learned
_(2–4 honest sentences. Good things to mention: how small the trainable-param
count was (<1%), where the model still failed (e.g. the harder algorithmic
problems), and one thing you'd try next — masking prompt tokens in the loss,
more target modules, or QLoRA on a bigger base model.)_
