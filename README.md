# requirements-to-code

Fine-tune a small, self-hostable open-source model (LoRA on
[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)) to turn a
natural-language **requirement** into a Python function — then **verify** each
generation by running it against assert-based tests in an isolated subprocess.
(Swap in the code-specialized [Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
via `config.py` for a stronger — but already near-ceiling — baseline.)

The point isn't a big model. It's the full loop, done honestly and locally:
select a base model → LoRA fine-tune with an explicit PyTorch training loop →
generate from a requirement → check the output passes its tests → report an
honest before/after `pass@1`.

## Why this exists

Most "AI engineering" work today is orchestration — RAG, agents, APIs. This
repo is the other half: actually training a model and measuring whether the
result is better. Everything runs **self-hosted** on a single consumer GPU (or a
free Colab/Kaggle T4), and the verify-against-requirements step mirrors how
you'd gate generated code in a real requirement-driven workflow.

## What's here

| File | Purpose |
|------|---------|
| `config.py` | Every knob in one place, incl. a `use_qlora` toggle |
| `data.py` | Loads an instruction→code dataset; fixes the prompt format |
| `train.py` | Hand-written PyTorch LoRA training loop (no `Trainer`) |
| `evaluate.py` | Generates, runs tests in a sandboxed subprocess, scores pass@1 |
| `eval_problems.json` | 60 held-out requirements + tests (easy → hard) |
| `results.md` | Before/after table to fill in |

## Quickstart

```bash
pip install -r requirements.txt

# 1. baseline: how good is the untuned model?
python evaluate.py --mode base

# 2. fine-tune (writes ./adapter)
python train.py

# 3. before/after comparison
python evaluate.py --mode both
```

On a free T4, the 0.5B + LoRA run is roughly 1–2 hours for 3 epochs over 2k
examples. Drop `train_size` in `config.py` for a faster smoke test.

### Windows

The code is cross-platform — the same three commands work in PowerShell or
`cmd`. A couple of platform notes:

- **CUDA vs CPU.** With an NVIDIA GPU the model trains/generates in `bfloat16`;
  on a CPU-only machine it automatically falls back to `float32` (bf16 CPU
  kernels are missing or very slow). CPU training of even a 0.5B model is slow —
  a GPU (local or a free Colab/Kaggle T4) is strongly recommended.
- **Install a real Python.** Use [python.org](https://www.python.org/downloads/)
  or `winget install Python.Python.3.12`, not the Microsoft Store stub. Then
  install the CUDA build of PyTorch from https://pytorch.org before
  `pip install -r requirements.txt`.
- **QLoRA (`use_qlora=True`).** `bitsandbytes` has official Windows wheels
  (0.43+) and is installed by `requirements.txt` on Windows and Linux alike.

## Design notes

- **Explicit training loop.** `train.py` does forward → loss → `backward()` →
  `opt.step()` by hand so the mechanics are visible, not buried in a framework.
- **LoRA.** The base model is frozen; only low-rank adapter matrices train —
  under 1% of parameters. `train.py` prints the exact count.
- **QLoRA path.** Set `use_qlora=True` to load the base model in 4-bit
  (needs `bitsandbytes`). Useful for larger bases on tight VRAM.
- **Safe verification.** Generated code is never `exec()`'d in-process. It's
  written to a temp file and run in a fresh interpreter with a timeout.

## Honest limitations

Greedy `pass@1` on 60 toy problems is a smoke test, not a benchmark. The prompt
tokens aren't masked from the loss (a known refinement). A 0.5B model will still
miss the harder algorithmic items. All of that is intentional scope — see
`results.md` for what I'd try next.

## License

MIT.
