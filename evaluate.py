"""Generate code from each held-out requirement, run it against its tests in an
isolated subprocess, and report pass@1. Compares base vs. fine-tuned so you get
an honest before/after number for the README.

Usage:
  python evaluate.py --mode base       # base model only
  python evaluate.py --mode finetuned  # requires ./adapter from train.py
  python evaluate.py --mode both       # default; prints a comparison table
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import cfg, pick_device, pick_dtype
from data import build_prompt


def load(mode):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=pick_dtype(device)
    )
    if mode == "finetuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter_dir)
    return model.to(device).eval(), tok, device


def generate(model, tok, device, requirement):
    prompt = build_prompt(requirement)
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def extract_code(text):
    """Model may wrap output in ```python ... ```; strip that if present."""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (fenced.group(1) if fenced else text).strip()


def run_in_subprocess(code, tests):
    """Write code+tests to a temp file and execute in a fresh interpreter with a
    timeout. Never exec() model output in this process."""
    program = f"{code}\n\n{tests}\n"
    # encoding="utf-8" matters on Windows, where the default is cp1252 and
    # generated code may contain non-ASCII characters.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=cfg.subprocess_timeout,
            # Force the child to read its source as UTF-8 regardless of the
            # Windows system code page.
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        return r.returncode == 0, (r.stderr.strip().splitlines()[-1:] or [""])[0]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        # On Windows a lingering handle (e.g. AV scanner) can briefly lock the
        # file; don't let cleanup failures crash the run.
        try:
            os.unlink(path)
        except OSError:
            pass


def evaluate(mode, problems):
    model, tok, device = load(mode)
    passed = 0
    rows = []
    for p in problems:
        raw = generate(model, tok, device, p["requirement"])
        code = extract_code(raw)
        ok, err = run_in_subprocess(code, p["tests"])
        passed += ok
        rows.append((p["id"], ok, err))
        print(f"[{mode}] {p['id']:16s} {'PASS' if ok else 'FAIL'}"
              + (f"  ({err})" if not ok else ""))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return passed, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "finetuned", "both"], default="both")
    args = ap.parse_args()

    with open(cfg.eval_file, encoding="utf-8") as f:
        problems = json.load(f)
    n = len(problems)

    results = {}
    modes = ["base", "finetuned"] if args.mode == "both" else [args.mode]
    for m in modes:
        passed, _ = evaluate(m, problems)
        results[m] = passed
        print(f"\n{m}: {passed}/{n} pass@1 = {passed / n:.0%}\n")

    if args.mode == "both":
        print("| Model      | pass@1        |")
        print("|------------|---------------|")
        for m in ["base", "finetuned"]:
            print(f"| {m:10s} | {results[m]}/{n} ({results[m]/n:.0%}) |")


if __name__ == "__main__":
    main()
