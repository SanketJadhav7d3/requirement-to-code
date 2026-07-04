"""Load an instruction->code dataset and format it for causal-LM fine-tuning.

The prompt format is fixed here and reused at eval time (evaluate.py imports
PROMPT_TEMPLATE) so training and inference never drift apart.
"""
from datasets import load_dataset
from config import cfg

PROMPT_TEMPLATE = "# Requirement:\n{req}\n\n# Implementation:\n"


def build_prompt(requirement: str) -> str:
    return PROMPT_TEMPLATE.format(req=requirement.strip())


def _pick_fields(example):
    """The alpaca-style set has 'instruction' + optional 'input' + 'output'.
    Fold any 'input' into the requirement so the prompt is self-contained."""
    instr = example.get("instruction", "").strip()
    extra = example.get("input", "").strip()
    req = instr if not extra else f"{instr}\n{extra}"
    return req, example.get("output", "").strip()


def get_tokenized_dataset(tokenizer):
    ds = load_dataset(cfg.dataset, split=f"train[:{cfg.train_size}]")

    def tokenize(example):
        req, code = _pick_fields(example)
        prompt = build_prompt(req)
        full = prompt + code + tokenizer.eos_token

        enc = tokenizer(
            full,
            truncation=True,
            max_length=cfg.max_len,
            padding="max_length",
        )
        # Train on the whole sequence. Masking the prompt tokens is a common
        # refinement (set those labels to -100); left simple here on purpose.
        enc["labels"] = [
            -100 if tok == tokenizer.pad_token_id else tok
            for tok in enc["input_ids"]
        ]
        return enc

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds
