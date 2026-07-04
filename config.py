"""Central config. Change things here, not scattered through the scripts."""
from dataclasses import dataclass

import torch


def pick_device() -> str:
    """CUDA if present, else CPU. Works the same on Linux and Windows."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str):
    """Device-aware dtype. bfloat16 is the right choice on a CUDA GPU, but on a
    CPU-only box (common on Windows) bf16 kernels are missing/painfully slow, so
    fall back to float32 there."""
    return torch.bfloat16 if device == "cuda" else torch.float32


@dataclass
class Config:
    # --- model ---
    # Plain (non-code) base: it starts weaker on code, which leaves real
    # headroom for the LoRA fine-tune to show a before/after gap. Swap back to
    # "Qwen/Qwen2.5-Coder-0.5B" for the already-code-specialized baseline.
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_len: int = 512

    # --- data ---
    dataset: str = "iamtarun/python_code_instructions_18k_alpaca"
    train_size: int = 2000          # small on purpose; bump if you have time
    seed: int = 42

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # q_proj/v_proj is the standard minimal set; add k_proj,o_proj for more capacity
    target_modules: tuple = ("q_proj", "v_proj")

    # --- QLoRA toggle ---
    # False -> plain LoRA in bf16 (fine on a free T4 for a 0.5B model).
    # True  -> load base model in 4-bit (needs bitsandbytes). Use this for
    #          bigger models / tighter VRAM. Ties to the "self-host on limited
    #          hardware" angle in the job posting.
    use_qlora: bool = False

    # --- training ---
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 2             # effective batch = batch_size * grad_accum
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    log_every: int = 20

    # --- paths ---
    adapter_dir: str = "adapter"

    # --- eval ---
    eval_file: str = "eval_problems.json"
    max_new_tokens: int = 256
    subprocess_timeout: int = 10    # seconds per generated program


cfg = Config()
