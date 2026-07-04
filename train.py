"""Fine-tune a small code model with LoRA using an explicit PyTorch loop.

The loop is deliberately hand-written (forward -> loss -> backward -> step)
rather than hidden inside Trainer, so the mechanics are visible and you can
speak to them. Run:  python train.py
"""
import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from config import cfg, pick_device, pick_dtype
from data import get_tokenized_dataset


def load_model_and_tokenizer(device):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(torch_dtype=pick_dtype(device))
    if cfg.use_qlora:
        # 4-bit base weights; only the LoRA adapters stay in higher precision.
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # shows you're training <1% of params
    return model, tok


def main():
    torch.manual_seed(cfg.seed)
    device = pick_device()
    print(f"device: {device} | qlora: {cfg.use_qlora}")

    model, tok = load_model_and_tokenizer(device)
    if not cfg.use_qlora:
        model.to(device)

    ds = get_tokenized_dataset(tok)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr
    )
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * cfg.warmup_ratio), total_steps
    )

    model.train()
    global_step = 0
    for epoch in range(cfg.epochs):
        running = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)                 # forward
            loss = out.loss / cfg.grad_accum     # scale for accumulation
            loss.backward()                      # autograd
            running += out.loss.item()

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()                       # update LoRA weights only
                sched.step()
                opt.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running / (i + 1)
                    lr_now = sched.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss {avg:.4f} lr {lr_now:.2e}")

        print(f"== epoch {epoch} done | mean loss {running / len(loader):.4f} ==")

    model.save_pretrained(cfg.adapter_dir)
    tok.save_pretrained(cfg.adapter_dir)
    print(f"saved adapter -> {cfg.adapter_dir}")


if __name__ == "__main__":
    main()
