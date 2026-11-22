"""Stage 2 & 3: Pre-training baseline, LoRA fine-tuning, and post-training evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from memorization.config import ExperimentConfig, get_torch_dtype, set_global_seed
from memorization.evaluation.runner import compute_paired_deltas, evaluate_all_canaries
from memorization.exceptions import ProtocolViolationError
from memorization.training.dataset import PreTokenizedCorpusDataset
from memorization.training.sampler import ControlledNoDuplicateBatchSampler

logger = logging.getLogger(__name__)


def load_corpus_bundle(corpus_path: Path) -> dict[str, Any]:
    """Load the verified `corpus_bundle.json` produced by Stage 1."""
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus bundle not found at {corpus_path}. Run `--stage corpus` first."
        )
    with corpus_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_lora_model(cfg: ExperimentConfig, device: torch.device):
    """Load base Qwen2.5 model and attach deterministic LoRA adapters (Attention + MLP)."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = get_torch_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    base_model.config.use_cache = True

    # Fix LoRA matrix A initialization seed
    set_global_seed(cfg.seed)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train_lora_adapter(
    model,
    records: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run Continual Pre-Training (full 96-token CLM loss, Controlled Shuffling, constant LR)."""
    from transformers import get_scheduler

    dataset = PreTokenizedCorpusDataset(records)
    batch_sampler = ControlledNoDuplicateBatchSampler(
        records=records,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = (len(batch_sampler) * cfg.num_epochs) // cfg.grad_accum_steps
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    model.config.use_cache = False
    model.train()

    loss_history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(cfg.num_epochs):
        batch_sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_sampler=batch_sampler)
        epoch_losses: list[float] = []
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
        for step_in_epoch, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()

            raw_loss = float(outputs.loss.item())
            epoch_losses.append(raw_loss)

            if step_in_epoch % cfg.grad_accum_steps == 0 or step_in_epoch == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = float(scheduler.get_last_lr()[0])
                loss_history.append(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": raw_loss,
                        "lr": current_lr,
                    }
                )
                pbar.set_postfix(loss=f"{raw_loss:.4f}", lr=f"{current_lr:.1e}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info("Epoch %d mean training loss: %.4f", epoch + 1, mean_loss)

    return loss_history


def save_experiment_outputs(
    pre_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
    loss_history: list[dict[str, Any]],
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist training loss log, all 96 evaluation records, and 48 paired deltas."""
    all_rows = pre_rows + post_rows
    if len(all_rows) != 96:
        raise ProtocolViolationError(f"Expected 96 total evaluations, got {len(all_rows)}.")

    df_all = pd.DataFrame(all_rows)
    df_paired = compute_paired_deltas(df_all)

    pd.DataFrame(loss_history).to_csv(cfg.results_dir / "training_loss.csv", index=False)
    with (cfg.results_dir / "eval_all_96.json").open("w", encoding="utf-8") as fp:
        json.dump(all_rows, fp, indent=2)
    df_all.to_csv(cfg.results_dir / "eval_all_96.csv", index=False)
    df_paired.to_csv(cfg.results_dir / "eval_paired_48.csv", index=False)

    logger.info("Saved 96 full evaluations to: %s", cfg.results_dir / "eval_all_96.csv")
    logger.info("Saved 48 paired post-vs-pre rows to: %s", cfg.results_dir / "eval_paired_48.csv")
    return df_all, df_paired


def run_training_and_evaluation(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Phase A (pre-eval), Phase B (LoRA training), and Phase C (post-eval)."""
    set_global_seed(cfg.seed)
    cfg.ensure_dirs()

    bundle = load_corpus_bundle(cfg.corpus_dir / "corpus_bundle.json")
    canaries = bundle["canaries"]
    records = bundle["records"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Starting experiment | model=%s | device=%s | dtype=%s | epochs=%d | bs=%d | lr=%.1e (%s)",
        cfg.model_name,
        device,
        cfg.dtype,
        cfg.num_epochs,
        cfg.batch_size,
        cfg.learning_rate,
        cfg.lr_scheduler_type,
    )

    model, tokenizer = build_lora_model(cfg, device)

    logger.info("[Phase A] Evaluating PRE-TRAINING baseline (48 generations + 48 likelihoods)...")
    with model.disable_adapter():
        pre_rows = evaluate_all_canaries(
            model=model,
            tokenizer=tokenizer,
            canaries=canaries,
            phase="pre",
            max_new_tokens=cfg.max_new_tokens,
            device=device,
        )

    logger.info("[Phase B] Starting LoRA Fine-Tuning (%d epochs)...", cfg.num_epochs)
    loss_history = train_lora_adapter(model, records, cfg, device)
    model.save_pretrained(cfg.adapter_dir)
    logger.info("Saved LoRA adapter weights to: %s", cfg.adapter_dir)

    logger.info("[Phase C] Evaluating POST-TRAINING model (48 generations + 48 likelihoods)...")
    model.config.use_cache = True
    model.eval()
    post_rows = evaluate_all_canaries(
        model=model,
        tokenizer=tokenizer,
        canaries=canaries,
        phase="post",
        max_new_tokens=cfg.max_new_tokens,
        device=device,
    )

    return save_experiment_outputs(pre_rows, post_rows, loss_history, cfg)
