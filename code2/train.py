"""Stage 2: Fine-tune with LoRA using Controlled Shuffling."""

import json
import random
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import config


class CorpusDataset(Dataset):
    """Serve pre-tokenized 96-token sequences with full Causal LM loss (no masking)."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.records[idx]["input_ids"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


class NoDuplicateBatchSampler(Sampler):
    """Controlled Shuffling: prevent duplicate copies of the same canary in any mini-batch."""

    def __init__(self, records: list[dict], batch_size: int, seed: int = config.SEED):
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.num_batches = len(records) // batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def build_epoch_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + 1000 * epoch)

        canary_groups = defaultdict(list)
        bg_indices = []
        for idx, rec in enumerate(self.records):
            if rec["record_type"] == "canary":
                canary_groups[rec["canary_id"]].append(idx)
            else:
                bg_indices.append(idx)

        sorted_cids = sorted(canary_groups, key=lambda cid: len(canary_groups[cid]), reverse=True)

        for _ in range(1000):
            batches = [[] for _ in range(self.num_batches)]
            used_in_batch = [set() for _ in range(self.num_batches)]
            failed = False

            for cid in sorted_cids:
                indices = list(canary_groups[cid])
                rng.shuffle(indices)
                eligible = [
                    b for b in range(self.num_batches)
                    if len(batches[b]) < self.batch_size and cid not in used_in_batch[b]
                ]
                if len(eligible) < len(indices):
                    failed = True
                    break

                rng.shuffle(eligible)
                eligible.sort(key=lambda b: len(batches[b]))
                for rec_idx, b in zip(indices, eligible[: len(indices)]):
                    batches[b].append(rec_idx)
                    used_in_batch[b].add(cid)

            if failed:
                continue

            bg_pool = list(bg_indices)
            rng.shuffle(bg_pool)
            bg_iter = iter(bg_pool)
            for b in range(self.num_batches):
                while len(batches[b]) < self.batch_size:
                    batches[b].append(next(bg_iter))
                rng.shuffle(batches[b])

            rng.shuffle(batches)
            return batches

        raise RuntimeError("NoDuplicateBatchSampler could not build valid mini-batches.")

    def __iter__(self):
        yield from self.build_epoch_batches(self.epoch)

    def __len__(self) -> int:
        return self.num_batches


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, get_scheduler

    config.set_seed()
    config.ensure_dirs()

    with (config.CORPUS_DIR / "corpus_bundle.json").open("r", encoding="utf-8") as fp:
        records = json.load(fp)["records"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {config.MODEL_NAME} on {device} | lr={config.LEARNING_RATE:.1e} | bs={config.BATCH_SIZE}")

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=config.get_torch_dtype(),
    ).to(device)

    config.set_seed()
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    dataset = CorpusDataset(records)
    sampler = NoDuplicateBatchSampler(records, batch_size=config.BATCH_SIZE, seed=config.SEED)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    total_steps = (len(sampler) * config.NUM_EPOCHS) // config.GRAD_ACCUM_STEPS
    scheduler = get_scheduler(
        name=config.LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    model.config.use_cache = False
    model.train()

    loss_history = []
    global_step = 0

    for epoch in range(config.NUM_EPOCHS):
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_sampler=sampler)
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / config.GRAD_ACCUM_STEPS).backward()

            raw_loss = float(loss.item())
            epoch_losses.append(raw_loss)

            if step % config.GRAD_ACCUM_STEPS == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lr = float(scheduler.get_last_lr()[0])
                loss_history.append({"global_step": global_step, "epoch": epoch + 1, "loss": raw_loss, "lr": lr})
                pbar.set_postfix(loss=f"{raw_loss:.4f}", lr=f"{lr:.1e}")

        print(f"Epoch {epoch + 1} mean loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

    model.save_pretrained(config.ADAPTER_DIR)
    pd.DataFrame(loss_history).to_csv(config.RESULTS_DIR / "training_loss.csv", index=False)
    print(f"Saved LoRA adapter to: {config.ADAPTER_DIR}")


if __name__ == "__main__":
    main()
