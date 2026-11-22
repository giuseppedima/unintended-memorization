"""PyTorch Dataset serving exact pre-tokenized 96-token sequences."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class PreTokenizedCorpusDataset(Dataset):
    """Serve pre-tokenized 96-token integer sequences without re-tokenization or loss masking."""

    def __init__(self, records: Sequence[dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.records[idx]["input_ids"], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }
