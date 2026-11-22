"""Controlled Shuffling batch sampler preventing duplicate canary copies per mini-batch."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterator, Sequence

from torch.utils.data import Sampler

from memorization.exceptions import ProtocolViolationError


class ControlledNoDuplicateBatchSampler(Sampler[list[int]]):
    """Ensure no two copies of the same canary ever share a mini-batch.

    Each epoch uses a distinct deterministic seed (`seed + 1000 * epoch`).
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        batch_size: int,
        seed: int = 42,
        max_attempts: int = 1000,
    ):
        self.records = list(records)
        self.batch_size = batch_size
        self.seed = seed
        self.max_attempts = max_attempts
        self.epoch = 0

        if len(self.records) % self.batch_size != 0:
            raise ProtocolViolationError(
                f"batch_size ({batch_size}) must divide dataset size ({len(self.records)})."
            )
        self.num_batches = len(self.records) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def build_epoch_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + 1000 * epoch)
        canary_groups, bg_indices = self._partition_indices()
        sorted_cids = sorted(
            canary_groups.keys(),
            key=lambda cid: len(canary_groups[cid]),
            reverse=True,
        )

        for _ in range(self.max_attempts):
            batches = self._try_build_batches(canary_groups, sorted_cids, bg_indices, rng)
            if batches is not None:
                self._verify_no_duplicates(batches)
                return batches

        raise ProtocolViolationError(
            f"ControlledNoDuplicateBatchSampler failed after {self.max_attempts} attempts."
        )

    def _partition_indices(self) -> tuple[dict[int, list[int]], list[int]]:
        canary_groups: dict[int, list[int]] = defaultdict(list)
        bg_indices: list[int] = []
        for idx, rec in enumerate(self.records):
            if rec["record_type"] == "canary":
                canary_groups[int(rec["canary_id"])].append(idx)
            else:
                bg_indices.append(idx)
        return canary_groups, bg_indices

    def _try_build_batches(
        self,
        canary_groups: dict[int, list[int]],
        sorted_cids: list[int],
        bg_indices: list[int],
        rng: random.Random,
    ) -> list[list[int]] | None:
        batches: list[list[int]] = [[] for _ in range(self.num_batches)]
        batch_cids: list[set[int]] = [set() for _ in range(self.num_batches)]

        for cid in sorted_cids:
            indices = list(canary_groups[cid])
            rng.shuffle(indices)
            eligible = [
                b
                for b in range(self.num_batches)
                if len(batches[b]) < self.batch_size and cid not in batch_cids[b]
            ]
            if len(eligible) < len(indices):
                return None
            rng.shuffle(eligible)
            eligible.sort(key=lambda b: len(batches[b]))
            for rec_idx, batch_idx in zip(indices, eligible[: len(indices)]):
                batches[batch_idx].append(rec_idx)
                batch_cids[batch_idx].add(cid)

        bg_shuffled = list(bg_indices)
        rng.shuffle(bg_shuffled)
        bg_iter = iter(bg_shuffled)
        for batch_idx in range(self.num_batches):
            while len(batches[batch_idx]) < self.batch_size:
                batches[batch_idx].append(next(bg_iter))
            rng.shuffle(batches[batch_idx])

        rng.shuffle(batches)
        return batches

    def _verify_no_duplicates(self, batches: Sequence[Sequence[int]]) -> None:
        for batch in batches:
            if len(batch) != self.batch_size:
                raise ProtocolViolationError(
                    f"Batch size mismatch: expected {self.batch_size}, got {len(batch)}."
                )
            seen_cids: set[int] = set()
            for rec_idx in batch:
                cid = self.records[rec_idx]["canary_id"]
                if cid is not None:
                    if cid in seen_cids:
                        raise ProtocolViolationError(
                            f"Duplicate canary_id {cid} detected in same mini-batch."
                        )
                    seen_cids.add(cid)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.build_epoch_batches(self.epoch)

    def __len__(self) -> int:
        return self.num_batches
