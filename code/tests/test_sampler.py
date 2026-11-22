import unittest

from memorization.data.dictionary import GROUP_SPECS
from memorization.training.sampler import ControlledNoDuplicateBatchSampler


def _make_mock_corpus() -> list[dict]:
    records = []
    cid = 0
    for _, _, occ in GROUP_SPECS:
        for _ in range(6):
            for _ in range(occ):
                records.append({"record_type": "canary", "canary_id": cid})
            cid += 1
    for _ in range(100):
        records.append({"record_type": "background", "canary_id": None})
    return records


class TestControlledSampler(unittest.TestCase):
    def test_controlled_sampler_no_duplicates_and_full_coverage(self) -> None:
        records = _make_mock_corpus()
        for batch_size in (4, 8):
            with self.subTest(batch_size=batch_size):
                sampler = ControlledNoDuplicateBatchSampler(
                    records, batch_size=batch_size, seed=42
                )
                for epoch in range(2):
                    batches = sampler.build_epoch_batches(epoch)
                    self.assertEqual(len(batches), 256 // batch_size)
                    flat_indices = [idx for batch in batches for idx in batch]
                    self.assertEqual(sorted(flat_indices), list(range(256)))

                    for batch in batches:
                        cids = [
                            records[idx]["canary_id"]
                            for idx in batch
                            if records[idx]["canary_id"] is not None
                        ]
                        self.assertEqual(len(cids), len(set(cids)))
