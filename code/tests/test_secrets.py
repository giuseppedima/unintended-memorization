import unittest

from memorization.data.dictionary import CANDIDATE_DICTIONARY_WORDS
from memorization.data.secrets import generate_constrained_secrets, verify_secret_constraints
from memorization.exceptions import ProtocolViolationError


class TestConstrainedSecrets(unittest.TestCase):
    def test_generate_constrained_secrets_satisfies_all_constraints(self) -> None:
        vocab = list(CANDIDATE_DICTIONARY_WORDS[:32])
        for seed in (42, 7, 123):
            with self.subTest(seed=seed):
                secrets = generate_constrained_secrets(
                    vocab, num_canaries=24, secret_len=4, seed=seed
                )
                verify_secret_constraints(secrets, vocab)
                self.assertEqual(len(secrets), 24)
                self.assertEqual(len({s[0] for s in secrets}), 24)

    def test_verify_secret_constraints_catches_bigram_collision(self) -> None:
        vocab = list(CANDIDATE_DICTIONARY_WORDS[:32])
        secrets = generate_constrained_secrets(vocab, seed=42)
        corrupted = list(secrets)
        corrupted[1] = (corrupted[0][0], corrupted[0][1], corrupted[1][2], corrupted[1][3])
        with self.assertRaises(ProtocolViolationError):
            verify_secret_constraints(corrupted, vocab)
