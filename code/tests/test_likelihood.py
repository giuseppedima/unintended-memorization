import unittest
import torch

from memorization.evaluation.likelihood import compute_canary_log_likelihood
from memorization.exceptions import ProtocolViolationError


class _MockOutputs:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _MockTokenizer:
    vocab = {"prompt": 1, "is": 2, "maple": 10, "orbit": 11, "lantern": 12, "cobalt": 13}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[tok] for tok in text.strip().split()]


class _MockCausalLM:
    def __call__(self, input_ids: torch.Tensor) -> _MockOutputs:
        seq_len = input_ids.shape[1]
        logits = torch.zeros((1, seq_len, 20), dtype=torch.float32)
        for k in range(4):
            target_tok = int(input_ids[0, 2 + k].item())
            logits[0, 1 + k, target_tok] = 10.0
        return _MockOutputs(logits)


class TestLikelihood(unittest.TestCase):
    def test_compute_canary_log_likelihood_causal_shift_alignment(self) -> None:
        result = compute_canary_log_likelihood(
            model=_MockCausalLM(),
            tokenizer=_MockTokenizer(),
            prompt_text="prompt is",
            secret_str="maple orbit lantern cobalt",
            device=torch.device("cpu"),
        )
        self.assertEqual(result["w1_rank"], 1)
        for idx in range(1, 5):
            self.assertGreater(result[f"l_w{idx}"], -0.01)

    def test_trailing_space_raises_protocol_violation(self) -> None:
        with self.assertRaises(ProtocolViolationError):
            compute_canary_log_likelihood(
                model=_MockCausalLM(),
                tokenizer=_MockTokenizer(),
                prompt_text="prompt is ",
                secret_str="maple orbit lantern cobalt",
                device=torch.device("cpu"),
            )
