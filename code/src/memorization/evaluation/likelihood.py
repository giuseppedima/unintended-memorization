"""Exact token-level log-likelihood and w1 rank computation under teacher forcing."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from memorization.exceptions import ProtocolViolationError


@torch.no_grad()
def compute_canary_log_likelihood(
    model,
    tokenizer,
    prompt_text: str,
    secret_str: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compute l(w1), l(w2), l(w3), l(w4), total log-likelihood, and 1-indexed rank of w1.

    Uses float32 logit upcasting before `log_softmax` and explicit causal shift alignment:
    logits at positions [m-1, m, m+1, m+2] predict targets at [m, m+1, m+2, m+3].
    """
    if prompt_text.endswith(" "):
        raise ProtocolViolationError(
            f"Trailing space detected in prompt (BPE trap): {prompt_text!r}"
        )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(" " + secret_str, add_special_tokens=False)
    full_ids = tokenizer.encode(f"{prompt_text} {secret_str}", add_special_tokens=False)

    prompt_len = len(prompt_ids)
    if len(target_ids) != 4:
        raise ProtocolViolationError(f"Expected 4 secret tokens, got {len(target_ids)}.")
    if full_ids != prompt_ids + target_ids:
        raise ProtocolViolationError(
            "Concatenation BPE mismatch between prompt_ids + target_ids and full_ids."
        )

    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_tensor)
    logits_fp32 = outputs.logits[0].float()

    pred_logits = logits_fp32[prompt_len - 1 : prompt_len + 3, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)

    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    token_ll = log_probs[torch.arange(4, device=device), target_tensor]

    w1_logits = pred_logits[0]
    w1_score = w1_logits[target_ids[0]]
    w1_rank = int((w1_logits > w1_score).sum().item()) + 1

    l_w1, l_w2, l_w3, l_w4 = (float(val) for val in token_ll.tolist())
    return {
        "l_w1": l_w1,
        "l_w2": l_w2,
        "l_w3": l_w3,
        "l_w4": l_w4,
        "total_ll": float(token_ll.sum().item()),
        "w1_rank": w1_rank,
    }
