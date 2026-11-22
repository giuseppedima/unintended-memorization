"""Deterministic greedy decoding and protocol-compliant post-processing."""

from __future__ import annotations

from typing import Any

import torch

from memorization.exceptions import ProtocolViolationError


def postprocess_generation(raw_continuation: str) -> str:
    """Protocol post-processing:
    1. Extract the first nonempty generated line.
    2. Collapse internal/leading/trailing whitespace into single spaces.
    3. Preserve case and punctuation verbatim.
    """
    non_empty_lines = [
        line for line in raw_continuation.splitlines() if line.strip()
    ]
    if not non_empty_lines:
        return ""
    return " ".join(non_empty_lines[0].split())


@torch.no_grad()
def greedy_extract_canary(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run deterministic greedy decoding (`do_sample=False`) in raw plain-text mode."""
    if prompt_text.endswith(" "):
        raise ProtocolViolationError(
            f"Trailing space detected in prompt (BPE trap): {prompt_text!r}"
        )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, input_ids.shape[1] :]
    raw_continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    first_line_norm = postprocess_generation(raw_continuation)

    return {
        "num_generated_tokens": int(gen_ids.shape[0]),
        "hit_token_limit": bool(gen_ids.shape[0] >= max_new_tokens),
        "raw_continuation": raw_continuation,
        "first_line_norm": first_line_norm,
    }
