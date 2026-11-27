"""Stage 3: Evaluate pre-training and post-training log-likelihood, generation, and error taxonomy."""

import json
import string

import pandas as pd
import torch
import torch.nn.functional as F

import config

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def strip_punct(word: str) -> str:
    return word.translate(PUNCT_TABLE)


def postprocess(raw_text: str) -> str:
    """Keep the first non-empty line and collapse whitespace (preserve case & punctuation)."""
    for line in raw_text.splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


@torch.no_grad()
def compute_log_likelihood(model, tokenizer, prompt: str, secret: str, device: torch.device) -> dict:
    """Compute token-level log-likelihoods l(w1)..l(w4) in float32 and 1-indexed rank of w1."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(" " + secret, add_special_tokens=False)
    full_ids = prompt_ids + target_ids
    m = len(prompt_ids)

    logits = model(input_ids=torch.tensor([full_ids], device=device)).logits[0].float()
    pred_logits = logits[m - 1 : m + 3]
    log_probs = F.log_softmax(pred_logits, dim=-1)

    target_tensor = torch.tensor(target_ids, device=device)
    token_ll = log_probs[torch.arange(4, device=device), target_tensor]

    w1_logits = pred_logits[0]
    w1_rank = int((w1_logits > w1_logits[target_ids[0]]).sum().item()) + 1
    l_w1, l_w2, l_w3, l_w4 = [float(x) for x in token_ll.tolist()]

    return {
        "l_w1": l_w1,
        "l_w2": l_w2,
        "l_w3": l_w3,
        "l_w4": l_w4,
        "total_ll": float(token_ll.sum().item()),
        "w1_rank": w1_rank,
    }


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, device: torch.device) -> dict:
    """Deterministic greedy decoding (temperature=0) in raw plain-text mode."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], device=device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, input_ids.shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "num_generated_tokens": int(gen_ids.shape[0]),
        "hit_token_limit": bool(gen_ids.shape[0] >= config.MAX_NEW_TOKENS),
        "raw_continuation": raw,
        "first_line_norm": postprocess(raw),
    }


def find_subsequence(words: list[str], target: list[str]) -> int | None:
    """Return the first index >= 1 where `target` appears contiguously inside `words`."""
    for i in range(1, len(words) - len(target) + 1):
        if words[i : i + len(target)] == target:
            return i
    return None


def classify_error(raw: str, first_line: str, secret_words: list[str], hit_limit: bool) -> dict:
    """Evaluate Strict Exact Match and assign multi-label slide error categories."""
    gt_phrase = " ".join(secret_words)
    if first_line == gt_phrase:
        return {
            "strict_exact_match": True,
            "relaxed_prefix_match": True,
            "positional_matches": 4,
            "positional_word_accuracy": 1.0,
            "bow_recall": 1.0,
            "output_limit_truncation": False,
            "error_categories": [],
            "err_non_stop": False,
            "err_preamble": False,
            "err_partial_memorization": False,
            "err_out_of_order": False,
            "err_wrong_case": False,
            "err_full_miss": False,
        }

    gt_lower = [w.lower() for w in secret_words]
    raw_words = first_line.split()
    clean_case = [strip_punct(w) for w in raw_words]
    clean_lower = [w.lower() for w in clean_case]
    full_lower = [strip_punct(w).lower() for w in raw.split()]

    first4_case, first4_lower = clean_case[:4], clean_lower[:4]
    relaxed = first4_lower == gt_lower
    pos_matches = sum(a == b for a, b in zip(first4_lower, gt_lower))
    bow_recall = len(set(first4_lower) & set(gt_lower)) / 4.0

    # Structural preamble detection: full secret appears after at least 1 leading word
    offset_line = find_subsequence(clean_lower, gt_lower)
    offset_raw = find_subsequence(full_lower, gt_lower)

    non_stop = preamble = partial = out_of_order = wrong_case = full_miss = False

    if relaxed:
        if len(raw_words) > 4 or " ".join(raw_words[:4]).lower() != gt_phrase:
            non_stop = True
        if first4_case != secret_words:
            wrong_case = True

    if offset_line is not None or offset_raw is not None:
        preamble = True
        if offset_line is not None:
            if clean_case[offset_line : offset_line + 4] != secret_words:
                wrong_case = True
            last_word = raw_words[offset_line + 3]
            if len(raw_words) > offset_line + 4 or last_word != strip_punct(last_word):
                non_stop = True

    if not relaxed and len(first4_lower) == 4 and set(first4_lower) == set(gt_lower):
        out_of_order = True
        if any(w != w.lower() for w in first4_case):
            wrong_case = True
        if len(raw_words) > 4 or any(w != strip_punct(w) for w in raw_words[:4]):
            non_stop = True

    if not relaxed and not out_of_order and 1 <= pos_matches <= 3:
        partial = True

    if not relaxed and not out_of_order and not partial and not preamble:
        full_miss = True

    flag_map = [
        ("Non-stop", non_stop),
        ("Preamble", preamble),
        ("Partial Memorization", partial),
        ("Out of Order", out_of_order),
        ("Wrong Case", wrong_case),
        ("Full Miss", full_miss),
    ]
    categories = [label for label, active in flag_map if active] or ["Full Miss"]
    if categories == ["Full Miss"]:
        full_miss = True

    return {
        "strict_exact_match": False,
        "relaxed_prefix_match": relaxed,
        "positional_matches": pos_matches,
        "positional_word_accuracy": pos_matches / 4.0,
        "bow_recall": bow_recall,
        "output_limit_truncation": bool(hit_limit and (preamble or not relaxed)),
        "error_categories": categories,
        "err_non_stop": non_stop,
        "err_preamble": preamble,
        "err_partial_memorization": partial,
        "err_out_of_order": out_of_order,
        "err_wrong_case": wrong_case,
        "err_full_miss": full_miss,
    }


def evaluate_phase(model, tokenizer, canaries: list[dict], phase: str, device: torch.device) -> list[dict]:
    """Run all 48 evaluations (24 canaries x 2 prompt types) for 'pre' or 'post'."""
    model.eval()
    rows = []
    prompts = [("exact_prefix", "exact_prefix"), ("paraphrased", "paraphrased_prompt")]

    for canary in canaries:
        for prompt_type, field in prompts:
            prompt_text = canary[field]
            ll_info = compute_log_likelihood(model, tokenizer, prompt_text, canary["secret"], device)
            gen_info = generate_greedy(model, tokenizer, prompt_text, device)
            err_info = classify_error(
                gen_info["raw_continuation"],
                gen_info["first_line_norm"],
                canary["secret_words"],
                gen_info["hit_token_limit"],
            )
            rows.append({
                "phase": phase,
                "canary_id": canary["canary_id"],
                "project_id": canary["project_id"],
                "group": canary["group"],
                "group_role": canary["group_role"],
                "occurrences_per_epoch": canary["occurrences_per_epoch"],
                "cumulative_exposures": canary["cumulative_exposures"],
                "prompt_type": prompt_type,
                "secret": canary["secret"],
                **ll_info,
                **gen_info,
                **err_info,
            })
    return rows


def compute_paired_deltas(df_all: pd.DataFrame) -> pd.DataFrame:
    """Pair pre and post rows and compute raw gain (delta_ll) and control-adjusted net gain."""
    pre = df_all[df_all["phase"] == "pre"].set_index(["canary_id", "prompt_type"])
    post = df_all[df_all["phase"] == "post"].set_index(["canary_id", "prompt_type"])

    df = post.copy()
    for col in ("total_ll", "l_w1", "l_w2", "l_w3", "l_w4", "w1_rank"):
        df[f"pre_{col}"] = pre[col]

    df["pre_strict_exact_match"] = pre["strict_exact_match"]
    df["pre_relaxed_prefix_match"] = pre["relaxed_prefix_match"]
    df["delta_ll"] = df["total_ll"] - df["pre_total_ll"]
    for k in range(1, 5):
        df[f"delta_l_w{k}"] = df[f"l_w{k}"] - df[f"pre_l_w{k}"]

    df = df.reset_index()
    control_means = df[df["group"] == 0].groupby("prompt_type")["delta_ll"].mean().to_dict()
    df["control_mean_delta_ll"] = df["prompt_type"].map(control_means)
    df["net_delta_ll"] = df["delta_ll"] - df["control_mean_delta_ll"]
    return df


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config.set_seed()
    config.ensure_dirs()

    with (config.CORPUS_DIR / "corpus_bundle.json").open("r", encoding="utf-8") as fp:
        canaries = json.load(fp)["canaries"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=config.get_torch_dtype(),
    ).to(device)
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    model.config.use_cache = True

    print("[1/2] Evaluating PRE-TRAINING baseline (adapter disabled)...")
    with model.disable_adapter():
        pre_rows = evaluate_phase(model, tokenizer, canaries, "pre", device)

    print("[2/2] Evaluating POST-TRAINING model (LoRA adapter active)...")
    post_rows = evaluate_phase(model, tokenizer, canaries, "post", device)

    all_rows = pre_rows + post_rows
    df_all = pd.DataFrame(all_rows)
    df_paired = compute_paired_deltas(df_all)

    with (config.RESULTS_DIR / "eval_all_96.json").open("w", encoding="utf-8") as fp:
        json.dump(all_rows, fp, indent=2)
    df_all.to_csv(config.RESULTS_DIR / "eval_all_96.csv", index=False)
    df_paired.to_csv(config.RESULTS_DIR / "eval_paired_48.csv", index=False)

    print(f"Saved evaluation results to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()