import unittest
import pandas as pd

from memorization.analysis.summaries import (
    build_error_table,
    build_summary_table,
    evaluate_hypotheses,
)
from memorization.evaluation.runner import compute_paired_deltas


class TestSummaries(unittest.TestCase):
    def test_summary_and_hypotheses_pipeline(self) -> None:
        rows_all = []
        for phase in ("pre", "post"):
            for prompt_type in ("exact_prefix", "paraphrased"):
                for group, exp in ((0, 0), (1, 2), (2, 10), (3, 40)):
                    base_ll = -50.0
                    gain = (
                        15.0
                        if group == 0
                        else (15.0 + exp * (0.8 if prompt_type == "exact_prefix" else 0.4))
                    )
                    ll = base_ll if phase == "pre" else base_ll + gain
                    is_non_stop = bool(
                        phase == "post" and prompt_type == "exact_prefix" and exp == 40
                    )
                    rows_all.append(
                        {
                            "phase": phase,
                            "canary_id": group,
                            "group": group,
                            "cumulative_exposures": exp,
                            "prompt_type": prompt_type,
                            "total_ll": ll,
                            "l_w1": ll / 4,
                            "l_w2": ll / 4,
                            "l_w3": ll / 4,
                            "l_w4": ll / 4,
                            "w1_rank": 1000 if phase == "pre" else max(1, 50 - exp),
                            "strict_exact_match": False,
                            "relaxed_prefix_match": is_non_stop,
                            "positional_word_accuracy": 0.0,
                            "bow_recall": 0.0,
                            "output_limit_truncation": False,
                            "err_non_stop": is_non_stop,
                            "err_preamble": False,
                            "err_partial_memorization": False,
                            "err_out_of_order": False,
                            "err_wrong_case": False,
                            "err_full_miss": not is_non_stop,
                        }
                    )

        df_all = pd.DataFrame(rows_all)
        df_paired = compute_paired_deltas(df_all)
        summary = build_summary_table(df_paired)
        errors = build_error_table(df_paired)
        hyp = evaluate_hypotheses(df_all, df_paired)

        self.assertEqual(len(summary), 8)
        self.assertEqual(len(errors), 8)
        self.assertTrue(hyp["Supported"].all())
