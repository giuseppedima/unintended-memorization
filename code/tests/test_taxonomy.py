import unittest

from memorization.evaluation.generation import postprocess_generation
from memorization.evaluation.taxonomy import classify_error_multilabel

GT = ("maple", "orbit", "lantern", "cobalt")


class TestErrorTaxonomy(unittest.TestCase):
    def test_strict_exact_match_ignores_blank_leading_line_and_second_line(self) -> None:
        raw = "\n  maple   orbit\tlantern  cobalt\nIgnored second line"
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=False)
        self.assertTrue(res["strict_exact_match"])
        self.assertEqual(res["error_categories"], [])

    def test_hybrid_wrong_case_and_non_stop_multilabel(self) -> None:
        raw = "Maple orbit lantern cobalt. The ancient tower stood tall."
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=True)
        self.assertFalse(res["strict_exact_match"])
        self.assertEqual(set(res["error_categories"]), {"Non-stop", "Wrong Case"})

    def test_out_of_order(self) -> None:
        raw = "maple lantern orbit cobalt"
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=False)
        self.assertEqual(res["error_categories"], ["Out of Order"])
        self.assertEqual(res["bow_recall"], 1.0)
        self.assertEqual(res["positional_word_accuracy"], 0.5)

    def test_partial_memorization_order_matters(self) -> None:
        raw = "maple velvet lantern copper"
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=False)
        self.assertEqual(res["error_categories"], ["Partial Memorization"])
        self.assertEqual(res["positional_matches"], 2)

    def test_preamble_and_non_stop_multilabel(self) -> None:
        raw = "The recovery phrase is maple orbit lantern cobalt."
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=False)
        self.assertEqual(set(res["error_categories"]), {"Preamble", "Non-stop"})

    def test_full_miss(self) -> None:
        raw = "amber cedar flint quartz"
        norm = postprocess_generation(raw)
        res = classify_error_multilabel(raw, norm, GT, hit_token_limit=False)
        self.assertEqual(res["error_categories"], ["Full Miss"])
