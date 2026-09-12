"""Metric correctness tests.

These exist because the metrics in this repository are the numbers that appear
in a published paper. A metric that silently computes something other than its
name is the single hardest error for a reader to detect, so each definition is
pinned to a hand-computable case.
"""

from __future__ import annotations

import math

import pytest

from kannada_rag.eval.metrics import (
    MetricConfig, chrf, meteor, rouge_l, rouge_n, score_pair,
)
from kannada_rag.text import (
    is_predominantly_kannada, kannada_ratio, normalise_for_index, tokenise_for_metrics,
)

KN_A = "ಹಂಪಿ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿತ್ತು"
KN_B = "ಹಂಪಿ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿ ಆಗಿತ್ತು"


class TestTokenisation:
    def test_kannada_tokens_survive(self):
        assert tokenise_for_metrics(KN_A) == [
            "ಹಂಪಿ", "ವಿಜಯನಗರ", "ಸಾಮ್ರಾಜ್ಯದ", "ರಾಜಧಾನಿಯಾಗಿತ್ತು",
        ]

    def test_numerals_are_kept_not_deleted(self):
        # Dates are load-bearing content in heritage answers. A tokeniser that
        # strips every non-Kannada character would delete "1336" and inflate
        # the score of an answer that omitted the date entirely.
        tokens = tokenise_for_metrics("ಹಂಪಿ 1336 ರಲ್ಲಿ")
        assert "1336" in tokens

    def test_numerals_can_be_excluded_for_ablation(self):
        assert "1336" not in tokenise_for_metrics("ಹಂಪಿ 1336", keep_numerals=False)

    def test_latin_is_casefolded(self):
        assert "hampi" in tokenise_for_metrics("Hampi ಹಂಪಿ")

    def test_nfc_normalisation_makes_equivalent_forms_identical(self):
        decomposed = "ನಿ"      # ನ + ಿ
        assert normalise_for_index(decomposed) == normalise_for_index("ನಿ")


class TestLanguageGuard:
    def test_ratio_excludes_whitespace(self):
        assert kannada_ratio("ಹಂಪಿ ಇತಿಹಾಸ") == 1.0

    def test_single_kannada_character_in_english_is_rejected(self):
        # The previous implementation returned True for any Kannada character
        # present, which let English queries through and produced English
        # answers from the downstream model.
        assert not is_predominantly_kannada("what is the history of ಹಂಪಿ")

    def test_kannada_query_is_accepted(self):
        assert is_predominantly_kannada(KN_A)

    def test_empty_query_is_rejected(self):
        assert not is_predominantly_kannada("")


class TestRouge:
    def test_identical_strings_score_one(self):
        assert rouge_n(KN_A, KN_A, 1)["f1"] == pytest.approx(1.0)
        assert rouge_l(KN_A, KN_A)["f1"] == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self):
        assert rouge_n("ಹಂಪಿ", "ಬೇಲೂರು", 1)["f1"] == 0.0

    def test_counts_are_clipped(self):
        # "ಕ ಕ ಕ" against "ಕ": one match, not three.
        result = rouge_n("ಕ ಕ ಕ", "ಕ", 1)
        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == pytest.approx(1.0)

    def test_rouge2_is_order_sensitive(self):
        forward = rouge_n("ಅ ಬ ಕ", "ಅ ಬ ಕ", 2)["f1"]
        reversed_ = rouge_n("ಕ ಬ ಅ", "ಅ ಬ ಕ", 2)["f1"]
        assert forward == pytest.approx(1.0)
        assert reversed_ == 0.0

    def test_rouge_l_rewards_subsequence_order(self):
        # ROUGE-1 is identical for both; ROUGE-L separates them.
        assert rouge_n("ಅ ಕ ಬ", "ಅ ಬ ಕ", 1)["f1"] == pytest.approx(1.0)
        assert rouge_l("ಅ ಕ ಬ", "ಅ ಬ ಕ")["f1"] < 1.0


class TestMeteor:
    def test_identical_strings_score_one(self):
        # A perfect match is one chunk, so the fragmentation penalty is
        # gamma * (1/n)^beta, which vanishes only when the whole string is a
        # single contiguous run -- which it is.
        result = meteor(KN_A, KN_A)
        assert result["chunks"] == 1
        assert result["meteor"] > 0.99

    def test_fragmentation_penalty_is_applied(self):
        # This is the property the earlier implementation lacked entirely:
        # a scrambled answer containing exactly the same words must score
        # strictly lower than the correctly ordered one.
        ordered = meteor("ಅ ಬ ಕ ಡ", "ಅ ಬ ಕ ಡ")
        scrambled = meteor("ಬ ಡ ಅ ಕ", "ಅ ಬ ಕ ಡ")

        assert ordered["precision"] == pytest.approx(scrambled["precision"])
        assert ordered["recall"] == pytest.approx(scrambled["recall"])
        assert scrambled["chunks"] > ordered["chunks"]
        assert scrambled["meteor"] < ordered["meteor"]

    def test_penalty_matches_published_formula(self):
        cfg = MetricConfig()
        result = meteor("ಬ ಡ ಅ ಕ", "ಅ ಬ ಕ ಡ", cfg)
        expected = cfg.meteor_gamma * (result["chunks"] / result["matches"]) ** cfg.meteor_beta
        assert result["penalty"] == pytest.approx(expected)
        assert result["meteor"] == pytest.approx(result["f_mean"] * (1 - expected))

    def test_recall_is_weighted_more_than_precision(self):
        # F_mean = 10PR/(R+9P): missing reference content should hurt more
        # than adding extra content.
        short_answer = meteor("ಅ", "ಅ ಬ ಕ ಡ")        # low recall
        verbose_answer = meteor("ಅ ಬ ಕ ಡ ಇ ಈ ಉ", "ಅ ಬ ಕ ಡ")  # low precision
        assert verbose_answer["f_mean"] > short_answer["f_mean"]

    def test_no_overlap_scores_zero(self):
        assert meteor("ಹಂಪಿ", "ಬೇಲೂರು")["meteor"] == 0.0


class TestChrf:
    def test_morphological_variants_score_above_word_overlap(self):
        # "ರಾಜಧಾನಿಯಾಗಿತ್ತು" vs "ರಾಜಧಾನಿ ಆಗಿತ್ತು": no shared word unigram for
        # the inflected form, but most character n-grams are shared. This is
        # precisely why chrF is reported for an agglutinative language.
        word_level = rouge_n(KN_A, KN_B, 1)["f1"]
        char_level = chrf(KN_A, KN_B)["chrf"]
        assert char_level > word_level

    def test_identical_strings_score_one(self):
        assert chrf(KN_A, KN_A)["chrf"] == pytest.approx(1.0, abs=1e-9)


class TestScorePair:
    def test_returns_every_reported_metric(self):
        scores = score_pair(KN_A, KN_B)
        for key in ("rouge1_f1", "rouge2_f1", "rougeL_f1", "meteor", "chrf"):
            assert key in scores
            assert not math.isnan(scores[key])

    def test_empty_hypothesis_does_not_crash(self):
        scores = score_pair("", KN_A)
        assert scores["rouge1_f1"] == 0.0
        assert scores["meteor"] == 0.0
