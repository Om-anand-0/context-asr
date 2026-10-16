from context_asr.evaluate import normalize, term_recall, wer


def test_wer_ignores_case_and_punctuation():
    assert wer("Hello, World!", "hello world") == 0.0
    assert wer("Don't stop.", "dont stop") == 0.0


def test_wer_counts_errors():
    assert wer("a b c d", "a x c d") == 0.25


def test_normalize():
    assert normalize("  Jon Snow's   SWORD!? ") == "jon snows sword"


def test_term_recall_exact_spelling():
    ref = "Daenerys went to Westeros. Daenerys returned."
    hyp = "Denaris went to Westeros. Daenerys returned."
    # 3 term occurrences in the reference, 2 spelled right.
    assert term_recall(ref, hyp, ["Daenerys", "Westeros"]) == 2 / 3


def test_term_recall_ignores_terms_absent_from_reference():
    assert term_recall("Westeros", "Westeros", ["Westeros", "Kubernetes"]) == 1.0


def test_term_recall_whole_words_only():
    assert term_recall("Arya", "Aryan", ["Arya"]) == 0.0


def test_term_recall_no_terms_present():
    assert term_recall("nothing here", "nothing here", ["Westeros"]) == 1.0


def test_term_recall_ignores_punctuation_and_duplicate_terms():
    assert term_recall("Thanks to Monro, Inc. today", "thanks to monro inc today", ["Monro Inc", "Monro Inc"]) == 1.0
