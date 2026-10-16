"""Metrics: WER for overall accuracy, term recall for the names we care about."""

import re

import jiwer


def normalize(text: str) -> str:
    text = re.sub(r"['’]", "", text.lower())  # "don't" -> "dont"
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize(reference), normalize(hypothesis))


def _count(term: str, text: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", text))


def term_hits(reference: str, hypothesis: str, terms: list[str]) -> tuple[int, int]:
    """(occurrences spelled right, occurrences in the reference) over all key terms.

    Both sides are normalized like WER (case, punctuation), so "Monro, Inc."
    matches "Monro Inc"; matching is whole words only.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    hit = expected = 0
    for term in {normalize(t) for t in terms} - {""}:
        want = _count(term, ref)
        expected += want
        hit += min(want, _count(term, hyp))
    return hit, expected


def term_recall(reference: str, hypothesis: str, terms: list[str]) -> float:
    """Fraction of key-term occurrences in the reference that the hypothesis spells right.

    Terms absent from the reference are ignored; returns 1.0 if none appear.
    """
    hit, expected = term_hits(reference, hypothesis, terms)
    return hit / expected if expected else 1.0
