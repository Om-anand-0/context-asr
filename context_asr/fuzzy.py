"""Near-miss name matching: "Brand Stark" is probably "Bran Stark" if Bran is in the glossary."""

import re

import jellyfish

from .entities import extract_entities


def similar(a: str, b: str) -> bool:
    """Same name, misheard? One-ish typo, or the same sound with a few more edits."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if min(len(a), len(b)) < 5:  # "Ross"/"Rose": too short to tell apart
        return False
    dist = jellyfish.damerau_levenshtein_distance(a, b)
    if dist <= max(1, len(b) // 8):  # "brand stark" ~ "bran stark"
        return True
    # "denaris" ~ "daenerys": same metaphone code (one slip allowed in long codes,
    # "TRKRN" ~ "TRKRYN"), and edits within half the length.
    # ponytail: single Metaphone, English-centric; swap in Double Metaphone if needed.
    code_a, code_b = jellyfish.metaphone(a), jellyfish.metaphone(b)
    code_slack = 1 if len(code_b) >= 5 else 0
    return (
        jellyfish.levenshtein_distance(code_a, code_b) <= code_slack
        and dist <= len(b) // 2
    )


def _prefix(term: str, size: int) -> str | None:
    words = term.split()
    return " ".join(words[:size]) if len(words) >= size else None


def closest(name: str, terms: list[str]) -> str | None:
    """The glossary spelling `name` was probably meant to be, or None.

    Compares against the leading words of each term, so "Denaris" can match the
    first name in "Daenerys Targaryen". Only leading words: the tail of a title
    ("House of the Dragon") is usually an ordinary word, and "Drogon" must not
    become "Dragon". None if `name` is already spelled like a glossary entry.
    """
    size = len(name.split())
    candidates = [p for t in terms if (p := _prefix(t, size))]
    if any(w.lower() == name.lower() for w in candidates):
        return None
    matches = [w for w in candidates if similar(name, w)]
    if not matches:
        return None
    return min(matches, key=lambda w: jellyfish.damerau_levenshtein_distance(name.lower(), w.lower()))


def correct_text(text: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    """Rewrite near-miss names in `text` to their glossary spelling."""
    changes = {}
    for entity in extract_entities(text):
        fix = closest(entity, terms)
        if fix:
            changes[entity] = fix
            text = re.sub(rf"(?<!\w){re.escape(entity)}(?!\w)", fix, text)
    return text, changes
