"""Find the things a speaker is talking about (proper nouns, titles)."""

import re
from functools import lru_cache

# Lowercase words allowed *inside* a capitalized run: "Game of Thrones".
CONNECTORS = {"of", "the", "and"}

# Capitalized only because they start a sentence (or are "I").
STOPWORDS = {
    "I", "I'm", "I've", "I'd", "I'll", "A", "An", "The", "So", "And", "But", "Or", "Then",
    "What", "When", "Where", "Why", "How", "Who", "Which", "This", "That", "These", "Those",
    "It", "It's", "He", "She", "We", "They", "You", "My", "Our", "Your", "His", "Her", "Their",
    "Yes", "No", "Yeah", "Okay", "OK", "Oh", "Well", "Now", "Just", "Also", "Actually",
    "If", "In", "On", "At", "For", "To", "With", "Is", "Are", "Was", "Were", "Do", "Did",
    "Let's", "There", "Here", "Maybe", "Because", "Like", "Not", "All", "Some", "Hi", "Hello",
    # Common spoken sentence openers.
    "Honestly", "Basically", "Anyway", "Meanwhile", "Right", "Sure", "Still", "Even", "Really",
    "Last", "Next", "First", "Up", "After", "Before", "Once", "Every", "Today", "Yesterday",
    "Tomorrow", "Thanks", "Please", "Great", "Good", "Wow", "Um", "Uh", "Hmm", "Look", "See",
}

# Split after . ! ? but not after an initial like "R." in "George R. R. Martin".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+")
_TOKEN = re.compile(r"\b[A-Z]\.(?=\s|$)|\w[\w'’-]*|[^\w\s]")


def extract_entities(text: str, use_spacy: bool = False) -> list[str]:
    """Return candidate entities in order of first appearance, deduplicated."""
    found = _spacy_entities(text) if use_spacy else _heuristic_entities(text)
    seen, result = set(), []
    for entity in found:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            result.append(entity)
    return result


def _is_name_word(token: str) -> bool:
    return token[:1].isupper() and re.split(r"['’]", token)[0] != "I"  # not I, I'm, I've


def _name_follows(tokens: list[str], i: int) -> bool:
    """Is tokens[i:] some connectors followed by a name word? ("of the Dragon")"""
    while i < len(tokens) and tokens[i] in CONNECTORS:
        i += 1
    return i < len(tokens) and _is_name_word(tokens[i])


def _heuristic_entities(text: str) -> list[str]:
    # ponytail: capitalization heuristic; false positives on sentence-start words
    # not in STOPWORDS. Use use_spacy=True when that matters.
    entities = []

    def flush(run):
        if run:
            run[-1] = re.sub(r"['’]s?$", "", run[-1])  # "Tyrion's wine" -> "Tyrion"
            entities.append(" ".join(run))

    for sentence in _SENTENCE_SPLIT.split(text):
        tokens = _TOKEN.findall(sentence)
        run: list[str] = []
        for i, token in enumerate(tokens):
            if _is_name_word(token) and not (i == 0 and token in STOPWORDS):
                run.append(token)
            elif run and token in CONNECTORS and _name_follows(tokens, i):
                run.append(token)
            else:
                flush(run)
                run = []
        flush(run)
    return entities


@lru_cache(maxsize=1)
def _spacy_model():
    import spacy  # optional extra: pip install context-asr[spacy]

    return spacy.load("en_core_web_sm")


_SPACY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP", "FAC"}


def _spacy_entities(text: str) -> list[str]:
    return [e.text for e in _spacy_model()(text).ents if e.label_ in _SPACY_LABELS]
