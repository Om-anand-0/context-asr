"""Decide which topics are still live and turn them into a Whisper prompt."""

from dataclasses import dataclass
from itertools import zip_longest


@dataclass
class Topic:
    terms: list[str]
    score: float = 1.0


class ContextMemory:
    def __init__(self, decay=0.7, min_score=0.05, max_tokens=200, tail_words=25):
        self.decay = decay
        self.min_score = min_score
        self.max_chars = max_tokens * 4  # rough estimate: 4 chars per token
        self.tail_words = tail_words
        self.topics: dict[str, Topic] = {}
        self.words: list[str] = []

    def _find(self, name):
        return next((k for k in self.topics if k.lower() == name.lower()), None)

    def add_topic(self, name: str, terms: list[str]):
        key = self._find(name)
        if key:
            self.topics[key].score += 1
        else:
            self.topics[name] = Topic(list(terms))

    def mention(self, name: str) -> bool:
        """+1 to a known topic. Returns False if the topic isn't tracked."""
        key = self._find(name)
        if key:
            self.topics[key].score += 1
        return key is not None

    def all_terms(self) -> list[str]:
        """Every glossary spelling currently live. Topic names are left out: they
        are the entity as transcribed, possibly misheard."""
        return [t for topic in self.topics.values() for t in topic.terms]

    def topic_for_term(self, term: str):
        """The live topic whose glossary already contains `term`, if any."""
        for name, topic in self.topics.items():
            if term.lower() in (t.lower() for t in topic.terms):
                return name
        return None

    def step(self):
        """Call once per chunk: decay every topic, forget the faded ones."""
        for name in list(self.topics):
            self.topics[name].score *= self.decay
            if self.topics[name].score < self.min_score:
                del self.topics[name]

    def add_text(self, text: str):
        self.words.extend(text.split())

    def prompt(self, glossary: bool = True) -> str:
        tail = " ".join(self.words[-self.tail_words:])
        if not glossary or not self.topics:
            return tail

        # Round-robin across topics (strongest first) so one big topic
        # can't crowd out the others.
        ranked = sorted(self.topics.values(), key=lambda t: -t.score)
        chosen, seen = [], set()
        for row in zip_longest(*(t.terms for t in ranked)):
            for term in row:
                if term is None or term.lower() in seen:
                    continue
                candidate = _format(chosen + [term], tail)
                if len(candidate) > self.max_chars:
                    return _format(chosen, tail)
                chosen.append(term)
                seen.add(term.lower())
        return _format(chosen, tail)


def _format(terms, tail):
    if not terms:
        return tail
    return f"Glossary: {', '.join(terms)}. {tail}".strip()
