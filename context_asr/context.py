"""Where related vocabulary comes from: Wikipedia, a JSON file, or both."""

import json
import logging
import re
from pathlib import Path
from typing import Protocol

import requests

from .entities import extract_entities

log = logging.getLogger(__name__)

# Bump the version when the ranking changes so stale term lists aren't reused.
DEFAULT_CACHE = Path.home() / ".cache" / "context-asr" / "wikipedia-v2.json"


class ContextSource(Protocol):
    def related_terms(self, entity: str) -> list[str]: ...


def _dedupe(terms):
    seen, result = set(), []
    for term in terms:
        if term and term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    return result


class FileSource:
    """Terms from a user JSON file (or dict): {"entity": ["term", ...]}."""

    def __init__(self, path_or_dict):
        data = path_or_dict
        if not isinstance(data, dict):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        self.terms = {key.lower(): list(value) for key, value in data.items()}

    def related_terms(self, entity: str) -> list[str]:
        return self.terms.get(entity.lower(), [])


class ChainSource:
    """Ask every source, merge the answers in order."""

    def __init__(self, *sources: ContextSource):
        self.sources = sources

    def related_terms(self, entity: str) -> list[str]:
        return _dedupe(t for s in self.sources for t in s.related_terms(entity))


def _mentions(term: str, text: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", text))


def rank_terms(title: str, text: str, links: list[str], max_terms: int = 40) -> list[str]:
    """Article title, names in the intro, then links by how often the article mentions them.

    `text` is the article plaintext; the intro is everything before the first
    "== Section ==" heading. Links never mentioned in the text (navboxes,
    "See also" clutter) are dropped.
    """
    title, *links = [re.sub(r"\s*\([^)]*\)$", "", t) for t in [title, *links]]  # "(film)"
    linked = {t.lower() for t in links}
    # A one-word intro "name" is often just a sentence-start word ("Critics ...");
    # keep it only if the article also links to it.
    intro_text = text.split("\n==", 1)[0]
    intro = [e for e in extract_entities(intro_text) if " " in e or e.lower() in linked]
    # Frequency is a decent stand-in for importance: "Tyrion Lannister" is all over
    # the Game of Thrones article, "Aidan Gillen" barely appears.
    counts = {t: _mentions(t, text) for t in links}
    ranked = sorted((t for t in links if counts[t]), key=lambda t: -counts[t])
    terms = [
        t for t in [title, *intro, *ranked]
        if not t.startswith("List of") and not t[:1].isdigit()  # years, "1925 Grand Prix"
    ]
    return _dedupe(terms)[:max_terms]


class WikipediaSource:
    """Best-matching Wikipedia article -> names from its intro and its most-mentioned links."""

    API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "context-asr/0.1 (https://github.com/Om-anand-0/context-asr)"}

    def __init__(self, cache_path=DEFAULT_CACHE, max_terms=40, timeout=5.0, session=None):
        self.cache_path = Path(cache_path)
        self.max_terms = max_terms
        self.timeout = timeout
        self.session = session or requests.Session()
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    def related_terms(self, entity: str) -> list[str]:
        key = entity.lower()
        if key not in self.cache:
            try:
                self.cache[key] = self._fetch(entity)
            except (requests.RequestException, ValueError, KeyError) as exc:
                log.warning("Wikipedia lookup for %r failed: %s", entity, exc)
                return []  # not cached, so a later chunk can retry
            self._save()
        return self.cache[key]

    def _fetch(self, entity: str) -> list[str]:
        # One request: the search result is the generator for the extract + links
        # query. The full plaintext (not just the intro) is what links get ranked by.
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": entity,
            "gsrlimit": 1,
            # One word is often a misheard fragment ("Sansus"): require an exact
            # title/redirect match. Multi-word names get fuzzy full-text search.
            "gsrwhat": "text" if " " in entity.strip() else "nearmatch",
            "prop": "extracts|links|pageprops",
            "ppprop": "disambiguation",
            "explaintext": 1,
            "pllimit": "max",
            "plnamespace": 0,
            "redirects": 1,
        }
        resp = self.session.get(self.API, params=params, timeout=self.timeout, headers=self.HEADERS)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        if not pages:
            return []
        page = next(iter(pages.values()))
        if "disambiguation" in page.get("pageprops", {}):
            return []  # common words ("Up", "Last") land on disambiguation pages
        links = [link["title"] for link in page.get("links", [])]
        return rank_terms(page["title"], page.get("extract", ""), links, self.max_terms)

    def _save(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not write cache %s: %s", self.cache_path, exc)
