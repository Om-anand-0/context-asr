import json

import requests

from context_asr.context import ChainSource, FileSource, WikipediaSource, rank_terms


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Stands in for requests.Session; records calls, returns canned JSON."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls += 1
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


GOT_PAYLOAD = {
    "query": {
        "pages": {
            "123": {
                "title": "Game of Thrones (TV series)",
                "extract": "Game of Thrones is an American fantasy drama created by "
                "David Benioff for HBO. It is set in Westeros.\n\n== Plot ==\n"
                "Daenerys Targaryen rises. Daenerys Targaryen rules. Winter Is Coming airs.",
                "links": [
                    {"title": "Daenerys Targaryen"},
                    {"title": "List of Game of Thrones episodes"},
                    {"title": "Westeros"},
                    {"title": "Winter Is Coming (film)"},
                    {"title": "2011"},
                    {"title": "Abbott Elementary"},
                ],
            }
        }
    }
}


def test_file_source_is_case_insensitive(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(json.dumps({"Game of Thrones": ["Westeros", "Daenerys"]}))
    source = FileSource(path)
    assert source.related_terms("game of thrones") == ["Westeros", "Daenerys"]
    assert source.related_terms("Unknown") == []


def test_chain_source_merges_and_dedupes():
    a = FileSource({"F1": ["Verstappen", "Hamilton"]})
    b = FileSource({"f1": ["hamilton", "Leclerc"]})
    assert ChainSource(a, b).related_terms("F1") == ["Verstappen", "Hamilton", "Leclerc"]


def test_rank_terms_orders_intro_names_first_and_filters():
    page = GOT_PAYLOAD["query"]["pages"]["123"]
    terms = rank_terms(page["title"], page["extract"], [l["title"] for l in page["links"]])
    assert terms[0] == "Game of Thrones"
    assert terms.index("David Benioff") < terms.index("Daenerys Targaryen")
    assert "Winter Is Coming" in terms
    assert not any(t.startswith("List of") for t in terms)
    assert "2011" not in terms
    assert terms.count("Westeros") == 1


def test_rank_terms_caps_length():
    links = [f"Name{i}" for i in range(100)]
    assert len(rank_terms("X", " ".join(links), links, max_terms=40)) == 40


def test_wikipedia_source_parses_and_caches(tmp_path):
    cache = tmp_path / "wiki.json"
    session = FakeSession(GOT_PAYLOAD)
    source = WikipediaSource(cache_path=cache, session=session)
    terms = source.related_terms("Game of Thrones")
    assert "Daenerys Targaryen" in terms and "Westeros" in terms

    # A fresh instance reads the disk cache instead of hitting the network.
    session2 = FakeSession(error=AssertionError("should not be called"))
    again = WikipediaSource(cache_path=cache, session=session2).related_terms("game of thrones")
    assert again == terms
    assert session2.calls == 0


def test_wikipedia_network_error_returns_empty(tmp_path):
    session = FakeSession(error=requests.ConnectionError("offline"))
    source = WikipediaSource(cache_path=tmp_path / "wiki.json", session=session)
    assert source.related_terms("Game of Thrones") == []


def test_wikipedia_no_results(tmp_path):
    source = WikipediaSource(cache_path=tmp_path / "wiki.json", session=FakeSession({"batchcomplete": ""}))
    assert source.related_terms("Zzxq") == []


def test_rank_terms_drops_intro_noise():
    extract = "Critics praised it. Many viewers loved Westeros. Set in the Iron Throne era."
    terms = rank_terms("Game of Thrones", extract, ["Westeros", "1925 Grand Prix season"])
    assert terms == ["Game of Thrones", "Westeros", "Iron Throne"]


def test_wikipedia_skips_disambiguation_pages(tmp_path):
    payload = {"query": {"pages": {"1": {"title": "Up", "pageprops": {"disambiguation": ""}, "links": [{"title": "Up (film)"}]}}}}
    source = WikipediaSource(cache_path=tmp_path / "wiki.json", session=FakeSession(payload))
    assert source.related_terms("Up") == []


def test_rank_terms_orders_links_by_frequency_and_drops_unmentioned():
    text = "Intro about Westeros.\n\n== Body ==\nAidan Gillen. Tyrion Lannister and Tyrion Lannister again."
    terms = rank_terms("Game of Thrones", text, ["Aidan Gillen", "Abbott Elementary", "Tyrion Lannister"])
    assert terms == ["Game of Thrones", "Tyrion Lannister", "Aidan Gillen"]
