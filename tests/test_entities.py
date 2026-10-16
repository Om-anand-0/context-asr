from context_asr.entities import extract_entities


def test_multiword_title_with_connectors():
    text = "I rewatched Game of Thrones with my sister."
    assert extract_entities(text) == ["Game of Thrones"]


def test_sentence_start_stopwords_are_skipped():
    text = "So I think the show is great. What a finale. The end."
    assert extract_entities(text) == []


def test_sentence_start_name_is_kept():
    assert extract_entities("Daenerys rides a dragon.") == ["Daenerys"]


def test_trailing_connector_is_not_included():
    assert extract_entities("We talked about Kubernetes and containers.") == ["Kubernetes"]


def test_connector_between_two_names_joins_them():
    assert extract_entities("We read Pride and Prejudice today.") == ["Pride and Prejudice"]


def test_runs_do_not_cross_sentences_or_commas():
    text = "I love Westeros. Tyrion is clever, Jon Snow is brave."
    assert extract_entities(text) == ["Westeros", "Tyrion", "Jon Snow"]


def test_possessive_is_stripped_and_duplicates_removed():
    text = "Tyrion's wine. Then Tyrion left."
    assert extract_entities(text) == ["Tyrion"]


def test_lone_pronoun_i_is_never_an_entity():
    assert extract_entities("Yes and I said no, then I'm done.") == []


def test_empty_text():
    assert extract_entities("") == []


def test_known_limitation_sentence_start_word():
    # Capitalized only because it starts the sentence; not a stopword, so it leaks.
    # Documented in the README; use_spacy=True avoids it.
    assert extract_entities("Somehow we watched Dune.") == ["Somehow", "Dune"]


def test_initials_stay_inside_a_name():
    assert extract_entities("It was written by George R. R. Martin in 1996.") == ["George R. R. Martin"]


def test_possessive_inside_a_name_is_kept():
    assert extract_entities("He joined the Night's Watch today.") == ["Night's Watch"]


def test_consecutive_connectors():
    assert extract_entities("We watched House of the Dragon.") == ["House of the Dragon"]
