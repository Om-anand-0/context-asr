from context_asr.fuzzy import closest, correct_text, similar


def test_one_edit_matches():
    assert similar("Brand Stark", "Bran Stark")


def test_same_sound_matches():
    assert similar("Peter Balish", "Petyr Baelish")
    assert similar("Denaris", "Daenerys")


def test_unrelated_names_do_not_match():
    assert not similar("Sandor Clegane", "Sansa Stark")
    assert not similar("Tyrion", "Tywin Lannister")
    assert not similar("Ross", "Rose")  # too short to trust


def test_closest_checks_word_windows_inside_terms():
    terms = ["Daenerys Targaryen", "Westeros"]
    assert closest("Denaris", terms) == "Daenerys"
    assert closest("Denaris Targarian", terms) == "Daenerys Targaryen"
    assert closest("Kubernetes", terms) is None


def test_closest_skips_exact_matches():
    assert closest("Westeros", ["Westeros"]) is None


def test_correct_text_rewrites_whole_words_only():
    text = "Brand Stark met Peter Balish. Brandon stayed."
    fixed, changes = correct_text(text, ["Bran Stark", "Petyr Baelish"])
    assert fixed == "Bran Stark met Petyr Baelish. Brandon stayed."
    assert changes == {"Brand Stark": "Bran Stark", "Peter Balish": "Petyr Baelish"}


def test_title_tail_is_not_a_correction_target():
    assert closest("Drogon", ["House of the Dragon"]) is None
