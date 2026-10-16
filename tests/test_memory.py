from context_asr.memory import ContextMemory


def test_prompt_format_with_tail():
    memory = ContextMemory()
    memory.add_topic("Game of Thrones", ["Westeros", "Daenerys"])
    memory.add_text("one two three")
    assert memory.prompt() == "Glossary: Westeros, Daenerys. one two three"


def test_prompt_tail_is_last_25_words():
    memory = ContextMemory()
    memory.add_text(" ".join(f"w{i}" for i in range(100)))
    assert memory.prompt().split() == [f"w{i}" for i in range(75, 100)]


def test_prompt_respects_token_budget():
    memory = ContextMemory(max_tokens=50)
    memory.add_topic("Big", [f"Term{i}" for i in range(500)])
    memory.add_text("hello there")
    prompt = memory.prompt()
    assert len(prompt) / 4 <= 50
    assert prompt.endswith("hello there")
    assert "Term0" in prompt


def test_topic_decays_and_is_dropped():
    memory = ContextMemory()
    memory.add_topic("F1", ["Verstappen"])
    for _ in range(8):
        memory.step()
    assert "F1" in memory.topics  # 0.7**8 ~ 0.058
    memory.step()
    assert "F1" not in memory.topics  # 0.7**9 ~ 0.040


def test_mention_boosts_score():
    memory = ContextMemory()
    memory.add_topic("F1", ["Verstappen"])
    memory.step()
    memory.mention("f1")
    assert abs(memory.topics["F1"].score - 1.7) < 1e-9


def test_two_topics_share_the_budget():
    memory = ContextMemory(max_tokens=30)
    memory.add_topic("Big", [f"Big{i}" for i in range(100)])
    memory.add_topic("Small", ["Alpha", "Beta"])
    prompt = memory.prompt()
    assert "Alpha" in prompt and "Beta" in prompt
    assert "Big0" in prompt


def test_no_glossary_when_disabled():
    memory = ContextMemory()
    memory.add_topic("F1", ["Verstappen"])
    memory.add_text("tail")
    assert memory.prompt(glossary=False) == "tail"


def test_empty_prompt():
    assert ContextMemory().prompt() == ""
