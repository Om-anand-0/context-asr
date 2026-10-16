from context_asr.context import FileSource
from context_asr.pipeline import ContextualTranscriber, group_segments


def test_group_segments_cuts_only_at_pauses():
    segs = [{"start": 0, "end": 40}, {"start": 50, "end": 90}, {"start": 100, "end": 140}]
    assert group_segments(segs, max_len=100) == [(0, 90), (100, 140)]


def test_group_segments_hard_splits_long_stretch():
    assert group_segments([{"start": 0, "end": 250}], max_len=100) == [(0, 100), (100, 200), (200, 250)]


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """Spells 'Daenerys' right only when the prompt contains it."""

    def __init__(self):
        self.prompts = []

    def transcribe(self, audio, initial_prompt=None, **kwargs):
        self.prompts.append(initial_prompt)
        name = "Daenerys" if initial_prompt and "Daenerys" in initial_prompt else "Denaris"
        return [FakeSegment(f" On Game of Thrones, {name} rules.")], None


def _run(use_context):
    import numpy as np

    t = ContextualTranscriber(model=FakeModel(), source=FileSource({"Game of Thrones": ["Daenerys"]}))
    t._chunks = lambda audio: [(0, 16000)]
    return t, t.transcribe(np.zeros(16000, dtype=np.float32), use_context=use_context)


def test_refine_pass_uses_new_context():
    t, result = _run(use_context=True)
    assert "Daenerys" in result.text
    assert result.chunks[0]["refined"] is True
    assert len(t.model.prompts) == 2


def test_baseline_skips_lookup():
    t, result = _run(use_context=False)
    assert "Denaris" in result.text
    assert len(t.model.prompts) == 1


class MishearingModel:
    """Says the topic in chunk 1, mishears a glossary name in chunk 2."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, initial_prompt=None, **kwargs):
        self.calls += 1
        text = " We love Game of Thrones." if self.calls <= 2 else " Then Brand Stark spoke."
        return [FakeSegment(text)], None


class NoLookups:
    def related_terms(self, entity):
        assert entity == "Game of Thrones", f"unexpected lookup: {entity}"
        return ["Bran Stark", "Westeros"]


def test_near_miss_resolves_locally_and_is_corrected():
    import numpy as np

    t = ContextualTranscriber(model=MishearingModel(), source=NoLookups())
    t._chunks = lambda audio: [(0, 16000), (16000, 32000)]
    result = t.transcribe(np.zeros(32000, dtype=np.float32))
    assert result.chunks[1]["text"] == "Then Bran Stark spoke."
    assert result.chunks[1]["corrections"] == {"Brand Stark": "Bran Stark"}
    assert result.chunks[1]["refined"] is False
