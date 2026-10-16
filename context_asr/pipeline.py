"""The loop: transcribe a chunk, notice entities, look them up, prompt the next chunk."""

import logging
from dataclasses import dataclass, field

import numpy as np

from .context import ContextSource
from .entities import extract_entities
from .fuzzy import closest, correct_text
from .memory import ContextMemory

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def group_segments(segments: list[dict], max_len: int) -> list[tuple[int, int]]:
    """Merge VAD speech segments into chunks of at most `max_len` samples.

    Cuts only between segments (at pauses); a single segment longer than
    `max_len` has no pause to cut at, so it gets hard-split.
    """
    chunks: list[tuple[int, int]] = []
    start = end = None
    for seg in segments:
        s, e = seg["start"], seg["end"]
        if start is not None and e - start <= max_len:
            end = e
            continue
        if start is not None:
            chunks.append((start, end))
        while e - s > max_len:
            chunks.append((s, s + max_len))
            s += max_len
        start, end = s, e
    if start is not None:
        chunks.append((start, end))
    return chunks


@dataclass
class Result:
    text: str
    chunks: list[dict] = field(default_factory=list)


class ContextualTranscriber:
    def __init__(
        self,
        model="small",
        source: ContextSource | None = None,
        hints: list[str] | None = None,
        language: str | None = None,
        use_spacy: bool = False,
        correct: bool = True,
        max_chunk_seconds: float = 15.0,
        device: str = "auto",
        compute_type: str = "default",
    ):
        if isinstance(model, str):
            from faster_whisper import WhisperModel

            model = WhisperModel(model, device=device, compute_type=compute_type)
        self.model = model
        self.source = source
        self.hints = hints or []
        self.language = language
        self.use_spacy = use_spacy
        self.correct = correct
        self.max_chunk_seconds = max_chunk_seconds
        self._lookups: dict[str, list[str]] = {}  # entity -> terms, per transcriber

    def transcribe(self, audio, use_context: bool = True) -> Result:
        """`audio` is a file path or a 16 kHz mono float32 array.

        use_context=False is the baseline: same chunks, prompt is only the transcript tail.
        """
        if isinstance(audio, str):
            from faster_whisper import decode_audio

            audio = decode_audio(audio, sampling_rate=SAMPLE_RATE)

        memory = ContextMemory()
        if use_context and self.hints:
            memory.add_topic("hints", self.hints)
            memory.topics["hints"].score = float("inf")  # never decays away

        result = Result(text="")
        for start, end in self._chunks(audio):
            chunk = audio[start:end]
            prompt = memory.prompt(glossary=use_context)
            text = self._decode(chunk, prompt)

            new_topics = self._learn(text, memory) if use_context else []
            refined = bool(new_topics)
            if refined:
                # The trigger word and the hard name are often in the same chunk,
                # so decode it again now that we know the topic.
                prompt = memory.prompt()
                text = self._decode(chunk, prompt)

            corrections = {}
            if use_context and self.correct:
                # Cheap post-correction: "Brand Stark" -> "Bran Stark" when Bran is in context.
                text, corrections = correct_text(text, memory.all_terms())

            log.info("[%.1fs] prompt=%r\n  -> %s", start / SAMPLE_RATE, prompt, text)
            memory.add_text(text)
            memory.step()
            result.chunks.append(
                {
                    "start": start / SAMPLE_RATE,
                    "end": end / SAMPLE_RATE,
                    "prompt": prompt,
                    "text": text,
                    "new_topics": new_topics,
                    "refined": refined,
                    "corrections": corrections,
                }
            )
        result.text = " ".join(c["text"] for c in result.chunks if c["text"])
        return result

    def _chunks(self, audio: np.ndarray) -> list[tuple[int, int]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        speech = get_speech_timestamps(audio, VadOptions())
        return group_segments(speech, int(self.max_chunk_seconds * SAMPLE_RATE))

    def _decode(self, chunk: np.ndarray, prompt: str) -> str:
        segments, _ = self.model.transcribe(
            chunk,
            initial_prompt=prompt or None,
            language=self.language,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _learn(self, text: str, memory: ContextMemory) -> list[str]:
        """Bump known topics, look up new entities. Returns names of topics added."""
        added = []
        for entity in extract_entities(text, use_spacy=self.use_spacy):
            if memory.mention(entity):
                continue
            # "Tyrion" when Game of Thrones is live: same topic, no lookup.
            topic = memory.topic_for_term(entity)
            if topic:
                memory.mention(topic)
                continue
            # "Brand Stark" when Bran Stark is live: a mishearing, not a new topic.
            # Resolve it locally instead of sending it to Wikipedia.
            near = next((n for n, t in memory.topics.items() if closest(entity, t.terms)), None)
            if near:
                memory.mention(near)
                continue
            if self.source is None:
                continue
            if entity not in self._lookups:
                self._lookups[entity] = self.source.related_terms(entity)
            if self._lookups[entity]:
                memory.add_topic(entity, self._lookups[entity])
                added.append(entity)
        return added
