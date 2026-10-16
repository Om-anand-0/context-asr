"""Speech-to-text that notices what you're talking about and biases Whisper toward the right names."""

from .context import ChainSource, ContextSource, FileSource, WikipediaSource
from .entities import extract_entities
from .evaluate import term_recall, wer
from .memory import ContextMemory
from .pipeline import ContextualTranscriber

__version__ = "0.1.0"
__all__ = [
    "ChainSource",
    "ContextMemory",
    "ContextSource",
    "ContextualTranscriber",
    "FileSource",
    "WikipediaSource",
    "extract_entities",
    "term_recall",
    "wer",
]
