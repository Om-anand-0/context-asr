# context-asr

Speech-to-text that notices what you're talking about and biases Whisper toward the right names.

## The problem

Whisper is great at ordinary speech and bad at rare names. Say "Daenerys Targaryen" and you get
"to Ennery's Targaryen"; say "Petyr Baelish" and you get "Potir Balish". Whisper can be steered
with an `initial_prompt`, but you have to know the vocabulary *before* you transcribe.

`context-asr` works that vocabulary out as it goes. When the speaker mentions an entity like
"Game of Thrones", it looks up related names (Westeros, Tyrion Lannister, Iron Throne, ...) and
puts them in the prompt for the audio that follows. It also re-decodes the current chunk, since the
topic and the hard name often come up in the same sentence:

```
first pass   Last night I finally finished watching Game of Thrones. Honestly, to Ennery's Targaryen and her dragon...
             └─ "Game of Thrones" → Wikipedia → Daenerys Targaryen, Westeros, Tyrion Lannister, ...
refined      Last night I finally finished watching Game of Thrones. Honestly, Daenerys Targaryen and her dragon...
```

When a name is misheard *after* its topic is live ("Brand Stark" while the glossary has "Bran
Stark"), it's matched phonetically against the glossary. It never goes to Wikipedia, and the
output is rewritten to the glossary spelling.

## How it works

```
 audio ──► VAD chunks (≤15 s, cut at pauses)
               │
               ▼
        ┌─────────────┐   prompt: "Glossary: Westeros, Tyrion Lannister, ... <last 25 words>"
        │   Whisper   │◄──────────────────────────────────────────────┐
        └──────┬──────┘                                               │
               │ text                                                 │
               ▼                                                      │
      extract_entities()  ──►  sounds like a glossary term? ──► mention it, fix the spelling
               │                  else new entity?  ──►  Wikipedia / terms.json
               │                                        │             │
               │                                        ▼             │
               │                           ContextMemory (topics      │
               │                           +1 on mention, ×0.7 per ───┘
               │                           chunk, round-robin glossary)
               ▼
   learned something new? → re-decode this chunk once with the new prompt
```

| Module | What it does |
|---|---|
| `entities.py` | `extract_entities(text)`: runs of capitalized words (with `of`/`the`/`and` inside titles), or spaCy NER with `use_spacy=True` |
| `context.py` | `WikipediaSource`, `FileSource`, `ChainSource`. All implement `related_terms(entity) -> list[str]` |
| `fuzzy.py` | `closest()` / `correct_text()`: near-miss names via Metaphone + edit distance |
| `memory.py` | `ContextMemory`: topic scores, decay, and the prompt builder (about 200 tokens max) |
| `pipeline.py` | `ContextualTranscriber`: chunking, the transcribe/learn/refine loop |
| `evaluate.py` | `wer()` and `term_recall()` |
| `cli.py` | the `context-asr` command |

## Install

```bash
pip install -e .              # or: pip install -e ".[spacy,dev]"
```

Requires Python 3.10+. The first run downloads the Whisper model from Hugging Face.

If `context-asr` isn't found after install (common with `pip install --user` on Windows),
run `python -m context_asr.cli` instead.

## Usage

```bash
# Transcribe (Wikipedia lookups on by default)
context-asr transcribe talk.mp3 --model small

# Offline, using your own vocabulary
context-asr transcribe talk.mp3 --no-wikipedia --terms-file examples/terms.json

# Always-on hints, plus a JSON report with every chunk's prompt
context-asr transcribe talk.mp3 --hint "Verstappen,Leclerc" --json report.json -v

# What would an entity add to the prompt?
context-asr lookup "Game of Thrones"
```

Other flags: `--device cpu|cuda|auto`, `--language en`, `--spacy`, `--no-correct`.
If you see `cublas64_12.dll is not found`, pass `--device cpu` or install the CUDA 12 libraries.

From Python:

```python
from context_asr import ContextualTranscriber, ChainSource, FileSource, WikipediaSource

t = ContextualTranscriber("small", source=ChainSource(FileSource("terms.json"), WikipediaSource()))
result = t.transcribe("talk.mp3")
print(result.text)
for chunk in result.chunks:
    print(chunk["prompt"], "->", chunk["text"])
```

`terms.json` maps an entity to terms, with case-insensitive keys:

```json
{"Formula 1": ["Max Verstappen", "Charles Leclerc", "DRS", "Monza"]}
```

## Results: Earnings-21

_Run in progress._ `benchmarks/earnings21.py` compares baseline, contextual and oracle prompting on the 11-call eval10 subset, scoring the dataset's own entity annotations.

## Benchmark with your own clip

1. Record or find a clip where the rare names matter and write the exact transcript to `ref.txt`.
2. List the names you care about.
3. Run:

```bash
context-asr transcribe clip.mp3 --compare --reference ref.txt     --key-terms "Daenerys Targaryen,Petyr Baelish,Westeros" --json report.json
```

This prints WER and term recall for both runs. The baseline uses the **same chunks**, but its
prompt is only the transcript tail, so the difference comes from the lookup alone. **Term
recall** is the main number: fixing five names barely moves WER, but those names are what people
notice. Use `--json` and `-v` to see each chunk's prompt, the topics it added and the corrections
it made. A short clip with a dozen names can only show the pipeline working; a difference of one
name is noise.

## Known limitations

- **Whisper's prompt is short.** It holds about 224 tokens, and we cap ours at about 200
  (estimated at 4 chars per token). A long topic list gets cut off, and two topics share the space
  round-robin.
- **The heuristic finds false entities.** Any capitalized word at the start of a sentence that
  isn't in `STOPWORDS` counts as an entity ("Somehow we..."). "And" can join two names into one
  ("Westeros and Essos"). Use `--spacy` for real NER.
- **Wikipedia is noisy and slow.** Each new entity costs one HTTP round trip, often 200–800 ms,
  cached afterwards in `~/.cache/context-asr/wikipedia-v2.json`. A misheard name whose topic
  isn't live yet can still fuzzy-match an unrelated article.
- **Phonetic correction can be wrong.** It only rewrites toward the leading words of glossary
  terms, but two real names that sound alike can still be confused. Turn it off with `--no-correct`.
- **Re-decoding costs time.** A chunk that teaches us something is transcribed twice.
- Network errors never break transcription. The lookup just returns nothing.

## Roadmap

- Streaming microphone input (the loop is already chunk-by-chunk)
- An LLM post-correction pass that fixes names using the glossary (beyond the phonetic one)
- Deepgram / AssemblyAI keyword boosting as alternative backends to the Whisper prompt

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests need no model and no network. Wikipedia responses are mocked.
`benchmarks/earnings21.py` reproduces the results table (needs a GPU for reasonable run time).

## License

MIT
