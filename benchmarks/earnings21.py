"""Benchmark on Earnings-21 eval10 (Rev, CC BY-SA 4.0): 11 real earnings calls, ~10 h.

Key terms come from the dataset's own entity annotations (people, companies,
products, places, ...), so nobody hand-picks the names being scored.

Conditions, all on the same VAD chunks:
  baseline    prompt = transcript tail only
  contextual  context-asr with Wikipedia lookups (the thing being tested)
  oracle      the call's true entity list as --hint (upper bound for prompting)

    python benchmarks/earnings21.py --models base small --device cuda

Results are cached per (model, condition, call) in --out, so an interrupted
run resumes. Prints a Markdown table of totals.
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import jiwer
import requests

from context_asr import ContextualTranscriber, WikipediaSource
from context_asr.evaluate import normalize, term_hits

BASE = "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21"
EVAL10 = [
    "4320211", "4341191", "4346818", "4359971", "4365024", "4366522",
    "4366893", "4367535", "4383161", "4384964", "4387332",
]
# Name-like entity classes. NORP ("American") and numbers/dates are excluded.
NAME_CLASSES = {"PERSON", "ORG", "PRODUCT", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "LAW"}


def fetch(path: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(f"{BASE}/{path}", timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def load_reference(call: str, data: Path) -> tuple[str, list[str]]:
    """Reference text, and the call's name entities ordered by frequency."""
    nlp = fetch(f"transcripts/nlp_references/{call}.nlp", data / f"{call}.nlp")
    tags = fetch(f"transcripts/wer_tags/{call}.wer_tag.json", data / f"{call}.wer_tag.json")
    types = {k: v["entity_type"] for k, v in json.loads(tags.read_text(encoding="utf-8")).items()}

    words, entity_tokens = [], {}
    for line in nlp.read_text(encoding="utf-8").splitlines()[1:]:
        token, _, _, _, punct, _, _, wer_tags = line.split("|")
        words.append(token + punct)
        for tag in ast.literal_eval(wer_tags):  # a list literal like "['1', '4']"
            if types.get(tag) in NAME_CLASSES:
                entity_tokens.setdefault(tag, []).append(token)

    counts = Counter(re.sub(r"^the ", "", " ".join(toks)) for toks in entity_tokens.values())
    return " ".join(words), [name for name, _ in counts.most_common()]


def run(model: str, args) -> dict:
    from faster_whisper import decode_audio

    out = Path(args.out) / model
    out.mkdir(parents=True, exist_ok=True)
    wiki = WikipediaSource()
    transcriber = None
    totals = {c: {"refs": [], "hyps": [], "hit": 0, "expected": 0} for c in args.conditions}

    for call in args.calls:
        reference, entities = load_reference(call, Path(args.data))
        audio = None
        for condition in args.conditions:
            cached = out / f"{call}.{condition}.json"
            if not cached.exists():
                if transcriber is None:
                    transcriber = ContextualTranscriber(model, language="en", device=args.device)
                if audio is None:
                    mp3 = fetch(f"media/{call}.mp3", Path(args.data) / f"{call}.mp3")
                    audio = decode_audio(str(mp3), sampling_rate=16000)
                transcriber.source = wiki if condition == "contextual" else None
                transcriber.hints = entities if condition == "oracle" else []
                transcriber._lookups = {}
                result = transcriber.transcribe(audio, use_context=condition != "baseline")
                cached.write_text(json.dumps({"text": result.text, "chunks": result.chunks}), encoding="utf-8")
                print(f"  {model} {call} {condition} done", flush=True)
            hyp = json.loads(cached.read_text(encoding="utf-8"))["text"]
            hit, expected = term_hits(reference, hyp, entities)
            t = totals[condition]
            t["refs"].append(normalize(reference))
            t["hyps"].append(normalize(hyp))
            t["hit"] += hit
            t["expected"] += expected
    return {
        c: {"wer": jiwer.wer(t["refs"], t["hyps"]), "recall": t["hit"] / t["expected"], **t}
        for c, t in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["base", "small"])
    parser.add_argument("--conditions", nargs="+", default=["baseline", "contextual", "oracle"])
    parser.add_argument("--calls", nargs="+", default=EVAL10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default=".local/earnings21", help="download cache")
    parser.add_argument("--out", default=".local/earnings21-results", help="per-call results")
    args = parser.parse_args()

    rows = []
    for model in args.models:
        for condition, r in run(model, args).items():
            rows.append(
                f"| {model} | {condition} | {r['wer']:.1%} | {r['recall']:.1%} | {r['hit']} / {r['expected']} |"
            )
    print(f"\nEarnings-21 eval10, {len(args.calls)} calls\n")
    print("| model | condition | WER | term recall | terms right |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
