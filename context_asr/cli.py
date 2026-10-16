"""Command line: `context-asr transcribe ...` and `context-asr lookup ...`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .context import ChainSource, FileSource, WikipediaSource


def _split(value):
    return [t.strip() for t in value.split(",") if t.strip()] if value else []


def _source(args):
    sources = []
    if args.terms_file:
        sources.append(FileSource(args.terms_file))
    if not args.no_wikipedia:
        sources.append(WikipediaSource())
    return ChainSource(*sources)


def cmd_lookup(args):
    for term in _source(args).related_terms(args.entity):
        print(term)


def cmd_transcribe(args):
    from .evaluate import term_recall, wer
    from .pipeline import ContextualTranscriber

    transcriber = ContextualTranscriber(
        model=args.model,
        source=_source(args),
        hints=_split(args.hint),
        language=args.language,
        use_spacy=args.spacy,
        device=args.device,
        correct=not args.no_correct,
    )
    runs = {"contextual": transcriber.transcribe(args.audio)}
    if args.compare:
        runs = {"baseline": transcriber.transcribe(args.audio, use_context=False), **runs}

    for name, result in runs.items():
        if len(runs) > 1:
            print(f"--- {name} ---")
        print(result.text)

    report = {name: {"text": r.text, "chunks": r.chunks} for name, r in runs.items()}
    if args.reference:
        reference = Path(args.reference).read_text(encoding="utf-8")
        key_terms = _split(args.key_terms)
        print(f"\n{'':<12}{'WER':>8}{'term recall':>14}")
        for name, result in runs.items():
            metrics = {"wer": wer(reference, result.text)}
            recall = "-"
            if key_terms:
                metrics["term_recall"] = term_recall(reference, result.text, key_terms)
                recall = f"{metrics['term_recall']:.0%}"
            report[name]["metrics"] = metrics
            print(f"{name:<12}{metrics['wer']:>8.1%}{recall:>14}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="context-asr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source_args(p):
        p.add_argument("--terms-file", help='JSON {"entity": ["term", ...]}')
        p.add_argument("--no-wikipedia", action="store_true", help="don't query Wikipedia")
        p.add_argument("-v", "--verbose", action="store_true", help="log prompts per chunk")

    t = sub.add_parser("transcribe", help="transcribe an audio file")
    t.add_argument("audio")
    t.add_argument("--model", default="small", help="faster-whisper model (default: small)")
    t.add_argument("--device", default="auto", help="auto, cpu or cuda")
    t.add_argument("--language", help="e.g. en (default: auto-detect)")
    t.add_argument("--compare", action="store_true", help="also run the no-context baseline")
    t.add_argument("--reference", help="reference transcript for WER / term recall")
    t.add_argument("--key-terms", help='comma-separated, e.g. "Daenerys,Westeros"')
    t.add_argument("--hint", help="comma-separated terms always in the prompt")
    t.add_argument("--spacy", action="store_true", help="use spaCy NER instead of the heuristic")
    t.add_argument("--no-correct", action="store_true", help="don't rewrite near-miss names")
    t.add_argument("--json", help="write a JSON report here")
    add_source_args(t)
    t.set_defaults(func=cmd_transcribe)

    lk = sub.add_parser("lookup", help="show the terms an entity would add")
    lk.add_argument("entity")
    add_source_args(lk)
    lk.set_defaults(func=cmd_lookup)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    if args.verbose:
        logging.getLogger("context_asr").setLevel(logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
