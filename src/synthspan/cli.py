"""Command-line interface: `synthspan generate ...`."""

from __future__ import annotations

import argparse
import random
import sys

from synthspan import __version__
from synthspan.balance import dedupe, label_counts
from synthspan.gazetteer import Gazetteer
from synthspan.generate import generate
from synthspan.template import Template
from synthspan.typos import augment
from synthspan.writers import write


def _cmd_generate(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    gaz = Gazetteer.from_csv(args.entities)
    templates = Template.load(args.templates)

    examples = generate(gaz, templates, args.n, rng=rng, balanced=args.balanced)
    if args.dedupe:
        examples = dedupe(examples)
    if args.typo_rate > 0:
        examples = augment(examples, rate=args.typo_rate, rng=rng, typo_entities=args.typo_entities)

    write(examples, args.out, fmt=args.format)

    counts = label_counts(examples)
    print(f"Wrote {len(examples)} examples to {args.out} ({args.format})", file=sys.stderr)
    print(f"Label spans: {dict(counts)}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthspan", description="Synthetic labeled-span data generator")
    p.add_argument("--version", action="version", version=f"synthspan {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate a labeled dataset")
    g.add_argument("--entities", required=True, help="CSV of linked entities (header = labels)")
    g.add_argument("--templates", required=True, help="Text file of templates ({LABEL} slots)")
    g.add_argument("-n", type=int, default=1000, help="Number of examples to generate")
    g.add_argument("--out", required=True, help="Output file path")
    g.add_argument("--format", choices=["jsonl", "conll", "spacy"], default="jsonl")
    g.add_argument("--typo-rate", type=float, default=0.0, help="Per-word typo probability (0-1)")
    g.add_argument("--typo-entities", action="store_true", help="Allow typos inside entity spans")
    g.add_argument("--balanced", action="store_true", help="Even coverage of templates/records")
    g.add_argument("--dedupe", action="store_true", help="Drop duplicate texts")
    g.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    g.set_defaults(func=_cmd_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
