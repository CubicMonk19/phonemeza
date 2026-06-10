"""Prepare an NCHLT-inlang pronunciation dictionary for G2P training.

Parses word<TAB>pron lines (pron = space-separated X-SAMPA phone tokens,
preserved exactly), dedupes pronunciation variants (keeps the first line per
word), shuffles with a fixed seed, and writes an 80/10/10 split as
g2p_train.csv / g2p_val.csv / g2p_test.csv with columns word,phonemes —
the format g2p.data.load_pairs expects.

Usage:
  python scripts/prepare_nchlt.py \
      --input data/raw/release/dictionaries/nchlt_zul.dict --lang zul \
      --outdir data/zul --seed 42
"""

import argparse
import csv
import random
from pathlib import Path


def parse_dict(path):
    """Parse an NCHLT .dict file into (word, phone_list) pairs, keeping the
    first pronunciation per word. Returns (pairs, n_variants_dropped)."""
    pairs = []
    seen = set()
    dropped = 0
    with open(path, encoding="utf-8", newline="") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            word, sep, pron = line.partition("\t")
            if not sep:
                raise ValueError(f"{path}:{lineno}: no TAB separator: {line!r}")
            phones = pron.split(" ")
            if not word or not pron or any(p == "" for p in phones):
                raise ValueError(f"{path}:{lineno}: malformed entry: {line!r}")
            if word in seen:
                dropped += 1
                continue
            seen.add(word)
            pairs.append((word, phones))
    return pairs, dropped


def write_split(path, pairs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "phonemes"])
        for word, phones in pairs:
            writer.writerow([word, " ".join(phones)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="NCHLT .dict file")
    ap.add_argument("--lang", required=True, help="language code (zul/xho/afr)")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = ap.parse_args()

    pairs, dropped = parse_dict(args.input)
    print(f"[{args.lang}] parsed {len(pairs)} unique words "
          f"({dropped} pronunciation-variant lines dropped)")

    random.Random(args.seed).shuffle(pairs)
    n = len(pairs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    splits = {
        "g2p_train.csv": pairs[:n_train],
        "g2p_val.csv": pairs[n_train:n_train + n_val],
        "g2p_test.csv": pairs[n_train + n_val:],
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        write_split(outdir / name, split)
        print(f"[{args.lang}] {name}: {len(split)} pairs")

    chars = {ch for w, _ in pairs for ch in w}
    phones = {p for _, ps in pairs for p in ps}
    lengths = [len(w) for w, _ in pairs]
    print(f"[{args.lang}] char vocab size:  {len(chars)} (raw, no specials)")
    print(f"[{args.lang}] phone vocab size: {len(phones)} (raw, no specials)")
    print(f"[{args.lang}] word length: min {min(lengths)} / "
          f"max {max(lengths)} / mean {sum(lengths) / n:.2f}")
    print(f"[{args.lang}] sample rows:")
    for word, phones_list in pairs[:3]:
        print(f"  {word},{' '.join(phones_list)}")


if __name__ == "__main__":
    main()
