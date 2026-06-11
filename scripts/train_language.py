"""Train a G2P model for one language end-to-end using the g2p package.

Usage:
  python scripts/train_language.py --data data/zul --lang zul \
      --out models/zul.pt --context-mode attn \
      --embed-dim 64 --hidden-size 256 --batch-size 64 --lr 1e-3 \
      --max-epochs 30 --patience 5 --seed 42
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from g2p.bundle import save_bundle
from g2p.data import G2PDataset, build_vocab, collate_fn, load_pairs
from g2p.model import Decoder, Encoder, Seq2Seq
from g2p.train import evaluate_metrics, train_model

RESULTS_CSV = Path("models") / "results.csv"
RESULTS_FIELDS = ["lang", "context_mode", "test_per", "test_word_acc",
                  "epochs_run", "train_size"]


def append_result(row):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="directory with g2p_{train,val,test}.csv")
    ap.add_argument("--lang", required=True, help="language code")
    ap.add_argument("--out", required=True, help="output bundle path (.pt)")
    ap.add_argument("--context-mode", default="attn",
                    choices=["none", "fixed", "attn"])
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.lang}/{args.context_mode}] using device: {device}")

    train_pairs = load_pairs(os.path.join(args.data, "g2p_train.csv"))
    val_pairs = load_pairs(os.path.join(args.data, "g2p_val.csv"))
    test_pairs = load_pairs(os.path.join(args.data, "g2p_test.csv"))
    print(f"[{args.lang}/{args.context_mode}] pairs: train={len(train_pairs)} "
          f"val={len(val_pairs)} test={len(test_pairs)}")

    # Vocabularies from the training split only.
    char_to_idx = build_vocab([list(w) for w, _ in train_pairs])
    phon_to_idx = build_vocab([p for _, p in train_pairs])
    idx_to_phon = {i: t for t, i in phon_to_idx.items()}
    print(f"[{args.lang}/{args.context_mode}] vocab: chars={len(char_to_idx)} "
          f"phones={len(phon_to_idx)}")

    train_ds = G2PDataset(train_pairs, char_to_idx, phon_to_idx)
    val_ds = G2PDataset(val_pairs, char_to_idx, phon_to_idx)
    test_ds = G2PDataset(test_pairs, char_to_idx, phon_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(char_to_idx), args.embed_dim, args.hidden_size,
                      num_layers=args.num_layers)
    decoder = Decoder(len(phon_to_idx), args.embed_dim, args.hidden_size,
                      num_layers=args.num_layers,
                      context_mode=args.context_mode)
    model = Seq2Seq(encoder, decoder).to(device)

    history = train_model(model, train_loader, val_loader, lr=args.lr,
                          max_epochs=args.max_epochs, patience=args.patience,
                          clip=args.clip, verbose=True)

    test_per, test_word_acc = evaluate_metrics(model, test_loader, idx_to_phon)
    print(f"RESULT lang={args.lang} context_mode={args.context_mode} "
          f"test_per={test_per:.4f} test_word_acc={test_word_acc:.4f} "
          f"epochs_run={history['epochs_run']} "
          f"best_val_loss={history['best_val_loss']:.4f}")

    config = {
        "embed_dim": args.embed_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "context_mode": args.context_mode,
        "lang": args.lang,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "clip": args.clip,
        "seed": args.seed,
    }
    metrics = {
        "test_per": test_per,
        "test_word_acc": test_word_acc,
        "best_val_loss": history["best_val_loss"],
        "epochs_run": history["epochs_run"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(out, model, char_to_idx, phon_to_idx, config, metrics)
    print(f"[{args.lang}/{args.context_mode}] bundle saved to {out}")

    append_result({
        "lang": args.lang,
        "context_mode": args.context_mode,
        "test_per": round(test_per, 4),
        "test_word_acc": round(test_word_acc, 4),
        "epochs_run": history["epochs_run"],
        "train_size": len(train_pairs),
    })


if __name__ == "__main__":
    main()
