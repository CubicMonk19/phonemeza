"""Training, evaluation and metric utilities for G2P models."""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from g2p.data import PAD_IDX

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def compute_loss(logits, tgt):
    """logits: (batch, tgt_len-1, vocab); tgt: (batch, tgt_len).
    Loss is over tgt[:, 1:] (excludes the leading <SOS>)."""
    targets = tgt[:, 1:].contiguous()
    return criterion(logits.reshape(-1, logits.size(-1)),
                     targets.reshape(-1))


def train_one_epoch(model, loader, optimizer, clip=1.0):
    device = next(model.parameters()).device
    model.train()
    total = 0.0
    n = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt, teacher_forcing=True)
        loss = compute_loss(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / n


@torch.no_grad()
def evaluate_loss(model, loader):
    device = next(model.parameters()).device
    model.eval()
    total = 0.0
    n = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt, teacher_forcing=True)
        loss = compute_loss(logits, tgt)
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / n


def train_model(model, train_loader, val_loader, lr=1e-3, max_epochs=30,
                patience=5, clip=1.0, verbose=True):
    """Train until val loss has not improved for `patience` epochs, or until
    `max_epochs`. Restores best weights. Returns a history dict with keys
    'train_losses', 'val_losses', 'best_val_loss', 'epochs_run'."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, clip=clip)
        vl = evaluate_loss(model, val_loader)
        train_losses.append(tr)
        val_losses.append(vl)
        epochs_run = epoch
        if verbose:
            print(f"Epoch {epoch:2d} | train_loss {tr:.4f} | "
                  f"val_loss {vl:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_run": epochs_run,
    }


# Edit distance for PER. Primary: editdistance. Fallbacks: nltk, then a
# pure-Python Levenshtein, so the package works without either installed.
try:
    import editdistance

    def seq_edit_distance(a, b):
        return editdistance.eval(a, b)
except ImportError:
    try:
        from nltk.metrics.distance import edit_distance as _nltk_ed

        def seq_edit_distance(a, b):
            return _nltk_ed(list(a), list(b))
    except ImportError:
        def seq_edit_distance(a, b):
            """Levenshtein distance over token sequences (lists)."""
            a, b = list(a), list(b)
            prev = list(range(len(b) + 1))
            for i, x in enumerate(a, 1):
                cur = [i]
                for j, y in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[-1] + 1,
                                   prev[j - 1] + (x != y)))
                prev = cur
            return prev[-1]


@torch.no_grad()
def evaluate_metrics(model, loader, idx_to_phon, sos_idx=1, eos_idx=2,
                     max_len=30):
    """Greedy-decode the loader and compute mean PER and word accuracy.
    Returns (per, word_acc)."""
    device = next(model.parameters()).device
    model.eval()
    total_per = 0.0
    total_correct = 0
    n = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        preds, _ = model.greedy_decode(src, max_len=max_len)
        tgt_list = tgt.cpu().tolist()
        for i, pred_ids in enumerate(preds):
            # Defensive: truncate prediction at first <EOS> if present.
            if eos_idx in pred_ids:
                pred_ids = pred_ids[:pred_ids.index(eos_idx)]
            # Reference: drop leading <SOS>, cut at first <EOS>, drop <PAD>.
            ref_ids = tgt_list[i][1:]
            if eos_idx in ref_ids:
                ref_ids = ref_ids[:ref_ids.index(eos_idx)]
            ref_ids = [t for t in ref_ids if t != PAD_IDX]

            pred_seq = [idx_to_phon.get(t, "<UNK>") for t in pred_ids]
            ref_seq = [idx_to_phon.get(t, "<UNK>") for t in ref_ids]

            dist = seq_edit_distance(pred_seq, ref_seq)
            total_per += dist / max(len(ref_seq), 1)
            total_correct += int(pred_seq == ref_seq)
            n += 1
    return total_per / n, total_correct / n
