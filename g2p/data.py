"""Data loading, vocabularies, dataset and batching for G2P."""

import pandas as pd
import torch
from torch.utils.data import Dataset

# Special tokens are fixed at indices <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3.
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def load_pairs(csv_path):
    """Load a G2P CSV into a list of (word, phoneme_list) tuples."""
    # keep_default_na=False so words like "null"/"nan" stay as strings.
    df = pd.read_csv(csv_path, keep_default_na=False)
    return [(str(w), str(p).split()) for w, p in zip(df["word"], df["phonemes"])]


def build_vocab(token_lists, specials=("<PAD>", "<SOS>", "<EOS>", "<UNK>")):
    """Build a token->index dict. Specials occupy indices 0..len(specials)-1;
    all remaining unique tokens follow in sorted order."""
    token_to_idx = {tok: i for i, tok in enumerate(specials)}
    unique = set()
    for tokens in token_lists:
        unique.update(tokens)
    unique -= set(specials)
    for tok in sorted(unique):
        token_to_idx[tok] = len(token_to_idx)
    return token_to_idx


class G2PDataset(Dataset):
    """Pairs of (word, phoneme sequence) encoded as integer index tensors."""

    def __init__(self, pairs, char_to_idx, phon_to_idx):
        self.pairs = pairs
        self.char_to_idx = char_to_idx
        self.phon_to_idx = phon_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        word, phonemes = self.pairs[idx]
        src_ids = [self.char_to_idx.get(ch, UNK_IDX) for ch in word]
        tgt_ids = (
            [SOS_IDX]
            + [self.phon_to_idx.get(p, UNK_IDX) for p in phonemes]
            + [EOS_IDX]
        )
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch):
    """Pad src and tgt sequences in a batch to equal length with <PAD>=0."""
    srcs, tgts = zip(*batch)
    src_len = max(s.size(0) for s in srcs)
    tgt_len = max(t.size(0) for t in tgts)

    src_padded = torch.full((len(batch), src_len), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_len), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, : s.size(0)] = s
        tgt_padded[i, : t.size(0)] = t
    return src_padded, tgt_padded
