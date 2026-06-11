"""Single-word inference on a saved model bundle."""

import torch

from g2p.bundle import load_bundle
from g2p.data import UNK_IDX

MAX_WORD_LEN = 40


class G2PPredictor:
    """Loads a bundle and phonemizes single words on CPU (or a given device)."""

    def __init__(self, bundle_path, device="cpu"):
        self.device = torch.device(device)
        model, char_to_idx, phon_to_idx, metadata = load_bundle(
            bundle_path, device=device)
        model.eval()
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_phon = {i: t for t, i in phon_to_idx.items()}
        self.metadata = metadata
        self.lang = metadata["config"]["lang"]

    def predict(self, word: str) -> dict:
        """Phonemize one word.

        Returns {"word", "phonemes", "attention", "chars"}; attention is a
        (decode_steps x len(word)) list of lists for attention models, else
        None. The word is normalized as in training: lowercased, diacritics
        preserved. Unknown characters map to <UNK> as in G2PDataset.
        """
        if not isinstance(word, str):
            raise ValueError("word must be a string")
        word = word.lower()
        if not word:
            raise ValueError("word must not be empty")
        if len(word) > MAX_WORD_LEN:
            raise ValueError(
                f"word too long ({len(word)} chars; max {MAX_WORD_LEN})")

        src = torch.tensor(
            [[self.char_to_idx.get(ch, UNK_IDX) for ch in word]],
            dtype=torch.long, device=self.device)
        preds, attn = self.model.greedy_decode(src, max_len=30)
        phonemes = [self.idx_to_phon.get(t, "<UNK>") for t in preds[0]]
        attention = attn[0].tolist() if attn is not None else None
        return {
            "word": word,
            "phonemes": phonemes,
            "attention": attention,
            "chars": list(word),
        }
