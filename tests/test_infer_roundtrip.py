"""End-to-end inference check: load the trained zul bundle and predict words
taken from the zul test split."""

from pathlib import Path

import pytest
import torch

from g2p.bundle import load_bundle
from g2p.data import UNK_IDX, load_pairs

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "zul.pt"
TEST_CSV = ROOT / "data" / "zul" / "g2p_test.csv"


@pytest.mark.skipif(not BUNDLE.exists() or not TEST_CSV.exists(),
                    reason="trained zul bundle / test split not present")
def test_zul_bundle_predicts_test_words():
    model, char_to_idx, phon_to_idx, meta = load_bundle(BUNDLE, device="cpu")
    assert meta["config"]["lang"] == "zul"
    idx_to_phon = {i: t for t, i in phon_to_idx.items()}

    words = [w for w, _ in load_pairs(TEST_CSV)[:5]]
    assert len(words) == 5

    for word in words:
        src = torch.tensor([[char_to_idx.get(ch, UNK_IDX) for ch in word]],
                           dtype=torch.long)
        preds, _ = model.greedy_decode(src, max_len=30)
        phones = [idx_to_phon[t] for t in preds[0]]
        assert len(phones) > 0, f"empty prediction for {word!r}"
        assert all(isinstance(p, str) and p for p in phones)
