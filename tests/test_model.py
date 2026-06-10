"""Sanity tests ported from the notebook's check cell, plus bundle round-trip."""

import numpy as np
import pytest
import torch

from g2p.bundle import load_bundle, save_bundle
from g2p.data import EOS_IDX, SOS_IDX
from g2p.model import Decoder, Encoder, Seq2Seq

SEED = 42
SRC_VOCAB = 30
TGT_VOCAB = 44
EMB, HID = 32, 64


def make_fake_batch():
    # Fake batch: row 1 has two trailing <PAD> to exercise attention masking.
    src = torch.tensor([[4, 5, 6, 7, 8],
                        [9, 10, 11, 0, 0]], dtype=torch.long)
    tgt = torch.randint(4, TGT_VOCAB, (2, 7), dtype=torch.long)
    tgt[:, 0] = SOS_IDX
    tgt[:, -1] = EOS_IDX
    return src, tgt


def make_model(mode, num_layers=1):
    encoder = Encoder(SRC_VOCAB, EMB, HID, num_layers=num_layers)
    decoder = Decoder(TGT_VOCAB, EMB, HID, num_layers=num_layers,
                      context_mode=mode)
    return Seq2Seq(encoder, decoder)


@pytest.mark.parametrize("mode", ["none", "fixed", "attn"])
def test_forward_logits_shape(mode):
    torch.manual_seed(SEED)
    src, tgt = make_fake_batch()
    model = make_model(mode)
    logits = model(src, tgt, teacher_forcing=True)
    assert logits.shape == (2, 6, TGT_VOCAB)


@pytest.mark.parametrize("mode", ["none", "fixed", "attn"])
def test_greedy_decode_runs(mode):
    torch.manual_seed(SEED)
    src, _ = make_fake_batch()
    model = make_model(mode)
    preds, attn = model.greedy_decode(src, max_len=10)
    assert len(preds) == 2
    assert all(isinstance(p, list) for p in preds)
    assert all(len(p) <= 10 for p in preds)
    if mode == "attn":
        assert attn is not None and len(attn) == 2
        for b in range(2):
            # One attention row per emitted token, over all src positions.
            assert attn[b].shape == (len(preds[b]), src.size(1))
    else:
        assert attn is None


def test_attention_masks_pad_and_rows_sum_to_one():
    torch.manual_seed(SEED)
    src, tgt = make_fake_batch()
    encoder = Encoder(SRC_VOCAB, EMB, HID, num_layers=1)
    decoder = Decoder(TGT_VOCAB, EMB, HID, num_layers=1, context_mode="attn")

    enc_out, (h0, c0) = encoder(src)
    src_mask = (src != 0)
    _, _, A = decoder(tgt[:, :-1], (h0, c0), encoder_outputs=enc_out,
                      src_mask=src_mask, return_attn=True)
    A1 = A[1].detach().numpy()  # row with <PAD> at positions 3, 4
    assert np.allclose(A1[:, 3:], 0.0), "attention leaked onto <PAD>"
    assert np.allclose(A1.sum(axis=1), 1.0, atol=1e-5), "rows must sum to 1"


def test_non_teacher_forcing_rejected():
    src, tgt = make_fake_batch()
    model = make_model("none")
    with pytest.raises(NotImplementedError):
        model(src, tgt, teacher_forcing=False)


@pytest.mark.parametrize("mode", ["none", "fixed", "attn"])
def test_bundle_round_trip(tmp_path, mode):
    torch.manual_seed(SEED)
    src, _ = make_fake_batch()
    model = make_model(mode)
    model.eval()

    char_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    char_to_idx.update({chr(ord("a") + i): 4 + i for i in range(SRC_VOCAB - 4)})
    phon_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    phon_to_idx.update({f"P{i}": 4 + i for i in range(TGT_VOCAB - 4)})

    config = {"embed_dim": EMB, "hidden_size": HID, "num_layers": 1,
              "context_mode": mode, "lang": "en"}
    metrics = {"per": 0.123, "word_acc": 0.456}

    path = tmp_path / f"bundle_{mode}.pt"
    save_bundle(path, model, char_to_idx, phon_to_idx, config, metrics)

    loaded, char2, phon2, meta = load_bundle(path, device="cpu")
    assert char2 == char_to_idx
    assert phon2 == phon_to_idx
    assert meta["config"]["context_mode"] == mode
    assert meta["config"]["src_vocab_size"] == SRC_VOCAB
    assert meta["config"]["tgt_vocab_size"] == TGT_VOCAB
    assert meta["config"]["lang"] == "en"
    assert meta["metrics"] == metrics

    preds_orig, _ = model.greedy_decode(src, max_len=10)
    preds_loaded, _ = loaded.greedy_decode(src, max_len=10)
    assert preds_orig == preds_loaded
