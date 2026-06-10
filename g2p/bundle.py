"""Single-file model bundles: weights + vocabularies + config + metrics."""

import torch

from g2p.model import Encoder, Decoder, Seq2Seq

BUNDLE_FORMAT_VERSION = 1


def save_bundle(path, model, char_to_idx, phon_to_idx, config, metrics):
    """Save a Seq2Seq model plus everything needed to reload and use it.

    config must contain: embed_dim, hidden_size, num_layers, context_mode,
    and lang (the language code). Vocab sizes are filled in from the vocab
    dicts. metrics is a dict of final test metrics, e.g.
    {"per": ..., "word_acc": ...}.
    """
    config = dict(config)
    config["src_vocab_size"] = len(char_to_idx)
    config["tgt_vocab_size"] = len(phon_to_idx)
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "char_to_idx": char_to_idx,
        "phon_to_idx": phon_to_idx,
        "config": config,
        "metrics": dict(metrics),
    }
    torch.save(bundle, path)


def load_bundle(path, device="cpu"):
    """Load a bundle saved by save_bundle.

    Reconstructs the Encoder/Decoder/Seq2Seq from the stored config and loads
    the weights onto `device` (CPU by default, regardless of the device the
    model was trained on). Returns (model, char_to_idx, phon_to_idx,
    metadata), where metadata holds the config and metrics.
    """
    bundle = torch.load(path, map_location=device)
    config = bundle["config"]

    encoder = Encoder(config["src_vocab_size"], config["embed_dim"],
                      config["hidden_size"], num_layers=config["num_layers"])
    decoder = Decoder(config["tgt_vocab_size"], config["embed_dim"],
                      config["hidden_size"], num_layers=config["num_layers"],
                      context_mode=config["context_mode"])
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    metadata = {
        "config": config,
        "metrics": bundle["metrics"],
        "format_version": bundle.get("format_version"),
    }
    return model, bundle["char_to_idx"], bundle["phon_to_idx"], metadata
