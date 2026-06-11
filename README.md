# phonemeza

Grapheme-to-phoneme conversion for South African languages with a
from-scratch LSTM encoder-decoder (see `g2p/`).

## Results

Models trained with `scripts/train_language.py` (embed_dim=64,
hidden_size=256, 1 layer, batch 64, Adam lr=1e-3, gradient clip 1.0, early
stopping on val loss with patience 5, seed 42). Metrics are on the held-out
test split (~10% of each dictionary); PER = phone error rate (mean
edit-distance / reference length), word accuracy = exact-match rate.

| lang | context mode | test PER | test word acc | epochs |
|------|--------------|---------:|--------------:|-------:|
| zul (isiZulu)   | attn | 0.0031 | 98.7% | 16 |
| xho (isiXhosa)  | attn | 0.0017 | 99.0% | 25 |
| afr (Afrikaans) | attn | 0.0284 | 82.4% | 20 |
| afr (Afrikaans) | none (bottleneck) | 0.0894 | 69.6% | 27 |

The attention-vs-bottleneck comparison from the original CMUdict study
carries over: on Afrikaans — the least regular orthography of the three —
dot-product attention cuts the PER by ~3x (0.089 -> 0.028) and lifts word
accuracy from 69.6% to 82.4% versus the no-context bottleneck decoder, which
must squeeze the whole word into a single fixed encoder state.

## Data

The pronunciation data comes from the **NCHLT-inlang within-language
Pronunciation Dictionaries v1.0**: 15,000 words per language with broad
phonemic transcriptions in the X-SAMPA phone set, licensed **CC BY 3.0**,
created by North-West University and distributed via
[SADiLaR](https://www.sadilar.org/). This project ships the isiZulu
(`nchlt_zul.dict`), isiXhosa (`nchlt_xho.dict`) and Afrikaans
(`nchlt_afr.dict`) dictionaries, prepared into train/val/test splits with
`scripts/prepare_nchlt.py`.

Citation (required by the dataset's README):

> Marelie Davel, Willem Basson, Charl van Heerden and Etienne Barnard,
> "NCHLT Dictionaries: Project Report", Technical report, North-West
> University, May 2013.
