# phonemeza

Grapheme-to-phoneme conversion for South African languages with a
from-scratch LSTM encoder-decoder (see `g2p/`).

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
