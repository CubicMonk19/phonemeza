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

## Run locally (no container)

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload      # http://127.0.0.1:8000
```

## Deploy

The app is containerized. The image is **CPU-only** (torch installed from the
PyTorch CPU wheel index — no CUDA), ~1.12 GB, runs as a non-root user, and
includes `espeak-ng` so the optional "reference pronunciation" audio works for
Afrikaans (`af` voice). `models/baselines/` is excluded from the image — the
`afr_none` bottleneck baseline is never served.

### Single container

```bash
# Build
docker build -t phonemeza:latest .

# Run (publish the app's port for direct access)
docker run -d --name phonemeza -p 8000:8000 phonemeza:latest

# Verify
curl http://127.0.0.1:8000/api/health
curl "http://127.0.0.1:8000/api/phonemize?word=umuntu&lang=zul"
```

### Production: Compose + Caddy (automatic HTTPS)

`docker-compose.yml` runs the app (internal only) behind `caddy:2`, which
publishes 80/443 and obtains/renews a Let's Encrypt certificate automatically
for the domain in `$DOMAIN`. Point the domain's DNS at the host and make sure
ports 80 and 443 are open first.

```bash
echo "DOMAIN=phonemeza.example.com" > .env   # your real domain
docker compose up -d --build

docker compose logs -f caddy                  # watch certificate provisioning
docker compose down                           # stop (named volumes persist certs)
```

The `caddy_data` volume persists issued certificates and the ACME account
across restarts — keep it to avoid hitting Let's Encrypt rate limits on
redeploys.

Both services use `restart: always`, so after a host reboot or crash the
Docker daemon (enabled on boot) brings the stack back up on its own — this is
the restart story; no systemd unit is needed.

### One-command deploy to EC2

`deploy/deploy.ps1` (Windows/PowerShell, the primary tool) builds the image,
ships it to an amd64 Ubuntu EC2 host over SSH, and runs the Compose stack.
`deploy/deploy.sh` is the equivalent for Linux/macOS operators. Configuration
lives in `deploy/.env` (gitignored — copy `deploy/.env.example` and fill in
`DOMAIN`, `EC2_HOST`, `SSH_USER`, `SSH_KEY`).

```powershell
copy deploy\.env.example deploy\.env   # then edit deploy\.env
.\deploy\deploy.ps1
```

What it does: resolves `DOMAIN` and asserts it points at `EC2_HOST` (aborts
otherwise, so a misconfigured DNS record can't burn Let's Encrypt's
5-failures/hour/domain budget) → `docker build` → `docker save` + `tar.exe`
compress → `scp` the image, `docker-compose.yml`, and `Caddyfile` to the host
→ `docker load` + `docker compose up -d` (passing `DOMAIN` through) → smoke
test `https://$DOMAIN/api/health` and a phonemize call, failing loudly on any
non-200. Re-running redeploys idempotently. The SSH user must be able to run
`docker` without sudo (member of the `docker` group).
