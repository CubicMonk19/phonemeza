#!/usr/bin/env bash
# Thin bash equivalent of deploy.ps1 for Linux/macOS operators. The PowerShell
# script is the primary tool on the Windows deploy box; this mirrors its steps.
# Reads deploy/.env (gitignored). Local and remote are both amd64.
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] || { echo "deploy/.env not found; copy deploy/.env.example to deploy/.env"; exit 1; }
set -a; . ./.env; set +a
: "${DOMAIN:?set DOMAIN in deploy/.env}"
: "${EC2_HOST:?set EC2_HOST in deploy/.env}"
: "${SSH_USER:?set SSH_USER in deploy/.env}"
: "${SSH_KEY:?set SSH_KEY in deploy/.env}"
REMOTE_DIR="${REMOTE_DIR:-/home/$SSH_USER/phonemeza}"
REPO_ROOT="$(cd .. && pwd)"
REMOTE="$SSH_USER@$EC2_HOST"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

echo "=== 1/6 DNS pre-flight ==="
# getent (Linux) or dig fallback; assert DOMAIN's A record == EC2_HOST.
resolved="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk '{print $1}' | sort -u || true)"
[ -n "$resolved" ] || resolved="$(dig +short A "$DOMAIN" || true)"
if ! grep -qx "$EC2_HOST" <<<"$resolved"; then
  echo "DNS mismatch: $DOMAIN resolves to [$(echo "$resolved" | tr '\n' ' ')] but EC2_HOST is $EC2_HOST." >&2
  echo "Fix DNS before deploying — failed ACME challenges count toward Let's Encrypt's 5/hour/domain limit." >&2
  exit 1
fi
echo "OK: $DOMAIN -> $EC2_HOST"

echo "=== 2/6 docker build ==="
docker build -t phonemeza:latest "$REPO_ROOT"

echo "=== 3/6 docker save + gzip ==="
docker save phonemeza:latest | gzip > image.tar.gz

echo "=== 4/6 scp image + compose + Caddyfile ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"
scp "${SSH_OPTS[@]}" image.tar.gz "$REPO_ROOT/docker-compose.yml" "$REPO_ROOT/Caddyfile" "$REMOTE:$REMOTE_DIR/"

echo "=== 5/6 docker load + compose up -d ==="
ssh "${SSH_OPTS[@]}" "$REMOTE" \
  "cd '$REMOTE_DIR' && gunzip -f image.tar.gz && docker load -i image.tar && rm -f image.tar && DOMAIN='$DOMAIN' docker compose up -d --remove-orphans"

echo "=== 6/6 HTTPS smoke test ==="
for i in $(seq 1 20); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/health" || echo 000)"
  [ "$code" = "200" ] && break
  echo "  waiting for /api/health (attempt $i, last=$code)..."
  sleep 6
done
[ "$code" = "200" ] || { echo "Smoke test FAILED: /api/health -> $code" >&2; exit 1; }
echo "OK: /api/health -> 200"
code="$(curl -s -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/phonemize?word=umuntu&lang=zul" || echo 000)"
[ "$code" = "200" ] || { echo "Smoke test FAILED: /api/phonemize -> $code" >&2; exit 1; }
echo "OK: /api/phonemize -> 200"
echo "Deploy complete: https://$DOMAIN"
