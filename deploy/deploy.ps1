<#
.SYNOPSIS
  Build, ship, and run the PhonemeZA container on an amd64 Ubuntu EC2 host.

.DESCRIPTION
  Windows/PowerShell deploy script (Docker Desktop + OpenSSH + tar.exe on PATH).
  Reads configuration from deploy/.env. Local and remote are both amd64, so the
  image is shipped as-is (no cross-build).

  Steps: DNS pre-flight -> docker build -> docker save + tar.exe compress ->
  scp image + compose + Caddyfile -> ssh load + `docker compose up -d` ->
  HTTPS smoke test.

  Re-running is idempotent: the freshly loaded image replaces the old one and
  `docker compose up -d` recreates only what changed.
#>

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Assert-LastExit($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

$scriptDir = $PSScriptRoot
$repoRoot  = Split-Path -Parent $scriptDir

# --- Load deploy/.env -------------------------------------------------------
$envPath = Join-Path $scriptDir '.env'
if (-not (Test-Path $envPath)) {
    throw "deploy/.env not found. Copy deploy/.env.example to deploy/.env and fill it in."
}
$cfg = @{}
foreach ($line in Get-Content $envPath) {
    $t = $line.Trim()
    if ($t -eq '' -or $t.StartsWith('#')) { continue }
    $i = $t.IndexOf('=')
    if ($i -lt 1) { continue }
    $key = $t.Substring(0, $i).Trim()
    $val = $t.Substring($i + 1).Trim()
    if ($val.Length -ge 2 -and
        (($val[0] -eq '"' -and $val[-1] -eq '"') -or
         ($val[0] -eq "'" -and $val[-1] -eq "'"))) {
        $val = $val.Substring(1, $val.Length - 2)
    }
    $cfg[$key] = $val
}

foreach ($req in 'DOMAIN', 'EC2_HOST', 'SSH_USER', 'SSH_KEY') {
    if (-not $cfg.ContainsKey($req) -or [string]::IsNullOrWhiteSpace($cfg[$req])) {
        throw "deploy/.env is missing required key: $req"
    }
}
$DOMAIN    = $cfg.DOMAIN
$EC2_HOST  = $cfg.EC2_HOST
$SSH_USER  = $cfg.SSH_USER
$SSH_KEY   = $cfg.SSH_KEY
$REMOTE_DIR = if ($cfg.ContainsKey('REMOTE_DIR') -and $cfg.REMOTE_DIR) {
    $cfg.REMOTE_DIR
} else {
    "/home/$SSH_USER/phonemeza"
}
if (-not (Test-Path $SSH_KEY)) { throw "SSH key not found at: $SSH_KEY" }

$remote = "$SSH_USER@$EC2_HOST"
$sshOpts = @('-i', $SSH_KEY, '-o', 'StrictHostKeyChecking=accept-new')

# --- 1. DNS pre-flight ------------------------------------------------------
Write-Step "1/6 DNS pre-flight"
try {
    $dns = Resolve-DnsName -Name $DOMAIN -Type A -ErrorAction Stop
} catch {
    throw "DNS lookup for $DOMAIN failed: $($_.Exception.Message)"
}
$ips = @($dns | Where-Object { $_.IPAddress } | Select-Object -ExpandProperty IPAddress)
if ($ips -notcontains $EC2_HOST) {
    throw ("DNS mismatch: $DOMAIN resolves to [$($ips -join ', ')] but EC2_HOST " +
           "is $EC2_HOST.`nFix the A record and let it propagate BEFORE deploying " +
           "— a failed ACME challenge counts toward Let's Encrypt's limit of " +
           "5 failed validations per hour per domain.")
}
Write-Host "OK: $DOMAIN -> $EC2_HOST" -ForegroundColor Green

# --- 2. Build ---------------------------------------------------------------
Write-Step "2/6 docker build"
docker build -t phonemeza:latest $repoRoot
Assert-LastExit 'docker build'

# --- 3. Save + compress (tar.exe, no gzip pipe) -----------------------------
Write-Step "3/6 docker save + tar.exe compress"
$stage = Join-Path $scriptDir 'build'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$imageTar = Join-Path $stage 'image.tar'
$imageGz  = Join-Path $stage 'image.tar.gz'
Remove-Item -Force -ErrorAction SilentlyContinue $imageTar, $imageGz

docker save -o $imageTar phonemeza:latest
Assert-LastExit 'docker save'
# -C $stage so the archive holds just "image.tar", not the full local path.
tar.exe -czf $imageGz -C $stage image.tar
Assert-LastExit 'tar.exe compress'
Remove-Item -Force $imageTar
$gzMb = [math]::Round((Get-Item $imageGz).Length / 1MB, 1)
Write-Host "Compressed image: $imageGz ($gzMb MB)" -ForegroundColor Green

# --- 4. Copy artifacts to the host ------------------------------------------
Write-Step "4/6 scp image + compose + Caddyfile"
$compose = Join-Path $repoRoot 'docker-compose.yml'
$caddy   = Join-Path $repoRoot 'Caddyfile'
ssh @sshOpts $remote "mkdir -p '$REMOTE_DIR'"
Assert-LastExit 'ssh mkdir'
scp @sshOpts $imageGz $compose $caddy "${remote}:$REMOTE_DIR/"
Assert-LastExit 'scp'

# --- 5. Load + run (idempotent) ---------------------------------------------
Write-Step "5/6 docker load + docker compose up -d"
# DOMAIN is passed inline so compose's {$DOMAIN} placeholder resolves without
# shipping a .env to the host. --remove-orphans keeps redeploys clean.
$remoteCmd = @(
    "cd '$REMOTE_DIR'",
    "tar -xzf image.tar.gz",
    "docker load -i image.tar",
    "rm -f image.tar image.tar.gz",
    "DOMAIN='$DOMAIN' docker compose up -d --remove-orphans"
) -join ' && '
ssh @sshOpts $remote $remoteCmd
Assert-LastExit 'ssh deploy'

# --- 6. Smoke test ----------------------------------------------------------
Write-Step "6/6 HTTPS smoke test"
function Get-Status($url) {
    try {
        return (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15).StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return 0
    }
}

# First deploy: allow time for Caddy to obtain the Let's Encrypt certificate.
$healthUrl = "https://$DOMAIN/api/health"
$status = 0
foreach ($attempt in 1..20) {
    $status = Get-Status $healthUrl
    if ($status -eq 200) { break }
    Write-Host "  waiting for $healthUrl (attempt $attempt, last=$status)..."
    Start-Sleep -Seconds 6
}
if ($status -ne 200) {
    throw "Smoke test FAILED: $healthUrl returned $status (expected 200). Check 'docker compose logs caddy' on the host."
}
Write-Host "OK: /api/health -> 200" -ForegroundColor Green

$phonUrl = "https://$DOMAIN/api/phonemize?word=umuntu&lang=zul"
$status = Get-Status $phonUrl
if ($status -ne 200) {
    throw "Smoke test FAILED: /api/phonemize returned $status (expected 200)."
}
Write-Host "OK: /api/phonemize -> 200" -ForegroundColor Green

Write-Host "`nDeploy complete: https://$DOMAIN" -ForegroundColor Green
