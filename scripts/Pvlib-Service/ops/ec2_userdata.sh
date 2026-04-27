#!/usr/bin/env bash
# =============================================================================
# ops/ec2_userdata.sh — EC2 instance bootstrap for pvlib-service (Gap 21, Phase F)
#
# PURPOSE
# -------
# Pull all runtime secrets from AWS SSM Parameter Store at boot time and write
# them to a tmpfs-backed .env file.  The Docker container mounts /run/pvlib/
# so secrets are never written to persistent disk.
#
# DESIGN DECISIONS (Opus §3 Gap 21)
# ----------------------------------
#   1. SSM SecureString parameters are decrypted in-flight by the AWS CLI with
#      --with-decryption; the plaintext never touches EBS.
#   2. /run/pvlib is a tmpfs mount (size capped at 4 MB); it survives reboots
#      only until the next instance stop — this script re-runs on every boot.
#   3. The EC2 instance role must have ssm:GetParameters on the parameter paths
#      listed in SSM_PARAMS below.  No IAM keys in userdata.
#   4. Docker Compose reads the .env from env_file: /run/pvlib/.env  — no
#      --env-file flag needed on individual docker run calls.
#   5. The script is idempotent: re-running it regenerates the .env cleanly.
#
# SSM PARAMETER PATHS  (all in /pvlib-service/<param> by convention)
# -------------------------------------------------------------------
#   /pvlib-service/TB_HOST
#   /pvlib-service/TB_USERNAME
#   /pvlib-service/TB_PASSWORD        (SecureString)
#   /pvlib-service/SOLCAST_API_KEY    (SecureString)
#   /pvlib-service/TB_ROOT_ASSET_IDS
#   /pvlib-service/EVAL_PLANT_IDS
#   /pvlib-service/TZ_LOCAL           (optional; default Asia/Colombo)
#
# USAGE
# -----
#   Attach as EC2 User Data (base64-encoded or raw text).
#   Or run manually on an existing instance:
#       sudo bash ops/ec2_userdata.sh
#
# REQUIREMENTS
# ------------
#   - Amazon Linux 2023 / Ubuntu 22.04 (tested)
#   - aws-cli v2 installed (pre-installed on Amazon Linux 2023)
#   - docker + docker compose v2 installed
#   - EC2 instance profile with ssm:GetParameters permission
# =============================================================================

set -euo pipefail

LOG_TAG="pvlib-userdata"
log()  { echo "[$(date -u +%FT%TZ)] [$LOG_TAG] $*" | tee -a /var/log/pvlib-userdata.log; }
err()  { log "ERROR: $*"; exit 1; }

# ── 0. Verify AWS CLI is available ──────────────────────────────────────────
command -v aws >/dev/null 2>&1 || err "aws CLI not found.  Install aws-cli v2 first."
log "AWS CLI: $(aws --version 2>&1 | head -1)"

# ── 1. Determine AWS region ──────────────────────────────────────────────────
AWS_REGION="${AWS_DEFAULT_REGION:-}"
if [[ -z "$AWS_REGION" ]]; then
    # Auto-detect from EC2 instance metadata (IMDSv2)
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)
    if [[ -n "$TOKEN" ]]; then
        AWS_REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            "http://169.254.169.254/latest/meta-data/placement/region" 2>/dev/null || true)
    fi
fi
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
log "Region: $AWS_REGION"
export AWS_DEFAULT_REGION="$AWS_REGION"

# ── 2. Mount tmpfs at /run/pvlib ─────────────────────────────────────────────
SECRETS_DIR="/run/pvlib"
mkdir -p "$SECRETS_DIR"

if ! mountpoint -q "$SECRETS_DIR"; then
    mount -t tmpfs -o size=4m,mode=0700 tmpfs "$SECRETS_DIR"
    log "Mounted tmpfs at $SECRETS_DIR"
else
    log "tmpfs already mounted at $SECRETS_DIR — refreshing .env"
fi

ENV_FILE="$SECRETS_DIR/.env"
# Start fresh (tmpfs is empty after mount; this handles re-runs)
: > "$ENV_FILE"
chmod 600 "$ENV_FILE"

# ── 3. Helper: fetch a single SSM parameter ──────────────────────────────────
ssm_get() {
    local param_path="$1"
    local value
    value=$(aws ssm get-parameter \
        --name "$param_path" \
        --with-decryption \
        --query "Parameter.Value" \
        --output text 2>/dev/null) || true
    echo "$value"
}

# ── 4. Bulk-fetch parameters from SSM ───────────────────────────────────────
SSM_PREFIX="/pvlib-service"

# Parameters to fetch: SSM path → env var name
declare -A SSM_PARAMS=(
    ["$SSM_PREFIX/TB_HOST"]="TB_HOST"
    ["$SSM_PREFIX/TB_USERNAME"]="TB_USERNAME"
    ["$SSM_PREFIX/TB_PASSWORD"]="TB_PASSWORD"
    ["$SSM_PREFIX/SOLCAST_API_KEY"]="SOLCAST_API_KEY"
    ["$SSM_PREFIX/TB_ROOT_ASSET_IDS"]="TB_ROOT_ASSET_IDS"
    ["$SSM_PREFIX/EVAL_PLANT_IDS"]="EVAL_PLANT_IDS"
    ["$SSM_PREFIX/TZ_LOCAL"]="TZ_LOCAL"
)

# Collect all parameter names for a single API call (cheaper than N calls)
PARAM_NAMES=()
for path in "${!SSM_PARAMS[@]}"; do
    PARAM_NAMES+=("$path")
done

log "Fetching ${#SSM_PARAMS[@]} parameters from SSM (prefix: $SSM_PREFIX) …"

# Fetch in one batch (GetParameters supports up to 10 names)
SSM_RESPONSE=$(aws ssm get-parameters \
    --names "${PARAM_NAMES[@]}" \
    --with-decryption \
    --query "Parameters[*].{Name:Name,Value:Value}" \
    --output json 2>/dev/null) || SSM_RESPONSE="[]"

# Parse JSON and write to .env  (requires jq)
if command -v jq >/dev/null 2>&1; then
    # Fast path: use jq
    echo "$SSM_RESPONSE" | jq -r '.[] | "\(.Name)=\(.Value)"' | \
    while IFS='=' read -r ssm_name ssm_value; do
        env_var="${SSM_PARAMS[$ssm_name]:-}"
        if [[ -n "$env_var" && -n "$ssm_value" ]]; then
            # Quote the value to handle spaces/special chars
            printf '%s=%q\n' "$env_var" "$ssm_value" >> "$ENV_FILE"
        fi
    done
else
    # Fallback: fetch one-by-one with aws cli text output
    log "jq not found — falling back to sequential parameter fetch"
    for ssm_path in "${!SSM_PARAMS[@]}"; do
        env_var="${SSM_PARAMS[$ssm_path]}"
        value=$(ssm_get "$ssm_path")
        if [[ -n "$value" ]]; then
            printf '%s=%q\n' "$env_var" "$value" >> "$ENV_FILE"
        else
            log "WARNING: SSM parameter '$ssm_path' not found or empty — skipping"
        fi
    done
fi

# ── 5. Apply defaults for optional parameters ────────────────────────────────
# TZ_LOCAL defaults to Asia/Colombo if not in SSM
grep -q "^TZ_LOCAL=" "$ENV_FILE" 2>/dev/null || echo "TZ_LOCAL=Asia/Colombo" >> "$ENV_FILE"

# Always set these non-secret operational defaults (overridable by SSM later)
cat >> "$ENV_FILE" <<'DEFAULTS'
SCHEDULER_ENABLED=true
SCHEDULER_INTERVAL_MINUTES=1
MAX_CONCURRENT_PLANTS=5
READ_LAG_SECONDS=30
READ_WINDOW_SECONDS=90
LOG_LEVEL=INFO
DEFAULTS

log ".env written to $ENV_FILE ($(wc -l < "$ENV_FILE") lines)"

# ── 6. Validate required secrets are present ────────────────────────────────
REQUIRED_VARS=(TB_HOST TB_USERNAME TB_PASSWORD TB_ROOT_ASSET_IDS)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" "$ENV_FILE"; then
        MISSING+=("$var")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    log "WARNING: Required vars not found in SSM: ${MISSING[*]}"
    log "         Service may fail to start.  Check IAM permissions and SSM paths."
fi

# ── 7. Install Docker + Docker Compose if needed ────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker …"
    if command -v dnf >/dev/null 2>&1; then
        # Amazon Linux 2023
        dnf install -y docker
    elif command -v apt-get >/dev/null 2>&1; then
        # Ubuntu
        apt-get update -qq
        apt-get install -y docker.io docker-compose-v2
    fi
    systemctl enable --now docker
    log "Docker installed: $(docker --version)"
fi

# Ensure current user can run docker (no-op if already in group or running as root)
if [[ "$EUID" -ne 0 ]]; then
    usermod -aG docker "$USER" 2>/dev/null || true
fi

# ── 8. Pull and start pvlib-service ─────────────────────────────────────────
COMPOSE_DIR="${PVLIB_COMPOSE_DIR:-/opt/pvlib-service}"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"

if [[ -f "$COMPOSE_FILE" ]]; then
    log "Starting pvlib-service via docker compose …"
    cd "$COMPOSE_DIR"
    docker compose --env-file "$ENV_FILE" pull --quiet 2>/dev/null || true
    docker compose --env-file "$ENV_FILE" up -d --remove-orphans
    log "pvlib-service started"
else
    log "WARNING: docker-compose.yml not found at $COMPOSE_FILE"
    log "         Deploy the compose file to $COMPOSE_DIR and re-run this script,"
    log "         or start the container manually:"
    log "           docker run -d --name pvlib-service \\"
    log "             --env-file $ENV_FILE \\"
    log "             -p 8000:8000 \\"
    log "             <your-registry>/pvlib-service:latest"
fi

log "Bootstrap complete."
