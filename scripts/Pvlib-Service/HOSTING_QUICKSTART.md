# Pvlib-Service — Hosting Quickstart

*Docker copy-paste runbook for operators. Version 2.0 — 2026-05-15.*

> **Audience**: the engineer or ops team who receives the Docker image and needs to get it running on a VPS (Contabo, Hetzner, AWS, etc.) against a live ThingsBoard tenant. Every command is copy-pasteable; substitute your own values where angle-brackets appear.

### Placeholder Definitions

Before running commands, replace any value in angle brackets (`<...>`) with your actual configuration:

| Placeholder | What to replace it with |
|---|---|
| `<server-ip>` | The public IP address or SSH hostname of your server. |
| `<secret>` | The password for your ThingsBoard user account (`TB_USERNAME`). |
| `<uuid1>,<uuid2>` | The ThingsBoard UUIDs of your root assets (used for plant discovery). |
| `<KSP_UUID>` | The ThingsBoard UUID of a single pilot plant (e.g., KSP_Plant) used for testing. |
| `<UUID>` | The ThingsBoard UUID of a specific plant asset. |

*(Note: Variables prefixed with `$` like `$VERSION_CODENAME` and `$(dpkg --print-architecture)` are dynamic bash commands. Do **not** replace them manually; your terminal will evaluate them automatically.)*

---

## 1. Prerequisites

| Item | Minimum |
|---|---|
| VPS | 2 vCPU, 2 GB RAM (e.g. Contabo VPS S, Hetzner CX22). 4 GB for fleets > 50 plants. |
| OS | Ubuntu 22.04+ or Debian 12+ |
| Docker | 24+ (install below) |
| Outbound HTTPS | Port 443 open to `api.pvgis.eu` (PVGIS), `api.solcast.com.au` (Solcast), your TB host |
| ThingsBoard | Self-hosted or Cloud; any recent version supporting REST API |
| Domain / IP | Public or private address for the Pvlib-Service API |

---

## 2. Initial Server Setup

```bash
# SSH into your VPS
ssh thingsnode@<server-ip>

# Update and install essentials
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y curl git

# Create service directory
sudo mkdir -p /opt/pvlib-service
sudo chown thingsnode:thingsnode /opt/pvlib-service
```

---

## 3. Install Docker

**Option A — Docker official repo (recommended)**

```bash
# Add Docker's official GPG key and repo
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker thingsnode
# Log out and back in so group takes effect
```

**Option B — Quick path (Ubuntu's built-in docker.io + standalone compose)**

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker thingsnode

# Install docker-compose standalone (v2)
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
# Log out and back in so group takes effect
```

> **Note:** With Option B, use `docker-compose` (hyphenated) instead of `docker compose` (space) in all commands.

Verify:

```bash
docker run --rm hello-world
docker compose version   # Option A
# or: docker-compose version   # Option B
```

---

## 4. Transfer the Docker Image

**Option A — Build on-server (recommended)**

```bash
# Clone the repo to your home directory
cd ~
git clone https://github.com/L0rd008/PV-Power-Prediction.git

# Copy the service files into the directory we created in Step 2
cp -r PV-Power-Prediction/scripts/Pvlib-Service/* /opt/pvlib-service/
cd /opt/pvlib-service

# Build the Docker image
docker compose build
```

> **Widgets repo** (for dashboard reference, not needed on the server):
> `https://github.com/L0rd008/thingsnode-pv-bi-insights-widgets`

**Option B — save/load tar (air-gapped)**

```bash
# On build machine
cd /path/to/PV-Power-Prediction/scripts/Pvlib-Service
docker compose build
docker save pvlib-service:latest | gzip > pvlib-service.tar.gz
scp pvlib-service.tar.gz docker-compose.yml thingsnode@<server-ip>:/opt/pvlib-service/

# On server
cd /opt/pvlib-service
docker load < pvlib-service.tar.gz
```

---

## 5. Create the `.env` File

```bash
cat > /opt/pvlib-service/.env << 'EOF'
# ── ThingsBoard connection ──────────────────────────────────────────
TB_HOST=https://windforce.thingsnode.cc
TB_USERNAME=PVLib-Service@thingsnode.cc
TB_PASSWORD=<secret>

# ── Plant discovery root ────────────────────────────────────────────
# Comma-separated UUIDs of the top-level ThingsBoard assets to BFS from.
# The service will discover all assets under these roots where
# isPlant=true AND pvlib_enabled=true.
TB_ROOT_ASSET_IDS=<uuid1>,<uuid2>

# ── Timezone (IANA) ─────────────────────────────────────────────────
TZ_LOCAL=Asia/Colombo

# ── Scheduler cadence ───────────────────────────────────────────────
# How often the live physics cycle runs (minutes). Default: 1.
SCHEDULER_INTERVAL_MINUTES=5
SCHEDULER_ENABLED=true

# ── Logging ─────────────────────────────────────────────────────────
# Use DEBUG for first-touch to surface station-resolution decisions.
# Revert to INFO in steady-state production.
LOG_LEVEL=INFO
DEBUG=false
MODE=pvlib

# ── Solcast (optional — fallback if weather station data absent) ────
SOLCAST_API_KEY=

# ── Read window ─────────────────────────────────────────────────────
READ_LAG_SECONDS=30
READ_WINDOW_SECONDS=90
MAX_CONCURRENT_PLANTS=10

# ── Loss Rollup Configuration ──────────────────────────────────────
LOSS_ROLLUP_ENABLED=true
LOSS_DEFAULT_SETPOINT_KEYS=setpoint_active_power
LOSS_MIN_VALID_SAMPLES=360
LOSS_LIFETIME_PAGE_DAYS=90
LOSS_TODAY_PARTIAL_ENABLED=true

# ── P-Value Job ─────────────────────────────────────────────────────
PVALUE_JOB_ENABLED=true

# ── Weekly accuracy evaluation ───────────────────────────────────────
# Comma-separated plant UUIDs; leave blank to disable.
EVAL_PLANT_IDS=

# ── Optional feature flags ───────────────────────────────────────────
REVENUE_JOB_ENABLED=false
AUTO_ONBOARD_ENABLED=false
# AUTOONBOARD_PER_PLANT_TIMEOUT_S=900
# PLANT_ATTRS_CACHE_TTL_S=300
EOF
chmod 600 /opt/pvlib-service/.env
```

> **Security**: this file contains credentials. Ensure it is readable only by the user running Docker (`chmod 600`). Do not commit it to version control.

---

## 6. Start the Service

```bash
cd /opt/pvlib-service
docker compose up -d --build
```

Or if not using docker-compose:

```bash
docker run -d \
  --name pvlib-service \
  --restart unless-stopped \
  --env-file /opt/pvlib-service/.env \
  -p 8000:8000 \
  pvlib-service:latest
```

Check it started:

```bash
docker logs pvlib-service --tail 40
curl http://localhost:8000/health
```

> **Note:** If your `docker-compose.yml` maps the container port to a different host port (e.g. `8002:8000`), use that host port in curl commands: `curl http://localhost:8002/health`.

Expected health response:

```json
{
  "status": "ok",
  "tb_connected": true,
  "plants_discovered": 22,
  "last_cycle_at": "...",
  "last_cycle_duration_ms": 4200
}
```

---

## 7. First-Run Verification (Smoke Test)

> **You do not register plants manually.** The service auto-discovers every plant reachable from your `TB_ROOT_ASSET_IDS` root assets where `isPlant=true AND pvlib_enabled=true`. No per-plant setup is needed here.
>
> This section is a **layered smoke test** — each step proves one layer of the stack works before the automatic scheduler touches the full fleet. Run them in order; stop and debug if any step fails before continuing. Substitute `<KSP_UUID>` with your pilot plant's asset UUID.

### 7a. Verify plant discovery

Confirms the root asset UUIDs are correct and the service can reach ThingsBoard.

```bash
curl -s "http://localhost:8000/pvlib/discover" | python3 -m json.tool | head -40
# Expect: non-empty list of plants with pvlib_enabled=true
# If empty: check TB_ROOT_ASSET_IDS and that plants have isPlant=true AND pvlib_enabled=true
```

### 7b. Single-plant physics test

Runs the full physics pipeline for one plant and returns immediately. Testing one plant first gives a clean error if something is misconfigured, rather than 20+ errors from the full fleet at once. The `data_source` field tells you which data tier fired:
- `tb_station` → weather station data is working (best case)
- `solcast` → station data absent/stale, Solcast fallback used
- `clearsky` → neither station nor Solcast available (pvlib clear-sky model only)

```bash
curl -s -X POST "http://localhost:8000/pvlib/run-asset" \
  -H "Content-Type: application/json" \
  -d '{"asset_id": "<KSP_UUID>"}' \
  | python3 -m json.tool
# Expect: {"status": "ok", "data_source": "tb_station"|"solcast"|"clearsky", ...}
# Note: start/end are optional — omitting them uses the current read window.
```

### 7c. Daily rollup (yesterday)

Proves the meter-energy integration pipeline works without waiting for the 00:05 midnight cron. Tests that `actual_power_keys` and `active_power_unit` are configured correctly for the plant.

```bash
YESTERDAY=$(date -u -d "yesterday" +%Y-%m-%d)  # Linux
curl -s -X POST "http://localhost:8000/admin/run-daily?date=${YESTERDAY}" | python3 -m json.tool
# Expect: {"plants_ok": N, "plants_failed": 0, "errors": 0}
# actual_daily_energy_kwh = -1 means meter key or unit is misconfigured
```

### 7d. P-values for one plant

Proves outbound connectivity to PVGIS and the historical simulation pipeline. Running on one plant first (2–10 min) avoids wasting time on a full-fleet run if PVGIS is unreachable.

```bash
curl -s -X POST "http://localhost:8000/admin/run-pvalues-plant?asset_id=<KSP_UUID>" | python3 -m json.tool
# Expect: {"plants_ok": 1, "plants_failed": 0}
# Runtime: 2–10 min (PVGIS ERA5 fetch + pvlib simulation)
```

### 7e. Check metrics

Confirms the observability endpoint is up for ongoing monitoring.

```bash
curl -s "http://localhost:8000/metrics"
# Expect Prometheus text format with pvlib_cycle_duration_seconds, pvlib_total_cycles_total, etc.
```

Once all five steps pass, the scheduler is running and handling the full fleet automatically. No further manual steps are needed for normal operation.

---

## 8. Auto-Onboard a New Plant (Phase 4)

Once the plant's `commissioning_date` attribute is set in TB and `AUTO_ONBOARD_ENABLED=true` is in the `.env` (restart required after env change), you can trigger onboarding manually:

```bash
# Full fleet onboarding (all plants with onboarding_completed != true)
curl -s -X POST "http://localhost:8000/admin/run-autoonboard" | python3 -m json.tool

# Single plant (faster for smoke test)
curl -s -X POST "http://localhost:8000/admin/run-autoonboard?asset_id=<UUID>" | python3 -m json.tool
```

Expected response:

```json
{"attempted": 1, "completed": 1, "failed": 0, "skipped": 0}
```

After completion, verify in TB that `onboarding_completed=true` and `onboarding_completed_at` are set on the plant asset.

---

## 9. Enable Revenue Reporting

1. Set `tariff_rate_lkr` attribute on each plant (LKR/kWh, e.g. `45.0`).
2. Set `commissioning_date` attribute (ISO date string, e.g. `"2021-03-15"`).
3. Add `REVENUE_JOB_ENABLED=true` to `.env`.
4. Restart the container:

```bash
cd /opt/pvlib-service
docker compose down && docker compose up -d
```

5. Backfill revenue for one plant:

```bash
curl -s -X POST "http://localhost:8000/admin/run-revenue-backfill?asset_id=<UUID>&years_back=5" \
  | python3 -m json.tool
# Expect: {"months_ok": N, "months_failed": 0, "years_ok": N, "years_failed": 0}
```

---

## 10. Upgrade

```bash
cd /opt/pvlib-service
git pull origin main
docker compose down && docker compose up -d --build
docker logs pvlib-service --tail 20
curl http://localhost:8000/health
```

Zero-downtime upgrades are not required for this workload — the service writes TB telemetry every few minutes and a 10–30 s gap during restart is invisible on dashboards.

---

## 11. Useful One-Liners

```bash
# Tail logs live
docker logs -f pvlib-service

# Force plant re-discovery (invalidate BFS cache)
curl -s -X POST http://localhost:8000/admin/refresh-plants

# Trigger immediate physics cycle
curl -s -X POST http://localhost:8000/admin/run-now

# Run loss-rollup for a specific date
curl -s -X POST "http://localhost:8000/admin/run-loss-rollup?date=2026-05-01"

# Recompute lifetime loss attributes for full fleet
curl -s -X POST http://localhost:8000/admin/recompute-lifetime

# Run monthly revenue for a specific month
curl -s -X POST "http://localhost:8000/admin/run-revenue-monthly?year=2026&month=4"
```

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Job for docker.service failed` during install | Leftover `docker.io` socket or daemon config | Run `sudo systemctl stop docker.socket` and `sudo systemctl daemon-reload` then start docker. If it still fails, run `sudo dockerd` to see the exact error (often an incompatible `/etc/docker/daemon.json`). |
| `Bind for 0.0.0.0:8002 failed` | Port 8002 is already in use by another service | Edit `docker-compose.yml` and change `"8002:8000"` to an unused port like `"9002:8000"`. Remember to use the new port in all `curl` commands. |
| `"tb_connected": false` in `/health` | Wrong `TB_HOST` / credentials | Check `.env`, verify TB API is reachable from the server |
| `plants_discovered: 0` | `TB_ROOT_ASSET_IDS` wrong or no `pvlib_enabled=true` plants | Verify UUIDs and that at least one plant has `pvlib_enabled=true` |
| `data_source: clearsky` for all plants | Weather station TB key misconfigured or not in Contains relation | Set `station.ghi_key` / `poa_key` in `pvlib_config` attr; check TB Contains relation |
| `actual_daily_energy_kwh = -1` | Wrong `actual_power_keys`, missing meter data, or W/kW unit mismatch | Check `actual_power_keys` attr and `active_power_unit`; look for WARN in logs |
| P-value job hangs > 15 min | PVGIS API rate-limiting or timeout | Check outbound connectivity to `api.pvgis.eu`; retry during off-peak hours |
| Revenue keys `-1` | `tariff_rate_lkr` attribute missing or `commissioning_date` not set | Set both attributes; re-run `/admin/run-revenue-backfill` |
| Auto-onboard plant not completing | Per-plant timeout exceeded | Increase `AUTOONBOARD_PER_PLANT_TIMEOUT_S` or run the plant individually |

For detailed reference including all TB attributes, environment variables, and performance tuning, see **[HOSTING_REFERENCE.md](HOSTING_REFERENCE.md)**.
