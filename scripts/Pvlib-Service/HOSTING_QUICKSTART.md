# Pvlib-Service — Hosting Quickstart

*EC2/Docker copy-paste runbook for operators. Version 1.0 — 2026-05-15.*

> **Audience**: the engineer or ops team who receives the Docker image and needs to get it running on an AWS EC2 instance against a live ThingsBoard tenant. Every command is copy-pasteable; substitute your own values where angle-brackets appear.

---

## 1. Prerequisites

| Item | Minimum |
|---|---|
| EC2 instance | `t4g.small` (2 vCPU ARM64, 2 GB RAM). `t3.small` works too (x86). |
| OS | Amazon Linux 2023 or Ubuntu 22.04 |
| Docker | 24+ (install below) |
| Outbound HTTPS | Port 443 open to `api.pvgis.eu` (PVGIS), `api.solcast.com.au` (Solcast), your TB host |
| ThingsBoard | Self-hosted or Cloud; any recent version supporting REST API |
| Domain / IP | Public or VPC-internal address for the Pvlib-Service API |

---

## 2. Launch EC2 (if starting from scratch)

```bash
# In AWS console or CLI — ARM64 Graviton is ~20 % cheaper at same perf
aws ec2 run-instances \
  --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
  --instance-type t4g.small \
  --key-name <your-keypair> \
  --security-group-ids <sg-id> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pvlib-service}]'
```

Open inbound ports in your security group:
- **22** (SSH) — from your IP only
- **8000** (API) — from TB server IP or internal VPC CIDR

---

## 3. Install Docker

```bash
# Amazon Linux 2023
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# Log out and back in so group takes effect
```

```bash
# Ubuntu 22.04
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```

Verify:

```bash
docker run --rm hello-world
```

---

## 4. Transfer the Docker Image

**Option A — ECR (recommended)**

```bash
# On your build machine
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ap-southeast-1.amazonaws.com

docker build -t pvlib-service .
docker tag pvlib-service:latest <account>.dkr.ecr.ap-southeast-1.amazonaws.com/pvlib-service:latest
docker push <account>.dkr.ecr.ap-southeast-1.amazonaws.com/pvlib-service:latest

# On the EC2 instance
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ap-southeast-1.amazonaws.com
docker pull <account>.dkr.ecr.ap-southeast-1.amazonaws.com/pvlib-service:latest
docker tag <account>.dkr.ecr.ap-southeast-1.amazonaws.com/pvlib-service:latest pvlib-service:latest
```

**Option B — save/load tar (air-gapped)**

```bash
# On build machine
docker save pvlib-service:latest | gzip > pvlib-service.tar.gz
scp pvlib-service.tar.gz ec2-user@<ec2-ip>:~

# On EC2
docker load < pvlib-service.tar.gz
```

---

## 5. Create the `.env` File

```bash
mkdir -p /opt/pvlib-service
cat > /opt/pvlib-service/.env << 'EOF'
# ── ThingsBoard connection ──────────────────────────────────────────
TB_HOST=https://tb.example.com
TB_USERNAME=pvlib@tenant.example.com
TB_PASSWORD=<secret>

# ── Plant discovery root ────────────────────────────────────────────
# Comma-separated UUIDs of the top-level ThingsBoard assets to BFS from.
# The service will discover all assets under these roots where
# isPlant=true AND pvlib_enabled=true.
TB_ROOT_ASSET_IDS=<uuid1>,<uuid2>

# ── Timezone (IANA) ─────────────────────────────────────────────────
TZ_LOCAL=Asia/Colombo

# ── Solcast (optional — fallback if weather station data absent) ────
SOLCAST_API_KEY=<your-solcast-key>

# ── Scheduler cadence ───────────────────────────────────────────────
# How often the live physics cycle runs (seconds). Default: 60.
CYCLE_INTERVAL_S=60

# ── Optional feature flags ──────────────────────────────────────────
# Enable to compute monthly/yearly LKR revenue telemetry.
REVENUE_JOB_ENABLED=false

# Enable to run the Sunday 03:00 zero-touch onboarding backfill.
AUTO_ONBOARD_ENABLED=false

# Per-plant timeout for the auto-onboard backfill (seconds). Default: 900.
# AUTOONBOARD_PER_PLANT_TIMEOUT_S=900

# Plant-attrs TTL cache (seconds). 0 disables caching. Default: 300.
# PLANT_ATTRS_CACHE_TTL_S=300
EOF
chmod 600 /opt/pvlib-service/.env
```

> **Security**: this file contains credentials. Ensure it is readable only by the user running Docker (`chmod 600`). Do not commit it to version control.

---

## 6. Start the Service

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

## 7. First-Run Verification

Run these in order, substituting `<KSP_UUID>` with your pilot plant's asset UUID.

### 7a. Discover plants

```bash
curl -s "http://localhost:8000/pvlib/discover" | python3 -m json.tool | head -40
# Expect: list of plants with pvlib_enabled=true
```

### 7b. Single-plant physics test

```bash
curl -s -X POST "http://localhost:8000/pvlib/run-asset?asset_id=<KSP_UUID>" | python3 -m json.tool
# Expect: {"status": "ok", "data_source": "tb_station"|"solcast"|"clearsky", ...}
```

### 7c. Daily rollup (yesterday)

```bash
YESTERDAY=$(date -u -d "yesterday" +%Y-%m-%d)  # Linux
curl -s -X POST "http://localhost:8000/admin/run-daily?date=${YESTERDAY}" | python3 -m json.tool
# Expect: {"plants_ok": N, "plants_failed": 0, "errors": 0}
```

### 7d. P-values for current year

```bash
curl -s -X POST "http://localhost:8000/admin/run-pvalues-plant?asset_id=<KSP_UUID>" | python3 -m json.tool
# Expect: {"plants_ok": 1, "plants_failed": 0}
# Runtime: 2-10 min (PVGIS fetch + computation)
```

### 7e. Check metrics

```bash
curl -s "http://localhost:8000/metrics"
# Expect Prometheus text format with pvlib_cycle_duration_seconds, pvlib_total_cycles_total, etc.
```

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
docker stop pvlib-service
docker rm pvlib-service
docker run -d --name pvlib-service --restart unless-stopped \
  --env-file /opt/pvlib-service/.env -p 8000:8000 pvlib-service:latest
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
# Pull / load new image (see §4)
docker stop pvlib-service
docker rm pvlib-service
docker run -d --name pvlib-service --restart unless-stopped \
  --env-file /opt/pvlib-service/.env -p 8000:8000 pvlib-service:latest
docker logs pvlib-service --tail 20
curl http://localhost:8000/health
```

Zero-downtime upgrades are not required for this workload — the service writes TB telemetry every 60 s and a 10–30 s gap during restart is invisible on dashboards.

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
| `"tb_connected": false` in `/health` | Wrong `TB_HOST` / credentials | Check `.env`, verify TB API is reachable from the EC2 instance |
| `plants_discovered: 0` | `TB_ROOT_ASSET_IDS` wrong or no `pvlib_enabled=true` plants | Verify UUIDs and that at least one plant has `pvlib_enabled=true` |
| `data_source: clearsky` for all plants | Weather station TB key misconfigured or not in Contains relation | Set `station.ghi_key` / `poa_key` in `pvlib_config` attr; check TB Contains relation |
| `actual_daily_energy_kwh = -1` | Wrong `actual_power_keys`, missing meter data, or W/kW unit mismatch | Check `actual_power_keys` attr and `active_power_unit`; look for WARN in logs |
| P-value job hangs > 15 min | PVGIS API rate-limiting or timeout | Check outbound connectivity to `api.pvgis.eu`; retry during off-peak hours |
| Revenue keys `-1` | `tariff_rate_lkr` attribute missing or `commissioning_date` not set | Set both attributes; re-run `/admin/run-revenue-backfill` |
| Auto-onboard plant not completing | Per-plant timeout exceeded | Increase `AUTOONBOARD_PER_PLANT_TIMEOUT_S` or run the plant individually |

For detailed reference including all TB attributes, environment variables, and performance tuning, see **[HOSTING_REFERENCE.md](HOSTING_REFERENCE.md)**.
