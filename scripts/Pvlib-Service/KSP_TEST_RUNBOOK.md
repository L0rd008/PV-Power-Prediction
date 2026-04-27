# KSP_Plant live-test runbook

One-time local smoke test before hand-off to the hosting team.

Plant under test: `KSP_Plant` (asset UUID `0e4b4070-50ff-11ef-b4ce-d5aee9e495ad`)
Test host: local Windows box, Python 3.12
Scope: KSP only (BFS root = KSP's own UUID, no other plants touched)

---

## Prerequisites check

Confirm on the TB side (one-time, before you start):

1. KSP_Plant asset has server attributes:
   - `isPlant = true`
   - `pvlib_enabled = true`
   - `latitude`, `longitude` set (not 0.0)
   - either `pvlib_config` blob OR flat PVsyst attrs (`tilt`, `azimuth`, `modules_per_string`, `strings_per_inverter`, etc.)
   - `weather_station_id` set to KSP_WSTN's device UUID (preferred) - otherwise the fallback path will try Contains-relation, then name-prefix search
2. KSP_WSTN is publishing recent `ghi` / `poa` and `air_temp` telemetry (within the last 2 minutes). If the station is stale, Tier-1 falls through to Solcast / clearsky.

---

## Step 1 - fill in credentials

Open `scripts\Pvlib-Service\.env` and set the two blank lines:

```text
TB_USERNAME=<your TB service-account email>
TB_PASSWORD=<your TB password>
```

Everything else is pre-filled for a KSP-only manual test.

---

## Step 2 - create venv and install

From an Administrator-free PowerShell prompt using **Python 3.12**:

```powershell
cd M:\Documents\Projects\MAGICBIT\Power-Prediction\scripts\Pvlib-Service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-py312-local.txt
```

`requirements-py312-local.txt` is a thin wrapper around `requirements.txt`, so the local smoke test uses the same locked dependency set as production on Python 3.12.

Do **not** use Python 3.13 for this runbook. With the current pinned stack, `pydantic-core` falls back to a local Rust/MSVC build on Windows and that install path is not supported here.

If `py -3.12` is not found, install Python 3.12 x64, reopen PowerShell, and rerun the commands above.

If you need a temporary local-only workaround on this machine today, `py -0p` shows Python 3.10 is installed. You can use `py -3.10 -m venv .venv` for a smoke test, but treat that as a stopgap and repeat the validation on Python 3.12 before hand-off.

---

## Step 3 - start the service (scheduler OFF)

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

You should see:

```text
pvlib-service starting (mode=pvlib, scheduler=false)
scheduler: disabled (SCHEDULER_ENABLED=false)
pvlib-service app created - routes registered
```

Leave this terminal running. Open a second terminal for the test calls.

---

## Step 4 - verify basic plumbing

In the second terminal:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expect: `200 OK` with body including `"status":"initializing"` (cold start, no cycle yet). That is the Gap-16 fix working.

```powershell
curl.exe http://127.0.0.1:8000/pvlib/discover
```

Expect a JSON list containing exactly one plant:

```json
[{"id": "0e4b4070-50ff-11ef-b4ce-d5aee9e495ad", "name": "KSP_Plant", "...": "..."}]
```

If the list is empty, KSP is missing `isPlant=true` or `pvlib_enabled=true` - fix on the TB side and retry.

---

## Step 5 - run the single-plant forecast

```powershell
curl.exe -X POST http://127.0.0.1:8000/pvlib/run-asset `
  -H "Content-Type: application/json" `
  -d '{"asset_id":"0e4b4070-50ff-11ef-b4ce-d5aee9e495ad"}'
```

Expected successful response (`status=ok`):

```json
{
  "asset_id": "0e4b4070-50ff-11ef-b4ce-d5aee9e495ad",
  "status": "ok",
  "source": "tb_station",
  "records": 2,
  "total_kwh": 0.214,
  "peak_kw": 8.123,
  "error": null
}
```

- `source` should be `tb_station` if KSP_WSTN was fresh. Fallbacks: `solcast`, `clearsky`.
- `records` is the number of 1-minute rows written (2 is normal for a 90-second window).
- `peak_kw` should be a plausible plant-level number. KSP rated capacity is 12.81 MW, so mid-day on a clear day this should hit several MW. Night or cloud means low output.

### Failure-mode expected outputs

If anything goes wrong, the service writes `-1` sentinels to TB for every 1-minute boundary in the window. The HTTP response will be `500` with a `detail` containing the reason:

| `detail.status` | Meaning |
|---|---|
| `config_error` | PlantConfig parsing blew up - check `pvlib_config` JSON validity |
| `no_location` | `latitude` / `longitude` are still 0.0 on the asset |
| `data_error` | All three irradiance tiers failed (WSTN down + no Solcast key + clearsky error) |
| `no_data` | `select_irradiance` returned an empty dataframe - usually a station timestamp-alignment issue |
| `physics_error` | `compute_ac_power` raised - look at the server log for the pvlib traceback |
| `write_error` | TB rejected the telemetry post - usually JWT expiry or rate-limit |

In every one of these cases the `-1` sentinels are still written to TB (best effort), so the widget will not freeze on stale data.

---

## Step 6 - verify in ThingsBoard

Open the TB UI, navigate to `KSP_Plant` -> **Latest telemetry** tab. Within seconds of the `run-asset` call you should see new entries for:

| Key | Expected value |
|---|---|
| `potential_power` | a number (kW) - `-1` on failure |
| `active_power_pvlib_kw` | same number (dual-write alias) |
| `pvlib_data_source` | `tb_station` / `solcast` / `clearsky` / `error:<reason>` |
| `pvlib_model_version` | `pvlib-h-a3-v1` |
| `ops_expected_unit` | `kW` |

Timestamp should be within the last ~90 seconds.

---

## Step 7 - enable continuous 1-minute cycling

Once Step 6 shows good telemetry:

1. Stop uvicorn (`Ctrl+C`).
2. Edit `.env`: `SCHEDULER_ENABLED=true`.
3. Restart uvicorn with the same command.

You will see a cycle log line every minute:

```text
scheduler: cycle complete in 2.4 s - 1 plants, 0 failed
```

Watch TB - `potential_power` on `KSP_Plant` should tick once per minute.

Let it run for at least 5 minutes, then confirm:

```powershell
curl.exe http://127.0.0.1:8000/health
```

This should return `200` with `"status":"ok"`, `"last_cycle_finished_at"` within 90 seconds of now, and `"cycles_completed" >= 5`.

```powershell
curl.exe http://127.0.0.1:8000/metrics
```

This should show `data_source_count{source="tb_station"} >= 5` (or whichever tier KSP used).

---

## Step 8 - sign-off checklist

Before hand-off to the hosting team, confirm all of:

- [ ] `run-asset` returned `status=ok` at least once
- [ ] `potential_power` is visible in KSP latest telemetry
- [ ] With the scheduler on, 5+ consecutive 1-minute writes land
- [ ] `/health` returns `200` after the first cycle
- [ ] The log shows no repeated tracebacks
- [ ] `data_source` matches the tier you expect (`tb_station` on a healthy day)

If all six pass, the service is production-ready for the hosting team.

---

## Roll-back

To restore the production fleet config (all three roots, scheduler on) after testing:

```powershell
copy .env.example .env
# Re-fill TB_USERNAME / TB_PASSWORD in the fresh copy
```

---

## Troubleshooting cheatsheet

| Symptom | Fix |
|---|---|
| `discover` returns empty | Check that KSP has `isPlant=true` AND `pvlib_enabled=true` (both server attributes) |
| `status=no_location` | Set `latitude` / `longitude` server attributes on KSP to non-zero values |
| `status=config_error` with a JSON parse message | `pvlib_config` blob is malformed - validate it in a JSON linter |
| `source=clearsky` despite KSP_WSTN existing | WSTN telemetry is stale (> 2 minutes old) OR `weather_station_id` is not set - check the station's latest telemetry timestamp |
| `write_error` with `401 Unauthorized` | `TB_USERNAME` / `TB_PASSWORD` are wrong, or the service account lacks write permission on KSP |
| `py -3.12` not found | Install Python 3.12 x64, reopen PowerShell, and rerun the setup |
| You used Python 3.13 and `pip install` fails on `pydantic-core` | Expected with the current pinned stack on Windows - recreate the venv with Python 3.12 instead |
| `ImportError: cannot import name 'retry'` | Run `pip install tenacity` - the requirements include it, but your venv may have been created from an older file |
