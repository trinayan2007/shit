# IMPLEMENTATION STATUS

Chronological log of phases, files, commands actually executed, results, and fixes.

---

## Phase 0 — Inspect PPT + dataset

- **Files:** `scripts/profile_data.py`, `scripts/verify_data.py`, `scripts/cardinality.py`,
  `artifacts/ppt_text/slide1..13.txt`, `artifacts/dataset_assessment.md`, `artifacts/dataset_assessment.json`
- **Commands executed:**
  - PPT: unzipped `AP-2 (1).pptx` — all 13 slides are images → OCR via `rapidocr-onnxruntime`
  - `python scripts/profile_data.py` (chunked both CSVs), `verify_data.py`, `cardinality.py`
- **Result:** both CSVs = 1,939,207 rows, identical eventID sets; `dec12` timestamps
  month-granularity (0% ISO) vs `nineteen` 100% ISO → **selected nineteenFeaturesDf.csv**;
  713,423 duplicate eventIDs; 412 resolvable principals; **no label column**.
- **MITRE IDs verified** live on attack.mitre.org (T1562.008, T1619, T1580, T1078,
  T1078.004, T1087.004, T1098.001, T1136.003, T1069.003, T1531, T1496).

## Phase 1 — Structure

- `backend/app/{config.py,main.py,api/,services/,ml/,database/}`, `backend/requirements.txt`,
  `frontend/` (Vite+React), `scripts/`, `config/mitre_mapping.json`, `.env.example`.
- Env: Python 3.12 venv via uv; `torch 2.14.1+cpu`, pandas, fastapi, uvicorn, pytest, httpx.

## Phase 2 — Preprocessing

- **Files:** `backend/app/services/{data_loader,preprocessing}.py`, `scripts/preprocess.py`
- **Command:** `.venv/bin/python scripts/preprocess.py`
- **Errors found & fixed:**
  1. First run dropped only 63 dups (expected 713,423) — within-chunk duplicates were
     invisible to the `seen`-set → added `chunk["eventID"].duplicated()`.
  2. `epoch` off by 1000× (pandas 2.x parses to **microseconds**) → switched to
     `(dt - epoch) // Timedelta("1s")`.
  3. Root display name was a mangled ARN slice → now `root@811596193553`.
- **Result:** 1,939,207 in → **1,225,784 out** (exactly matches unique-ID count),
  0 invalid timestamps, 412 principals, 1,939,144→1,225,784 verified.

## Phase 3+4 — Features + sequences

- **Files:** `backend/app/services/{feature_engineering,sequence_builder}.py`,
  `scripts/build_features.py`
- **Commands:** `.venv/bin/python scripts/build_features.py` (ran twice — see fixes)
- **Errors found & fixed:**
  1. `pd.Epoch` doesn't exist → `pd.to_datetime(epoch, unit="s", utc=True)`.
  2. Dead `if False else` expression in vocab builder removed.
  3. `feature_vocab.json` was saved **without** `event_name_freq` → live re-encoding
     (E2E) crashed with KeyError → vocab now saved complete; pipeline re-run,
     produced bit-identical split/windows (determinism check passed).
- **Result:** 138-dim features; split at 2019-08-23T23:50:54Z (80/20 by time);
  980,627 train / 245,157 val rows; **61,341 train / 15,576 val windows**; 388
  padded short runs.

## Phase 5+6 — Training + inference

- **Files:** `backend/app/ml/{gru_autoencoder,dataset,train,predict}.py`,
  `scripts/{train,infer}.py`
- **Command:** `.venv/bin/python scripts/train.py --epochs 8` → **391.5 s**, best-val
  checkpoint saved. Losses: train 0.058→0.016, val 0.058→0.034.
- **Artifacts:** `models/gru_autoencoder.pt`, `model_config.json`,
  `preprocessing_artifacts.json`, `anomaly_threshold.json` (**threshold 0.05553479** =
  quantile 0.99 of actual train errors).
- **Command:** `.venv/bin/python scripts/infer.py --split all` → train flagged 1.001%
  (calibration ✓), val flagged 12.31% (temporal drift), step errors saved per split.
- **Fix:** `--split train` initially skipped step errors (CLI flag) → always collect;
  `STEP_ERRORS_NPY` constant replaced with per-split paths.

## Phase 7+8+9 — Risk + MITRE + response

- **Files:** `backend/app/services/{classification,mitre_mapping,response_engine}.py`,
  `config/mitre_mapping.json`, `scripts/detect.py`
- **Command:** `.venv/bin/python scripts/detect.py --split all` → **76,917 detections**
  (LOW 69,940 / MEDIUM 4,743 / HIGH 2,234), MITRE: no_mapping 5,186 · T1580 3,900 ·
  T1619 2,173 · T1078 1,472 · T1496 1,449 · T1087.004 942 · T1069.003 382 ·
  T1098.001 34 · T1098 14 · T1078.004 13 · T1136.003 11.
- **Fix:** `automatic_action_for` imported from wrong module (classification vs
  response_engine); `peak_event` column added to schema.

## Phase 10+11 — SQLite + FastAPI

- **Files:** `backend/app/database/{database,models}.py`, `backend/app/api/{deps,
  system,dashboard,events,threats,users}.py`, `backend/app/main.py`
- **Commands executed:** uvicorn on port **8001** (8000 occupied by another service);
  curl-verified: `/api/health`, `/api/dashboard`, `/api/events`, `/api/threats`,
  `/api/users`, `POST /api/threats/61431/block` → executed SIMULATED, re-click → 409,
  blocking LOW → 400.
- **Errors found & fixed:**
  1. Background uvicorn died with parent shell → relaunched with `setsid nohup`.
  2. KPI `detected_anomalies` was medium+high → now exact `score >= threshold` SQL (2,532).
  3. Dashboard `recent_threats.top_events` was an unparsed JSON **string** → React
     crashed (`join is not a function`) → `json.loads` added in dashboard router.

## Phase 12 — React dashboard

- **Files:** `frontend/{package.json,vite.config.js,index.html,src/{main.jsx,App.jsx,api.js,styles.css}}`
- **Commands:** `npm install` · `npm run build` (✓ 830 modules) · `npm run dev` (5173,
  proxy → 8001 — curl-verified through proxy).
- **Headless Chrome verification:** DOM renders INSIDER THREAT DETECTION + all panels;
  KPIs **1,225,784 / 412 / 2,532 / 2,234** identical to API; **zero JS console errors**
  (after the top_events fix).

## Phase 15 — Tests

- **Files:** `backend/tests/{conftest,test_data,test_pipeline,test_model,test_rules,test_api}.py`
- **Command:** `.venv/bin/python -m pytest backend/tests -q`
- **First run:** 51 passed / 3 failed — all three were wrong *test expectations*:
  (a) `find_runs` fixture ignored a principal change, (b) overfit test fed
  incompressible random noise to an 8-dim bottleneck (loss rightly plateaued) →
  changed to compressible structured sequences, (c) `risk=LOW` on /threats is
  correctly 422, not total=0.
- **Final:** **54 passed, 0 failed** (re-run after dashboard fix: 54 passed).

## Phase 16 — End-to-end validation

- **File:** `scripts/e2e_trace.py`
- **Command:** `.venv/bin/python scripts/e2e_trace.py` → **PASSED**:
  real detection id=67981 (`backup`, 2019-11-17) traced through all 9 stages;
  live model score 0.16310054 vs stored 0.16310057 (Δ≈3e-8); risk HIGH = HIGH;
  MITRE T1580 = T1580; response policy matches; API + dashboard KPIs equal SQL counts;
  raw CSV row re-encoded bit-exact vs `features.npy`.

## Phase 18 — Docs

- `README.md` (all required sections incl. Implemented/Simulated/Proposed split),
  this log, `.env.example`, `artifacts/dataset_assessment.{md,json}`.

## What was executed vs not

**Actually executed here:** full pipeline (preprocess → features → train → infer →
detect), uvicorn server, all API endpoints (curl + TestClient), `npm install`,
`npm run build`, vite dev server, headless-Chrome render check, 54 pytest tests,
E2E trace.

**Not executed / environment notes:**
- Port 8000 is occupied by another service in this environment → backend uses **8001**.
- Real AWS integration is intentionally **not** executed (simulated adapter only).
- GPU unused (CPU training: 8 epochs ≈ 6.5 min on this machine).
