# Insider Threat Classification and Detection in Cloud-Based Environments
### using a Hybrid Model with MITRE ATT&CK Framework and Automated Blocking

Full-stack security application that analyzes **real AWS CloudTrail logs** with a
**hybrid GRU-Autoencoder**, scores behavioral anomaly, classifies risk
(LOW/MEDIUM/HIGH), maps detected behavior to **MITRE ATT&CK** (evidence-based),
executes a **risk-based response** (simulated blocking by default), and presents
everything in a single SOC dashboard (React) backed by FastAPI + SQLite.

---

## 1. Project description

Authorized cloud users can misuse legitimate access; CloudTrail records everything
but nobody can read 1.9M rows by hand. This system:

1. preprocesses the raw CloudTrail dataset (dedup, identity resolution, timestamps)
2. engineers per-event features and builds chronological behavioral sequences
3. trains a **hybrid GRU-Autoencoder** that reconstructs normal behavior
4. scores each window by its **actual reconstruction error**
5. applies a transparent **risk classification** rule layer (LOW/MEDIUM/HIGH)
6. maps evidence in detected windows to **MITRE ATT&CK techniques**
7. executes a **risk-based response** (log / escalate / simulated block)
8. persists everything to SQLite and serves it through FastAPI
9. renders one dark SOC dashboard in React (no marketing pages)

## 2. Architecture

```
ACTUAL CLOUDTRAIL DATA (data/nineteenFeaturesDf.csv, 1,939,207 rows)
        │  scripts/preprocess.py          (chunked, dedup 713,423 rows)
        ▼
artifacts/clean_events.csv (1,225,784 events, 412 principals)
        │  scripts/build_features.py      (temporal 80/20 split, train-only vocab)
        ▼
features.npy memmap (1,225,785 × 138) + train/val windows (61,341 / 15,576)
        │  scripts/train.py               (8 epochs, best-val checkpoint)
        ▼
models/gru_autoencoder.pt + anomaly_threshold.json (quantile 0.99)
        │  scripts/infer.py               (per-window + per-step errors)
        │  scripts/detect.py              (risk rules + MITRE + response → SQLite)
        ▼
backend/data/app.db (76,917 detections, 412 principals, responses)
        │  FastAPI (backend/app/main.py)
        ▼
React dashboard (frontend/) — KPIs, timeline, distribution, table, drawer, block
```

## 3. Dataset

| File | Rows | Role |
|---|---|---|
| `data/dec12_18features.csv` | 1,939,207 | **rejected** — month-granularity timestamps (0% ISO-valid) |
| `data/nineteenFeaturesDf.csv` | 1,939,207 | **used** — full ISO-8601 timestamps, real nulls, +`requestID` |

Both files share the identical unique eventID set (1,225,784) — they are two
preprocessing stages of the same events. Full measured assessment:
[`artifacts/dataset_assessment.md`](artifacts/dataset_assessment.md) / [`.json`](artifacts/dataset_assessment.json).

- **Coverage:** 2017-02-12T19:57:06Z → 2020-10-07T21:03:30Z (AWS account `811596193553`)
- **Services:** 164 event sources (ec2, sts, s3, iam, rds, lambda, …), 1,242 distinct event names
- **Identities:** IAMUser 1,825,650 · AWSService 57,616 · AssumedRole 42,315 · Root 10,997
- **Errors:** 104 distinct error codes, 22.25% of rows carry one
- **NO ground-truth insider labels exist** → the core ML problem is **unsupervised
  behavioral anomaly detection** (see §17 Limitations)

### Dataset structure (used columns)
`eventID, eventTime, sourceIPAddress, userAgent, eventName, eventSource, awsRegion,
eventVersion, userIdentitytype, eventType, requestID, userIdentityaccountId,
userIdentityprincipalId, userIdentityarn, userIdentityaccessKeyId,
userIdentityuserName, errorCode, errorMessage, requestParametersinstanceType`

## 4. Dataset limitations

1. **No insider-threat labels** — no supervised accuracy/precision/F1/ROC-AUC is
   claimed anywhere. Risk levels come from documented rules, not a trained classifier.
2. Public CTF-style corpus (flaws.cloud lineage), not a labeled insider dataset —
   training assumes majority-normal behavior (standard unsupervised assumption).
3. `backup` + `Level6` hold 94.7% of rows; `RunInstances` is 68% of all events.
4. No S3 object-level events (`GetObject`/`PutObject` = 0) → exfiltration techniques
   cannot be evidenced, so they never fire.
5. `dec12_18features.csv` unusable for sequencing (month-granularity timestamps).

## 5. Feature engineering (138 features)

| Block | Dim | Notes |
|---|---|---|
| temporal | 6 | hour sin/cos, dow sin/cos, is_weekend, is_night |
| userIdentitytype one-hot | 6 | IAMUser/AWSService/AssumedRole/Root/AWSAccount/EMPTY |
| eventType one-hot | 4 | AwsApiCall/AwsServiceEvent/AwsConsoleSignIn/AwsConsoleAction |
| awsRegion one-hot | 17 | all regions in data |
| service family top-25 + OTHER | 26 | `ec2.amazonaws.com` → `ec2` |
| eventName top-40 + OTHER + log-freq | 42 | high-cardinality handled via top-K + freq |
| semantic class flags | 4 | is_read / is_write / is_discovery / is_admin |
| error features | 10 | has_error + top-8 codes + OTHER |
| IP features | 14 | is_service_ip + top-12 + OTHER |
| userAgent class | 7 | console/powershell/awscli/boto3/javasdk/service/other |
| instance type | 2 | presence + accelerator (p2/g3/f1/…) |

Rules: **vocabulary built from the TRAIN split only** (leakage-safe); identity
fields (userName/ARN/accessKeyId/principalId) are **never model features** (used
only for grouping/display); no blind one-hot over high-cardinality columns.

## 6. Sequence construction

- Group by `principal_key` = `arn` → else `principalId` → else normalized `accountId`
  → else `sourceIP` (userName alone is wrong: it's empty for root/role/service events)
- Stable chronological order: (`epoch`, original row order) — **events never shuffled**
- Sliding windows: **length 32, stride 16**, tail window added for full coverage
- Short runs (<32 events): one zero-padded window; padding masked out of loss AND score
- **Temporal 80/20 split** at epoch 1566604254 (2019-08-23): windows are built
  strictly within their split, so no window crosses the boundary
- Result: **61,341 train / 15,576 val windows** (76,917 total)

## 7. GRU-Autoencoder architecture (`backend/app/ml/gru_autoencoder.py`)

```
input (B, 32, 138)
  → Linear(138→64) + ReLU          per-step feature representation
  → GRU encoder (64→64, 1 layer)   sequential/temporal modeling → latent state
  → GRU decoder                    reconstructs temporal structure from latent
  → Linear(64→64) ReLU → Linear(64→138)   back to feature space
  → reconstructed (B, 32, 138)
```

The GRU genuinely models temporal relationships (recurrent over T) and the
autoencoder genuinely reconstructs the input sequence; both are trained end-to-end
with masked MSE. Gradient flow through both parts is asserted by a test
(`test_model_is_genuinely_recurrent_and_trainable`).

## 8. Anomaly scoring & threshold calibration

- per-step error = mean squared reconstruction error across the 138 features
- **window anomaly score = masked mean of per-step errors** (pads excluded)
- threshold = **quantile 0.99 of the TRAINING score distribution** produced by the
  trained model → `0.05553479` (`models/anomaly_threshold.json` also records p95,
  mean, std, epochs, seed). Not an arbitrary 0.5/0.7/0.8 — a test asserts that.
- measured: train flags 1.001% (calibration check ✓), val period flags 12.31%
  (real temporal drift — later period deviates more from learned normality)

## 9. Risk classification (transparent rule layer, NOT a trained classifier)

| Level | Rule |
|---|---|
| LOW | score < p95(train) — consistent with learned normal behavior |
| MEDIUM | p95 ≤ score < threshold (elevated), OR score ≥ threshold without high-risk evidence |
| HIGH | score ≥ threshold AND (high-risk privileged events in window OR ≥5 authz failures OR score ≥ p99.9) |

Every detection stores its rule trace (`reasons`) which the drawer displays.
Measured distribution: **LOW 69,940 / MEDIUM 4,743 / HIGH 2,234**.
Because no labels exist, **no classifier metrics are fabricated**; evaluation is
train/val reconstruction loss, score distributions, threshold, and qualitative
case studies (§16 E2E trace).

## 10. MITRE ATT&CK mapping (separate evidence layer)

`config/mitre_mapping.json` — maintainable rules; technique IDs **verified against
attack.mitre.org**: T1562.008, T1098.001, T1098, T1136.003, T1531, T1078.004,
T1078, T1087.004, T1069.003, T1580, T1619, T1496.

The GRU-Autoencoder does **not** predict technique IDs. The mapper inspects the
actual events inside each detected window; rules have `min_count` thresholds and
priority order; if none matches → **"No confident MITRE mapping"** (5,186 val
windows honestly returned no-mapping). Measured: T1580 ×3,900 · T1619 ×2,173 ·
T1078 ×1,472 · T1496 ×1,449 · T1087.004 ×942 …

## 11. Response mechanism

Policy: **LOW → LOG** · **MEDIUM → NOTIFY/ESCALATED** · **HIGH → BLOCK (operator-triggered)**

`ResponseAdapter` interface with two implementations:
- **`SimulatedAdapter` (DEFAULT)** — records the action, flips principal status
  `ACTIVE → BLOCKED` in local SQLite only. Clearly labeled **"SIMULATED RESPONSE"**.
  *No AWS credentials are used; no AWS account is ever disabled.*
- `RealAWSAdapter` — **PROPOSED**, disabled unless `AWS_REAL_BLOCKING_ENABLED=true`;
  raises `NotImplementedError` (slot for future boto3 IAM integration).

Guards: block only for HIGH (400 otherwise), 409 on double-execution, every action
logged in the `responses` table with mode/status/timestamp.

## 12. Database (SQLite)

`backend/data/app.db`:
- `principals` (412): key, display name, identity type, event count, status, first/last seen
- `detections` (76,917): window, score, threshold, risk, reasons JSON, evidence JSON,
  peak event JSON, MITRE fields, response action/status — one row per scored window
- `responses`: executed actions (simulated)
- `meta`: dataset stats, model info, daily event counts, per-split detection stats

## 13. Setup & running

```bash
# 0. dataset — put the two CSVs into the data/ folder (see data/README.md)
#    data/dec12_18features.csv  +  data/nineteenFeaturesDf.csv
#    (from Kaggle: "AWS CloudTrail Dataset from flaws.cloud" — unzip archive.zip into data/)

# 1. backend deps (Python 3.12 venv, CPU-only torch)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r backend/requirements.txt
# (no uv? install it with: pip install uv  — or use: python3 -m venv .venv)

cp .env.example .env        # edit if needed; no secrets required

# 2. pipeline: preprocess → features/sequences → train → inference → detect
.venv/bin/python scripts/preprocess.py
.venv/bin/python scripts/build_features.py
.venv/bin/python scripts/train.py --epochs 8
.venv/bin/python scripts/infer.py --split all
.venv/bin/python scripts/detect.py --split all

# 3. backend server
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001

# 4. frontend
cd frontend && npm install && npm run dev      # http://localhost:5173
```

The Vite dev server proxies `/api` → `http://127.0.0.1:8001` (override with
`VITE_API_TARGET`). Production build: `npm run build` (→ `frontend/dist`).

## 14. API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, model loaded?, DB counts, dataset info, mode |
| `GET /api/dashboard` | KPIs, timeline, risk distribution, recent threats, MITRE dist |
| `GET /api/events` | paginated activity table (`limit, offset, risk, principal_id, q`) |
| `GET /api/events/{id}` | full detail: peak event, reasons, evidence, MITRE rationale |
| `GET /api/threats` | non-LOW detections, sorted by score/time (`risk` filter MEDIUM/HIGH) |
| `GET /api/threats/{id}` | full detail + response history |
| `POST /api/threats/{id}/block` | execute **simulated** response for a HIGH threat |
| `GET /api/users`, `GET /api/users/{id}` | principals + per-user risk/MITRE/threats |

All values come from the database/model artifacts — **nothing is hardcoded**;
input validation returns 422/400/404/409 as appropriate.

## 15. Testing

```bash
.venv/bin/python -m pytest backend/tests -q     # 54 tests
cd frontend && npm run build                    # production build check
```

Coverage: dataset parsing (real CSV), preprocessing (full dedup on real data),
identity resolution, feature encoding, sequence windows, model forward + gradient
flow, masking, artifact loading, threshold calibration sanity, risk rules, MITRE
config/IDs/mapping/no-mapping, response engine (simulated + blocked real adapter),
DB operations, and all API endpoints incl. block flow and validation errors.

## 16. End-to-end validation (mandatory trace)

```bash
.venv/bin/python scripts/e2e_trace.py            # or --threat-id N
```

Traces **one real detection** (id 67981: principal `backup`, 2019-11-17) through:
DB row → window indices → features memmap → re-encoding raw CSV row (bit-exact
match) → **live GRU forward pass** (score `0.16310054` vs stored `0.16310057`,
Δ≈3e-8) → calibrated threshold → risk rules (HIGH, `AuthorizeSecurityGroupIngress`
evidence) → MITRE (T1580, `DescribeSnapshots×17`) → response policy →
FastAPI payloads → dashboard KPIs vs SQL counts. **All stages asserted equal.**

The rendered dashboard was additionally verified in headless Chrome: KPIs
1,225,784 / 412 / 2,532 / 2,234 match the API exactly, zero JS console errors.

## 17. Limitations

- Unsupervised by necessity (no labels): results are anomaly scores + rules, not
  a validated threat classifier; no accuracy/precision/F1/ROC-AUC claimed.
- Reconstruction-error scoring reflects deviation from *learned* behavior, which
  conflates rare-but-legitimate activity with malice — hence the human-triggered
  block (HIGH only) rather than full autonomy.
- Identity imbalance (2 principals ≈ 95%) biases "normal"; period drift (12.3%
  flagged in val vs 1.0% train by construction) is expected and visible.
- Dataset has no object-level S3 events → some ATT&CK techniques can't be evidenced.

## 18. Implemented vs simulated vs proposed

**IMPLEMENTED (runs end-to-end in this repo):**
raw chunked ingestion · dedup/preprocess · feature engineering (138-dim) ·
behavioral windows · **GRU-Autoencoder training + checkpointing** · calibrated
anomaly threshold · inference with per-step errors · risk classification ·
evidence-based MITRE mapping · response engine · **SQLite persistence** ·
**FastAPI** (8 endpoints, validation, errors) · **React SOC dashboard** (KPIs,
timeline, risk/MITRE charts, recent threats, activity table, details drawer,
block flow) · 54 automated tests · E2E trace script.

**SIMULATED:**
AWS blocking (`SimulatedAdapter`): flips local DB status, labeled
"SIMULATED RESPONSE" — *no real AWS account is disabled; no credentials involved.*

**PROPOSED (future work):**
production deployment/scaling · real AWS IAM automation via `RealAWSAdapter`
(revoke access keys, quarantine SCP, forced logout) · streaming ingestion
(EventBridge → SQS) · analyst feedback loop for threshold tuning · multi-account
support.
