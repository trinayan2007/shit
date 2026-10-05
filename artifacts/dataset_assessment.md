# DATASET + PPT ASSESSMENT

All numbers below are **measured** from the actual files (via `scripts/profile_data.py`,
`scripts/verify_data.py`, `scripts/cardinality.py`), not assumed.

## 1. Dataset files

| File | Size (bytes) | Size |
|---|---|---|
| `data/dec12_18features.csv` | 830,904,812 | ~793 MiB |
| `data/nineteenFeaturesDf.csv` | 1,042,474,776 | ~994 MiB |

Both extracted from `~/Downloads/archive.zip`.

## 2. Row counts

Both files: **1,939,215 lines** by `wc -l` = **1,939,207 data rows** + 1 header + 7 embedded newlines
inside quoted `errorMessage` fields (verified with pandas chunked read: exactly 1,939,207 rows in each).

## 3. Schemas (actual headers)

`dec12_18features.csv` (18 columns):
eventID, eventTime, sourceIPAddress, userAgent, eventName, eventSource, awsRegion,
eventVersion, userIdentitytype, eventType, userIdentityaccountId, userIdentityprincipalId,
userIdentityarn, userIdentityaccessKeyId, userIdentityuserName, errorCode, errorMessage,
requestParametersinstanceType

`nineteenFeaturesDf.csv` (19 columns): identical set **plus `requestID`** inserted after `eventType`.

## 4. Comparison of the two CSVs — verdict: DIFFERENT PREPROCESSING STAGES OF THE SAME EVENTS

- Row counts identical: 1,939,207 = 1,939,207.
- `eventID` overlap: **shared unique IDs = 1,225,784; only_dec12 = 0; only_nineteen = 0**.
  The unique-ID sets are identical (both files contain the same 713,423 duplicate-ID rows too).
- `dec12_18features.csv`: `eventTime` is **month-granularity** (e.g. `2017-02`, `2019-08`);
  all 1,939,207 values fail ISO-8601 validation (0 valid, 1,939,207 invalid). 35 distinct months, range
  `2017-02 .. 2020-10`.
- `nineteenFeaturesDf.csv`: `eventTime` is **full ISO-8601 UTC** (`2017-02-12T19:57:06Z`);
  **1,939,207 / 1,939,207 valid**. Range `2017-02-12T19:57:06Z .. 2020-10-07T21:03:30Z`.
- `dec12` fills missing values with sentinel strings (`Unknown`, `None`, `NoError`, `NotApplicable`);
  `nineteen` keeps true empties (`""`).
- **Conclusion: `nineteenFeaturesDf.csv` is the same event set with better fidelity**
  (full timestamps, real nulls, extra `requestID`). **Selected for the project.**
  Using `dec12` would make chronological sequence construction impossible (month-only timestamps).

## 5. Timestamp structure

- Column `eventTime`, ISO-8601 `YYYY-MM-DDTHH:MM:SSZ` in the selected file; 0 empty; 100% valid.
- Coverage: 2017-02-12 → 2020-10-07 (~3.7 years).
- Not globally sorted (files are roughly time-ordered in blocks, but ordering must be enforced per identity).

## 6. Identity / user structure (measured, nineteenFeaturesDf)

- `userIdentitytype`: IAMUser 1,825,650 · AWSService 57,616 · AssumedRole 42,315 · Root 10,997 ·
  AWSAccount 2,628 · empty 1.
- `userIdentityuserName`: **8 distinct** — backup 915,834 · Level6 905,082 · empty 110,185 ·
  SecurityMokey 4,522 · flaws 3,372 · piper 147 · Level5 39 · HIDDEN_DUE_TO_SECURITY_REASONS 26.
- `userIdentityarn`: 58 distinct (includes assumed-role sessions, root, empty).
- `userIdentityprincipalId`: 114 distinct (AIDA…, AROA…:session, bare account IDs, empty).
- `userIdentityaccountId`: 313 distinct (mostly `811596193553` / `811596193553.0` — needs normalization).
- Empty `userIdentityarn` 60,275 (3.11%), empty `principalId` 59,371 (3.06%), empty `userName` 110,185 (5.68%).

**Identity grouping decision (documented):** build `principal_key` with fallback:
`userIdentityarn` → else `userIdentityprincipalId` → else `userIdentityaccountId` (normalized, strip `.0`)
→ else `sourceIPAddress`. `userName` alone is NOT used (empty for root/assumed-role/service events).
A display name is derived separately (userName if present, else last ARN segment, else type+key fragment).

## 7. AWS services / event sources

- `eventSource`: 164 distinct. Top: ec2 1,543,250 · sts 96,796 · s3 87,991 · iam 65,991 ·
  rds 17,827 · lambda 12,265 · apigateway 8,680 · cloudtrail 7,993 · logs 7,901 · redshift 6,740.
- `eventName`: 1,242 distinct. Top: RunInstances 1,323,105 · DescribeSnapshots 102,510 ·
  AssumeRole 79,322 · GetBucketAcl 42,651 · DescribeInstances 18,935 · GetCallerIdentity 17,128.
- `awsRegion`: 17 distinct (all standard regions; us-west-2 top 328,933).
- `eventType`: AwsApiCall 1,938,504 · AwsServiceEvent 603 · AwsConsoleSignIn 99 · AwsConsoleAction 1.
- Sensitive events present (for MITRE evidence rules): CreateUser 43 · CreateAccessKey 53 ·
  AttachUserPolicy 32 · AddUserToGroup 3 · ConsoleLogin 81 · StopLogging 2 · DeleteTrail 1 ·
  UpdateTrail 1 · PutBucketAcl 93 · PutBucketPolicy 79 · ModifySnapshotAttribute 56 ·
  AuthorizeSecurityGroupIngress 14 · DeleteUser 6 · DeleteAccessKey 22 · DeleteBucket 100.
  Object-level S3 (`GetObject`/`PutObject`) = **0 occurrences** — no data-object events in this dataset.

## 8. Error fields

- `errorCode`: 104 distinct, empty 431,493 (22.25%). Top non-empty: Client.RequestLimitExceeded
  779,330 · Client.UnauthorizedOperation 351,760 · AccessDenied 120,988 · Client.Unsupported 101,226 ·
  Server.InsufficientInstanceCapacity 59,323 · Client.InstanceLimitExceeded 43,272 · NoSuchBucket 29,170.
- `errorMessage`: empty 439,435 (22.66%).
- `requestParametersinstanceType`: empty 616,170 (31.77%) — populated only on RunInstances-type calls.

## 9. Labels

**There is NO ground-truth insider-threat label column in either file.** Columns are the raw 18/19
CloudTrail-derived features only. `errorCode` is an AWS API error code, not a threat label.

- Supervised malicious/benign classification with accuracy/precision/F1: **NOT SUPPORTED**.
- Unsupervised behavioral anomaly detection: **SUPPORTED** (chronological, identity-grouped event streams).
- Therefore the core ML problem is framed as **anomaly detection + a rule-based risk layer**, exactly as
  the prompt requires. No ML classification metrics will be fabricated.

## 10. Missing values (nineteenFeaturesDf)

requestParametersinstanceType 31.77% · errorMessage 22.66% · errorCode 22.25% ·
userIdentityuserName 5.68% · accessKeyId 3.29% · arn 3.11% · principalId 3.06% · accountId 2.97% ·
requestID 0.04% · userAgent 0.00005% (1) · userIdentitytype 1 · all others 0.

## 11. Duplicates

eventID unique = 1,225,784; duplicate rows by eventID = **713,423** (up to 8 repeats of one ID —
identical in both files). Preprocessing **deduplicates on `eventID` (keep first)** → 1,225,784 unique
events used downstream; dedup count is recorded in artifacts.

## 12. Recommended feature engineering

Per-event numeric vector (all from actual columns):
temporal (hour sin/cos, day-of-week sin/cos, is_weekend, is_night) · `userIdentitytype` one-hot ·
`eventType` one-hot · `awsRegion` one-hot · `eventSource` grouped to service-family + top-K one-hot ·
`eventName` top-K one-hot + other + log-frequency + read/write/administration class flags ·
`errorCode` presence + top-K one-hot + category flags (unauthorized/throttle/notfound) ·
`sourceIPAddress` top-K one-hot + service-IP flag + log-frequency · `userAgent` client-class flags ·
`requestParametersinstanceType` presence + accelerator-instance flag.
NOT used as features: accessKeyId, ARN, principalId, userName (identifiers/credentials → leakage, and
explicitly forbidden). High-cardinality fields capped via top-K + frequency encoding (no blind one-hot).

## 13. Recommended sequence construction

Group by `principal_key` → stable sort by (`eventTime`, `eventID`) → sliding windows, **length 32,
stride 16** → short sequences (<32) padded with zeros + mask (padded steps excluded from loss/score) →
chronological **temporal split at the 80th-percentile eventTime** (train on earlier period, validate on
later period; windows built within each split — never shuffled before windowing).

## 14. Recommended model input

Tensor `(batch, 32, F)` float32 with F ≈ 130 engineered features; target = the input itself
(autoencoder). Anomaly score = mean per-step squared reconstruction error over unmasked steps.

## 15. Dataset limitations (documented in README)

1. No insider-threat labels → no supervised metrics; risk layer is transparent rules, not a trained classifier.
2. This is the public **flaws.cloud** AWS account dataset (account 811596193553, users backup/Level6/flaws/
   piper) — a CTF/red-team corpus with attack-like bursts, not a labeled insider corpus; "normal" is
   assumed to be the majority behavior (standard unsupervised assumption).
3. Two dominant identities (backup, Level6) hold 94.7% of rows; heavy class imbalance by identity.
4. RunInstances = 68% of all events; bursts of Client.RequestLimitExceeded dominate error stats.
5. No S3 object-level events (GetObject/PutObject = 0) → exfiltration MITRE mappings cannot be evidenced.
6. `dec12_18features.csv` unusable for sequencing (month-granularity timestamps) — selected file is `nineteen`.

## 16. PPT methodology (extracted from OCR of all 13 slides — `artifacts/ppt_text/slide*.txt`)

Pipeline: CloudTrail → preprocessing → feature engineering → **HYBRID (GRU-Autoencoder)** → threat
classification → MITRE ATT&CK mapping → risk assessment → (Low: alert / Medium: notify admin /
High: automated blocking) → interactive dashboard. Hybrid roles: Autoencoder = representation learning
via reconstruction error; GRU = sequential/temporal pattern learning. Threat levels Low/Medium/High.

## 17. Conflicts PPT vs dataset (and resolutions)

1. PPT implies classified threats with the hybrid model; dataset has **no labels** → implemented as
   unsupervised GRU-Autoencoder anomaly scoring + documented rule-based risk layer (LOW/MEDIUM/HIGH).
   No fabricated classifier accuracy anywhere.
2. PPT's automated blocking → implemented as **clearly labeled SIMULATED** response engine behind an
   adapter interface (real AWS IAM adapter is a proposal, disabled by default).
3. MITRE ATT&CK mapping cannot be "predicted by the model" → implemented as a separate **evidence-based
   mapping layer** over detected window contents (actual eventNames/errorCodes present in the window);
   insufficient evidence → "No confident MITRE mapping".
4. PPT does not specify sequence mechanics → filled with the documented, leakage-safe choices in §13.
