"""Feature engineering: vocabulary building (train split only) + row encoding.

Design principles (documented in README):
- Vocabulary/top-K lists are built from the TRAIN split only (temporal split first),
  so encoding never leaks validation-period information.
- High-cardinality fields (eventName, eventSource, errorCode, sourceIPAddress)
  are capped at top-K one-hot + OTHER bucket + frequency features.
- Identity fields (principalKey/arn/accessKeyId/userName) are NEVER model features;
  they are used only for sequence grouping and display.
- Output: float32 matrix in a memmap file, rows in (split, principal, time) order.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import (
    ARTIFACTS_DIR,
    CLEAN_EVENTS_CSV,
    FEATURES_META,
    FEATURES_NPY,
    PREPROCESS_STATS,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    TRAIN_FRACTION,
)
from app.services.preprocessing import load_clean_events

logger = logging.getLogger(__name__)

# top-K configuration (recorded in artifacts)
TOP_K = {
    "eventName": 40,
    "eventSourceFamily": 25,
    "errorCode": 8,
    "sourceIPAddress": 12,
}

IDENTITY_TYPES = ["IAMUser", "AWSService", "AssumedRole", "Root", "AWSAccount", ""]
EVENT_TYPES = ["AwsApiCall", "AwsServiceEvent", "AwsConsoleSignIn", "AwsConsoleAction"]

# eventName semantic classes (domain knowledge, not labels)
ADMIN_EVENTS = {
    "CreateUser", "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AttachUserPolicy", "PutUserPolicy", "AddUserToGroup", "AttachRolePolicy",
    "PutRolePolicy", "CreateRole", "CreatePolicy", "DeleteUser", "DeleteAccessKey",
    "StopLogging", "StartLogging", "DeleteTrail", "UpdateTrail", "PutBucketAcl",
    "PutBucketPolicy", "AuthorizeSecurityGroupIngress", "ModifySnapshotAttribute",
    "ConsoleLogin", "ImportKeyPair", "RegisterImage", "DeleteBucket",
}
READ_PREFIXES = ("Get", "List", "Describe", "Head")
WRITE_PREFIXES = ("Create", "Put", "Delete", "Attach", "Detach", "Run", "Terminate",
                  "Modify", "Update", "Authorize", "Revoke", "Register", "Import",
                  "Add", "Remove", "Start", "Stop", "Reboot", "Enable", "Disable")
DISCOVERY_PREFIXES = ("Describe", "List", "GetAccount", "GetCaller")


def service_family(event_source: str) -> str:
    """'ec2.amazonaws.com' -> 'ec2'."""
    return event_source.split(".")[0] if event_source else ""


def userAgent_class(ua: str) -> str:
    """Map a raw userAgent string to a small fixed client class."""
    if not ua:
        return "other"
    low = ua.lower()
    if "console.amazonaws.com" in low:
        return "console"
    if "awsps" in low or "powershell" in low:
        return "powershell"
    if "aws-cli" in low or "awscli" in low:
        return "awscli"
    if "boto3" in low or "botocore" in low:
        return "boto3"
    if "aws-sdk-java" in low:
        return "javasdk"
    if low in ("ec2.amazonaws.com", "lambda.amazonaws.com", "config.amazonaws.com",
               "cloudtrail.amazonaws.com", "autscaling.amazonaws.com"):
        return "service"
    if low.endswith(".amazonaws.com"):
        return "service"
    return "other"


UA_CLASSES = ["console", "powershell", "awscli", "boto3", "javasdk", "service", "other"]


def is_ip_address(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


ACCELERATOR_TOKENS = ("p2.", "p3.", "p4.", "g3.", "g4", "f1.", "inf1", "trn1")


def temporal_features(epoch: int) -> list[float]:
    """hour sin/cos, day-of-week sin/cos, is_weekend, is_night (local UTC)."""
    t = pd.to_datetime(epoch, unit="s", utc=True)
    hour = t.hour
    dow = t.dayofweek
    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
        1.0 if hour < 6 or hour >= 22 else 0.0,
    ]


def build_vocabulary(
    train_df: pd.DataFrame,
    train_event_freq: Counter,
    train_ip_freq: Counter,
) -> dict:
    """Build encodable vocabulary from TRAIN-split data only."""
    vocab = {
        "identity_types": [t for t in IDENTITY_TYPES],
        "event_types": list(EVENT_TYPES),
        "regions": sorted(train_df["awsRegion"].unique().tolist()),
        "event_families": sorted(train_df["eventSource"].map(service_family).unique().tolist()),
        "top_event_names": [
            n for n, _ in Counter(train_df["eventName"]).most_common(TOP_K["eventName"])
        ],
        "top_event_families": [
            f for f, _ in Counter(train_df["eventSource"].map(service_family)).most_common(
                TOP_K["eventSourceFamily"]
            )
        ],
        "top_error_codes": [
            c for c, _ in Counter(train_df.loc[train_df["errorCode"] != "", "errorCode"])
            .most_common(TOP_K["errorCode"])
        ],
        "top_ips": [
            ip for ip, _ in train_ip_freq.most_common(TOP_K["sourceIPAddress"])
        ],
        "ua_classes": list(UA_CLASSES),
        "event_name_freq": {
            k: float(math.log1p(v)) for k, v in train_event_freq.most_common()
        },
        "ip_freq": {
            k: float(math.log1p(v)) for k, v in train_ip_freq.most_common()
        },
        "top_k": dict(TOP_K),
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
    }
    return vocab


def feature_names(vocab: dict) -> list[str]:
    """Deterministic feature layout (must match encode_frame exactly)."""
    names = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_night",
    ]
    names += [f"idtype={t or 'EMPTY'}" for t in vocab["identity_types"]]
    names += [f"eventtype={t}" for t in vocab["event_types"]]
    names += [f"region={r}" for r in vocab["regions"]]
    names += [f"family={f}" for f in vocab["top_event_families"]] + ["family=OTHER"]
    names += [f"event={n}" for n in vocab["top_event_names"]] + ["event=OTHER"]
    names += ["event_logfreq"]
    names += ["is_read", "is_write", "is_discovery", "is_admin"]
    names += ["has_error"] + [f"err={c}" for c in vocab["top_error_codes"]] + ["err=OTHER"]
    names += ["ip_is_service"] + [f"ip={ip}" for ip in vocab["top_ips"]] + ["ip=OTHER"]
    names += [f"ua={c}" for c in vocab["ua_classes"]]
    names += ["has_instance_type", "is_accelerator_instance"]
    return names


def encode_frame(df: pd.DataFrame, vocab: dict) -> np.ndarray:
    """Encode a clean-events frame to float32 features using a fixed vocabulary."""
    n = len(df)
    cols: list[np.ndarray] = []

    # temporal (6)
    temporal = np.array(
        [temporal_features(int(e)) for e in df["epoch"]], dtype=np.float32
    ).reshape(n, 6)
    cols.append(temporal)

    # one-hots
    def onehot(values: pd.Series, universe: list[str], prefix: str) -> np.ndarray:
        idx = {v: i for i, v in enumerate(universe)}
        out = np.zeros((n, len(universe)), dtype=np.float32)
        for i, v in enumerate(values):
            j = idx.get(v)
            if j is not None:
                out[i, j] = 1.0
        return out

    cols.append(onehot(df["userIdentitytype"], vocab["identity_types"], "idtype"))
    cols.append(onehot(df["eventType"], vocab["event_types"], "eventtype"))
    cols.append(onehot(df["awsRegion"], vocab["regions"], "region"))

    families = df["eventSource"].map(service_family)
    fam_oh = onehot(families, vocab["top_event_families"], "family")
    fam_other = (
        (~families.isin(vocab["top_event_families"])).astype(np.float32).to_numpy()
    ).reshape(n, 1)
    cols.append(np.hstack([fam_oh, fam_other.reshape(n, 1)]))

    ev_oh = onehot(df["eventName"], vocab["top_event_names"], "event")
    ev_other = (
        (~df["eventName"].isin(vocab["top_event_names"])).astype(np.float32).to_numpy()
    ).reshape(n, 1)
    cols.append(np.hstack([ev_oh, ev_other.reshape(n, 1)]))

    ev_freq = df["eventName"].map(vocab["event_name_freq"]).fillna(0.0)
    cols.append(ev_freq.to_numpy(dtype=np.float32).reshape(n, 1))

    # semantic flags (4)
    is_read = df["eventName"].str.startswith(READ_PREFIXES).astype(np.float32).to_numpy()
    is_write = df["eventName"].str.startswith(WRITE_PREFIXES).astype(np.float32).to_numpy()
    is_disc = df["eventName"].str.startswith(DISCOVERY_PREFIXES).astype(np.float32).to_numpy()
    is_admin = df["eventName"].isin(ADMIN_EVENTS).astype(np.float32).to_numpy()
    cols.append(np.stack([is_read, is_write, is_disc, is_admin], axis=1))

    # error features
    has_err = (df["errorCode"] != "").astype(np.float32).to_numpy().reshape(n, 1)
    err_oh = onehot(df["errorCode"], vocab["top_error_codes"], "err")
    err_other = (
        (df["errorCode"] != "")
        & (~df["errorCode"].isin(vocab["top_error_codes"]))
    ).astype(np.float32).to_numpy().reshape(n, 1)
    cols.append(np.hstack([has_err, err_oh, err_other.reshape(n, 1)]))

    # IP features
    ip_service = (~df["sourceIPAddress"].map(is_ip_address)).astype(np.float32).to_numpy()
    cols.append(ip_service.reshape(n, 1))
    ip_oh = onehot(df["sourceIPAddress"], vocab["top_ips"], "ip")
    ip_other = (
        (~df["sourceIPAddress"].isin(vocab["top_ips"])).astype(np.float32).to_numpy()
    ).reshape(n, 1)
    cols.append(np.hstack([ip_oh, ip_other.reshape(n, 1)]))

    # userAgent classes (7)
    ua = df["userAgent"].map(userAgent_class)
    cols.append(onehot(ua, vocab["ua_classes"], "ua"))

    # instance-type features (2)
    has_inst = (df["requestParametersinstanceType"] != "").astype(np.float32).to_numpy()
    is_accel = df["requestParametersinstanceType"].str.startswith(
        ACCELERATOR_TOKENS
    ).astype(np.float32).to_numpy()
    cols.append(np.stack([has_inst, is_accel], axis=1))

    X = np.hstack(cols).astype(np.float32)
    assert X.shape[1] == len(feature_names(vocab)), (
        f"feature layout mismatch: {X.shape[1]} vs {len(feature_names(vocab))}"
    )
    return X


def run_feature_pipeline() -> dict:
    """Full pipeline: temporal split -> train-only vocab -> sorted feature matrix.

    Writes:
      artifacts/features.npy       (N+1, F) float32 memmap (last row = zero pad row)
      artifacts/features_meta.npz  (epochs, principal codes, split flag, perm, pad idx)
      artifacts/feature_report.json
    """
    if not CLEAN_EVENTS_CSV.exists():
        raise FileNotFoundError("run scripts/preprocess.py first")

    df = load_clean_events(
        columns=[
            "eventID", "eventTime", "epoch", "principal_key", "display_name",
            "eventName", "eventSource", "awsRegion", "userIdentitytype", "eventType",
            "sourceIPAddress", "userAgent", "errorCode", "requestParametersinstanceType",
        ]
    )
    df["epoch"] = df["epoch"].astype(np.int64)
    epochs = df["epoch"].to_numpy()
    n = len(df)
    logger.info("loaded clean events: %d", n)

    # temporal split (leakage-safe): 80th percentile of event time
    split_epoch = int(np.quantile(epochs.astype(np.float64), TRAIN_FRACTION))
    train_mask = epochs <= split_epoch
    val_mask = ~train_mask
    logger.info(
        "temporal split at %s (train=%d, val=%d)",
        pd.to_datetime(split_epoch, unit="s", utc=True),
        int(train_mask.sum()),
        int(val_mask.sum()),
    )

    # frequencies + vocab from TRAIN split only
    train_df = df[train_mask]
    train_event_freq = Counter(train_df["eventName"])
    train_ip_freq = Counter(train_df["sourceIPAddress"])
    vocab = build_vocabulary(train_df, train_event_freq, train_ip_freq)
    vocab["split_epoch"] = split_epoch
    vocab["feature_names"] = feature_names(vocab)
    vocab["feature_dim"] = len(vocab["feature_names"])

    # encode all rows in chunks -> temporary memmap in ORIGINAL order
    F = vocab["feature_dim"]
    tmp_path = ARTIFACTS_DIR / "features_tmp.npy"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(n, F))

    chunk = 100_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        feats[start:end] = encode_frame(df.iloc[start:end], vocab)
    feats.flush()
    logger.info("encoded %d rows x %d features", n, F)

    # sort: split -> principal -> epoch -> original order (stable)
    principal_codes, principal_list = pd.factorize(df["principal_key"], sort=True)
    orig_idx = np.arange(n, dtype=np.int64)
    split_flag = (~train_mask).astype(np.int8)  # 0=train, 1=val
    order = np.lexsort((orig_idx, epochs, principal_codes.astype(np.int64), split_flag))

    # write final memmap in sorted order (+ appended zero pad row)
    final = np.lib.format.open_memmap(FEATURES_NPY, mode="w+", dtype=np.float32, shape=(n + 1, F))
    B = 50_000
    for start in range(0, n, B):
        end = min(start + B, n)
        final[start:end] = feats[order[start:end]]
    final[n] = 0.0  # dedicated zero row used for padding short sequences
    final.flush()
    del feats, final
    tmp_path.unlink(missing_ok=True)

    sorted_split = split_flag[order]
    sorted_epochs = epochs[order]
    sorted_principal = principal_codes[order].astype(np.int16)

    np.savez_compressed(
        FEATURES_META,
        principal_list=np.array(principal_list, dtype=object),
        principal_codes=sorted_principal,
        epochs=sorted_epochs,
        split=sorted_split,
        orig_index_of_sorted=order.astype(np.int64),
        pad_row_index=np.int64(n),
        split_epoch=np.int64(split_epoch),
        feature_dim=np.int64(F),
        n_rows=np.int64(n),
    )

    report = {
        "rows": n,
        "feature_dim": F,
        "split_epoch": split_epoch,
        "split_epoch_human": str(pd.to_datetime(split_epoch, unit="s", utc=True)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "principals": int(len(principal_list)),
        "vocab_source": "train split only (temporal 80/20)",
        "top_k": TOP_K,
        "feature_names_head": vocab["feature_names"][:20],
    }
    (ARTIFACTS_DIR / "feature_report.json").write_text(json.dumps(report, indent=2))
    # full vocab INCLUDING frequency maps is required to re-encode raw rows
    # (e2e traces, retraining on new data); size is small (~40 KB).
    (ARTIFACTS_DIR / "feature_vocab.json").write_text(
        json.dumps(vocab, indent=2, default=str)
    )
    logger.info("feature pipeline done: %s", report)
    return report


def load_vocab() -> dict:
    path = ARTIFACTS_DIR / "feature_vocab.json"
    if not path.exists():
        raise FileNotFoundError("feature_vocab.json missing - run scripts/build_features.py")
    return json.loads(path.read_text())
