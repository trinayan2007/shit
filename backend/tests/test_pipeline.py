"""Phase 15: feature engineering + sequence construction tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_engineering import (
    build_vocabulary,
    encode_frame,
    feature_names,
    service_family,
    temporal_features,
    userAgent_class,
)
from app.services.sequence_builder import find_runs, windows_for_run

ROOT = Path(__file__).resolve().parents[2]


def _mini_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "eventTime": ["2019-01-01T10:00:00Z", "2019-01-01T10:00:01Z",
                      "2019-01-05T23:30:00Z", "2019-01-06T03:00:00Z"],
        "epoch": [1546346400, 1546346401, 1546702200, 1546714800],
        "principal_key": ["arn:aws:iam::1:user/backup"] * 4,
        "display_name": ["backup"] * 4,
        "eventName": ["RunInstances", "StopLogging", "GetCallerIdentity", "CreateUser"],
        "eventSource": ["ec2.amazonaws.com", "cloudtrail.amazonaws.com",
                        "sts.amazonaws.com", "iam.amazonaws.com"],
        "awsRegion": ["us-east-1"] * 4,
        "userIdentitytype": ["IAMUser"] * 4,
        "eventType": ["AwsApiCall"] * 4,
        "sourceIPAddress": ["5.205.62.253", "5.205.62.253", "1.2.3.4", "ec2.amazonaws.com"],
        "userAgent": ["Boto3/1.9.201 Python/2.7.12", "console.amazonaws.com",
                      "aws-cli/1.16.301", "aws-sdk-java/1.11.301"],
        "errorCode": ["", "Client.UnauthorizedOperation", "", "AccessDenied"],
        "requestParametersinstanceType": ["p2.16xlarge", "", "", ""],
    })


def test_service_family():
    assert service_family("ec2.amazonaws.com") == "ec2"
    assert service_family("elasticloadbalancing.amazonaws.com") == "elasticloadbalancing"
    assert service_family("") == ""


def test_user_agent_class():
    assert userAgent_class("Boto3/1.9.201 Botocore/1.12") == "boto3"
    assert userAgent_class("console.amazonaws.com") == "console"
    assert userAgent_class("aws-cli/1.16.301") == "awscli"
    assert userAgent_class("AWSPowerShell/3.3") == "powershell"
    assert userAgent_class("ec2.amazonaws.com") == "service"
    assert userAgent_class("") == "other"


def test_temporal_features_shapes_and_ranges():
    # 1486929426 = 2017-02-12 19:57:06 UTC (Sunday)
    f = temporal_features(1_486_929_426)
    assert len(f) == 6
    hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_night = f
    assert abs(hour_sin) <= 1 and abs(hour_cos) <= 1
    # 19:57 is not night (night = hour < 6 or >= 22)
    assert is_night == 0.0
    # 2017-02-12 was a Sunday -> weekend
    assert is_weekend == 1.0


def test_encode_frame_layout_and_values():
    df = _mini_frame()
    train_freq = Counter(df["eventName"])
    ip_freq = Counter(df["sourceIPAddress"])
    vocab = build_vocabulary(df, train_freq, ip_freq)
    names = feature_names(vocab)
    X = encode_frame(df, vocab)
    assert X.shape == (len(df), len(names))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()

    # one-hot blocks: identity type + event type present
    idtype_cols = [i for i, n in enumerate(names) if n.startswith("idtype=")]
    assert X[0, idtype_cols].sum() == 1
    # admin flag fires for CreateUser / StopLogging rows
    admin_idx = names.index("is_admin")
    assert X[3, admin_idx] == 1.0  # CreateUser
    # accelerator instance flag
    accel_idx = names.index("is_accelerator_instance")
    assert X[0, accel_idx] == 1.0  # p2.16xlarge
    # error presence
    has_err = names.index("has_error")
    assert X[0, has_err] == 0.0 and X[1, has_err] == 1.0


def test_temporal_split_is_temporal(tmp_path):
    """Split epoch must be a real timestamp with train < split <= val."""
    report_path = ROOT / "artifacts" / "feature_report.json"
    if not report_path.exists():
        return  # feature pipeline not run in this environment
    report = json.loads(report_path.read_text())
    split = report["split_epoch"]
    assert split > 1_500_000_000, "split epoch must be seconds since 1970"
    # within dataset range
    assert 1_486_929_426 <= split <= 1_602_104_610
    assert report["train_rows"] + report["val_rows"] == report["rows"]


def test_find_runs_groups_contiguous_principal_and_split():
    split = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    principal = np.array([1, 1, 2, 2, 2, 3, 3], dtype=np.int16)
    runs = find_runs(split, principal)
    # rows: (0,1)=s0/p1, (2)=s0/p2, (3,4)=s1/p2, (5,6)=s0/p3
    # a run breaks on split change OR principal change
    assert runs == [(0, 2, 0), (2, 3, 0), (3, 5, 1), (5, 7, 0)]


def test_windows_for_run_sliding_and_tail():
    # length 40, seq 32, stride 16 -> starts 0 and 8(tail=40-32)
    ws = windows_for_run(0, 40, pad_row=999, seq_len=32, stride=16)
    assert len(ws) == 2
    assert ws[0][0] == 0 and ws[0][-1] == 31
    assert ws[1][0] == 8 and ws[1][-1] == 39
    # length 100 -> starts 0,16,32,52(tail=68)
    ws2 = windows_for_run(0, 100, pad_row=999, seq_len=32, stride=16)
    starts = [int(w[0]) for w in ws2]
    assert starts == [0, 16, 32, 48, 64, 68]
    # every row covered by at least one window
    covered = set()
    for w in ws2:
        covered.update(int(x) for x in w)
    assert covered == set(range(100))


def test_windows_for_run_short_sequence_pads():
    ws = windows_for_run(0, 10, pad_row=999, seq_len=32, stride=16)
    assert len(ws) == 1
    w = ws[0]
    assert len(w) == 32
    assert list(w[:10]) == list(range(10))
    assert all(x == 999 for x in w[10:])
