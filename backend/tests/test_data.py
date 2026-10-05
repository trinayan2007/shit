"""Phase 15: dataset parsing + preprocessing tests (real files where possible)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.data_loader import iter_raw_chunks
from app.services.preprocessing import (
    derive_display_name,
    normalize_account_id,
    resolve_principal,
    run_preprocess,
)

ROOT = Path(__file__).resolve().parents[2]


def test_raw_dataset_header_matches_expected_schema():
    """The real CSV must have the documented 19-column schema."""
    chunks = iter_raw_chunks(columns=None, chunksize=5)
    first = next(chunks)
    expected = {
        "eventID", "eventTime", "sourceIPAddress", "userAgent", "eventName",
        "eventSource", "awsRegion", "eventVersion", "userIdentitytype",
        "eventType", "requestID", "userIdentityaccountId",
        "userIdentityprincipalId", "userIdentityarn", "userIdentityaccessKeyId",
        "userIdentityuserName", "errorCode", "errorMessage",
        "requestParametersinstanceType",
    }
    assert set(first.columns) == expected
    assert len(first) == 5
    # ISO timestamp in the real data
    assert first["eventTime"].iloc[0].endswith("Z")


def test_iter_raw_chunks_respects_column_subset():
    chunks = iter_raw_chunks(columns=["eventID", "eventName"], chunksize=10)
    first = next(chunks)
    assert list(first.columns) == ["eventID", "eventName"]


def test_normalize_account_id():
    assert normalize_account_id("811596193553.0") == "811596193553"
    assert normalize_account_id("811596193553") == "811596193553"
    assert normalize_account_id("") == ""
    assert normalize_account_id("ANONYMOUS_PRINCIPAL") == "ANONYMOUS_PRINCIPAL"


def test_resolve_principal_prefers_arn_then_pid_then_account_then_ip():
    arn_row = {"userIdentityarn": "arn:aws:iam::1:user/backup",
               "userIdentityprincipalId": "AIDA1", "userIdentityaccountId": "1.0",
               "sourceIPAddress": "1.2.3.4"}
    assert resolve_principal(arn_row) == "arn:arn:aws:iam::1:user/backup"

    pid_row = {"userIdentityarn": "", "userIdentityprincipalId": "AIDA1",
               "userIdentityaccountId": "1.0", "sourceIPAddress": "1.2.3.4"}
    assert resolve_principal(pid_row) == "pid:AIDA1"

    acct_row = {"userIdentityarn": "", "userIdentityprincipalId": "",
                "userIdentityaccountId": "1.0", "sourceIPAddress": "1.2.3.4"}
    assert resolve_principal(acct_row) == "acct:1"

    ip_row = {"userIdentityarn": "", "userIdentityprincipalId": "",
              "userIdentityaccountId": "", "sourceIPAddress": "1.2.3.4"}
    assert resolve_principal(ip_row) == "ip:1.2.3.4"


def test_derive_display_name_variants():
    assert derive_display_name({"userIdentityuserName": "backup"}, "x") == "backup"
    assert derive_display_name(
        {"userIdentityuserName": "", "userIdentityarn": "arn:aws:iam::1:root",
         "userIdentityaccountId": "811596193553.0"},
        "arn:arn:aws:iam::1:root",
    ) == "root@811596193553"
    assert derive_display_name(
        {"userIdentityuserName": "",
         "userIdentityarn": "arn:aws:iam::811596193553:user/Level6"},
        "arn:arn:aws:iam::811596193553:user/Level6",
    ) == "Level6"
    # hidden username must not leak through
    assert derive_display_name(
        {"userIdentityuserName": "HIDDEN_DUE_TO_SECURITY_REASONS"}, "pid:X"
    ).startswith("pid:")


def test_run_preprocess_on_real_dataset(tmp_path):
    """Full preprocess on the REAL CSV: dedup count must match assessment."""
    out = tmp_path / "clean.csv"
    stats_path = tmp_path / "stats.json"
    stats = run_preprocess(out_path=out, stats_path=stats_path)

    # measured in artifacts/dataset_assessment.md
    assert stats["rows_in"] == 1_939_207
    assert stats["duplicate_event_ids_dropped"] == 713_423
    assert stats["rows_out"] == 1_225_784
    assert stats["invalid_timestamps_dropped"] == 0
    assert stats["time_min"] == "2017-02-12T19:57:06Z"
    assert stats["time_max"] == "2020-10-07T21:03:30Z"
    assert stats_path.exists()

    # output integrity
    df = pd.read_csv(out, usecols=["eventID", "eventTime", "epoch", "principal_key"])
    assert len(df) == 1_225_784
    assert df["eventID"].is_unique, "dedup failed"
    # epoch must be true seconds-since-epoch (1486929426 == 2017-02-12T19:57:06Z)
    assert int(df["epoch"].min()) == 1_486_929_426
    assert int(df["epoch"].max()) == int(df["epoch"].max())  # sanity
    assert df["epoch"].max() > 1_500_000_000, "epoch unit regression"
