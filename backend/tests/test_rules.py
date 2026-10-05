"""Phase 15: risk classification + MITRE mapping + response engine tests."""

from __future__ import annotations

import json

import pytest

from app.config import MITRE_MAPPING_FILE
from app.services.classification import (
    HIGH,
    LOW,
    MEDIUM,
    Evidence,
    Thresholds,
    classify_risk,
)
from app.services.mitre_mapping import NO_MAPPING, MitreMapper
from app.services.response_engine import (
    RealAWSAdapter,
    ResponseResult,
    SimulatedAdapter,
    automatic_action_for,
    get_adapter,
)

TH = Thresholds(threshold=0.05, p95=0.03, p999=0.20)


def _ev(names, errs=None):
    return Evidence.from_window(names, errs or [""] * len(names))


# ---------------- risk classification ----------------

def test_low_when_score_below_p95():
    d = classify_risk(0.01, _ev(["RunInstances"] * 32), TH)
    assert d.risk_level == LOW
    assert any("below p95" in r for r in d.reasons)


def test_medium_between_p95_and_threshold():
    d = classify_risk(0.04, _ev(["RunInstances"] * 32), TH)
    assert d.risk_level == MEDIUM
    assert any("elevated" in r for r in d.reasons)


def test_medium_above_threshold_without_high_risk_evidence():
    d = classify_risk(0.08, _ev(["DescribeInstances"] * 32), TH)
    assert d.risk_level == MEDIUM
    assert any("threshold" in r for r in d.reasons)


def test_high_above_threshold_with_high_risk_evidence():
    names = ["DescribeInstances"] * 30 + ["StopLogging", "DeleteTrail"]
    d = classify_risk(0.08, _ev(names), TH)
    assert d.risk_level == HIGH
    assert any("high-risk privileged events" in r for r in d.reasons)
    assert "StopLogging" in d.evidence_summary["high_risk_events"]


def test_high_above_threshold_with_auth_error_burst():
    names = ["RunInstances"] * 32
    errs = ["Client.UnauthorizedOperation"] * 8
    d = classify_risk(0.08, _ev(names, errs), TH)
    assert d.risk_level == HIGH
    assert any("authorization-failure" in r for r in d.reasons)


def test_high_on_extreme_score_even_without_evidence():
    d = classify_risk(0.25, _ev(["RunInstances"] * 32), TH)
    assert d.risk_level == HIGH
    assert any("extreme tail" in r for r in d.reasons)


def test_thresholds_load_from_real_calibration(tmp_path):
    p = tmp_path / "th.json"
    p.write_text(json.dumps({
        "threshold": 0.0555,
        "quantile": 0.99,
        "train_score_stats": {"p95": 0.039, "p99": 0.0555, "max": 0.23},
    }))
    th = Thresholds.load(p)
    assert th.threshold == 0.0555
    assert th.p95 == 0.039
    assert th.p999 == 0.23


def test_evidence_extraction():
    ev = Evidence.from_window(
        ["CreateUser", "RunInstances", "StopLogging"],
        ["", "Client.RequestLimitExceeded", ""],
    )
    assert "CreateUser" in ev.sensitive_events
    assert "StopLogging" in ev.high_risk_events
    assert ev.auth_errors == 0
    ev2 = Evidence.from_window(["RunInstances"], ["AccessDenied"])
    assert ev2.auth_errors == 1 and ev2.error_events == 1


# ---------------- MITRE mapping ----------------

def test_mitre_config_has_verified_ids():
    cfg = json.loads(MITRE_MAPPING_FILE.read_text())
    ids = {r["technique_id"] for r in cfg["rules"]}
    # every ID used must be a real ATT&CK technique (verified on attack.mitre.org)
    assert {
        "T1562.008", "T1098.001", "T1098", "T1136.003", "T1531",
        "T1078.004", "T1087.004", "T1069.003", "T1580", "T1619",
        "T1496", "T1078",
    } <= ids


def test_mitre_maps_cloudtrail_stoplogging_to_t1562_008():
    m = MitreMapper()
    r = m.map_window(["StopLogging", "DescribeTrails", "GetTrailStatus"])
    assert r.status == "mapped"
    assert r.technique_id == "T1562.008"
    assert "StopLogging" in " ".join(r.supporting_behavior)
    assert r.rationale  # mapping rationale present


def test_mitre_no_confident_mapping_when_evidence_insufficient():
    m = MitreMapper()
    r = m.map_window(["GetCallerIdentity"])
    assert r.status == "no_mapping"
    assert r.technique_name is None
    assert "insufficient evidence" in (r.rationale or "")


def test_mitre_respects_min_count_thresholds():
    m = MitreMapper()
    # 1 Describe call is below the T1580 min_count=10
    assert m.map_window(["DescribeInstances"]).status == "no_mapping"
    # 10+ Describe calls satisfy T1580
    r = m.map_window(["DescribeInstances"] * 12)
    assert r.technique_id == "T1580"


def test_mitre_priority_prefers_specific_over_general():
    m = MitreMapper()
    # StopLogging (priority 5) should beat generic discovery rules
    r = m.map_window(["StopLogging"] + ["DescribeInstances"] * 20)
    assert r.technique_id == "T1562.008"


def test_no_mapping_is_the_documented_string():
    m = MitreMapper()
    r = m.map_window([])
    assert r.technique_id is None
    assert NO_MAPPING == "No confident MITRE mapping"


# ---------------- response engine ----------------

def test_simulated_adapter_never_touches_aws():
    res = SimulatedAdapter().execute_block("arn:aws:iam::1:user/x", {})
    assert isinstance(res, ResponseResult)
    assert res.mode == "SIMULATED"
    assert res.status == "EXECUTED"
    assert res.action == "BLOCK"
    assert "No AWS credentials" in res.detail
    assert "no real AWS account" in res.detail


def test_real_adapter_blocked_by_default():
    with pytest.raises(RuntimeError):
        RealAWSAdapter().execute_block("x", {})  # env flag false in tests


def test_default_adapter_is_simulated():
    assert isinstance(get_adapter(), SimulatedAdapter)


def test_automatic_action_policy():
    assert automatic_action_for("LOW") == ("LOG", "LOGGED")
    assert automatic_action_for("MEDIUM") == ("NOTIFY", "ESCALATED")
    assert automatic_action_for("HIGH") == ("BLOCK", "PENDING")
