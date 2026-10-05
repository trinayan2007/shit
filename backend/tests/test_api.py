"""Phase 15: database operations + API endpoint tests (TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.database.database import get_conn, get_meta, set_meta
from app.database.models import insert_detection, insert_response, upsert_principal


@pytest.fixture()
def seeded_db():
    """Seed a minimal but realistic detection set in the isolated test DB."""
    conn = get_conn()
    conn.execute("DELETE FROM responses")
    conn.execute("DELETE FROM detections")
    conn.execute("DELETE FROM principals")
    conn.commit()

    pid = upsert_principal(
        principal_key="arn:arn:aws:iam::1:user/backup",
        display_name="backup",
        identity_type="IAMUser",
        event_count=915834,
        first_seen="2017-02-12T19:57:06Z",
        last_seen="2020-10-07T21:03:30Z",
    )
    pid2 = upsert_principal(
        principal_key="arn:arn:aws:iam::1:user/Level6",
        display_name="Level6",
        identity_type="IAMUser",
        event_count=905082,
        first_seen="2017-03-01T00:00:00Z",
        last_seen="2020-10-01T00:00:00Z",
    )

    def mk(idx, score, risk, action, status, mitre_id="T1580", split="val"):
        return insert_detection({
            "window_index": idx,
            "split": split,
            "principal_id": pid,
            "window_start": "2020-06-01T10:00:00Z",
            "window_end": "2020-06-01T10:05:00Z",
            "anomaly_score": score,
            "threshold": 0.0555,
            "risk_level": risk,
            "reasons": ["score >= calibrated threshold"],
            "evidence_summary": {"n_events": 32, "auth_errors": 0},
            "top_events": ["RunInstances x20", "DescribeSnapshots x5"],
            "mitre_status": "mapped" if mitre_id else "no_mapping",
            "mitre_technique_id": mitre_id,
            "mitre_technique_name": "Cloud Infrastructure Discovery" if mitre_id else None,
            "mitre_tactic": "Discovery" if mitre_id else None,
            "mitre_rationale": "Heavy resource enumeration in window." if mitre_id else "No evidence.",
            "mitre_support": ["DescribeInstances x12"],
            "peak_event": {
                "eventTime": "2020-06-01T10:04:59Z",
                "eventName": "DescribeInstances",
                "eventSource": "ec2.amazonaws.com",
                "sourceIPAddress": "5.205.62.253",
                "awsRegion": "us-west-2",
                "errorCode": "",
                "peak_step_error": 0.12,
            },
            "response_action": action,
            "response_status": status,
        })

    d_low = mk(1, 0.01, "LOW", "LOG", "LOGGED", mitre_id=None, split="train")
    d_med = mk(2, 0.07, "MEDIUM", "NOTIFY", "ESCALATED", mitre_id="T1496")
    d_high = mk(3, 0.42, "HIGH", "BLOCK", "PENDING", mitre_id="T1562.008")
    set_meta("dataset", {"rows_out": 1225784, "rows_in": 1939207})
    set_meta("model", {"threshold": 0.0555, "threshold_method": "p99 of train errors"})
    return {"pid": pid, "pid2": pid2, "low": d_low, "med": d_med, "high": d_high}


@pytest.fixture()
def client(seeded_db):
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------- DB ----------------

def test_upsert_principal_idempotent():
    a = upsert_principal("pid:X", "X", "IAMUser", 10, "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z")
    b = upsert_principal("pid:X", "X", "IAMUser", 20, "2019-01-01T00:00:00Z", "2021-01-01T00:00:00Z")
    assert a == b, "upsert must not duplicate principals"
    row = get_conn().execute("SELECT * FROM principals WHERE id=?", (a,)).fetchone()
    assert row["event_count"] == 20
    assert row["first_seen"] == "2019-01-01T00:00:00Z"
    assert row["last_seen"] == "2021-01-01T00:00:00Z"


def test_insert_detection_roundtrip_json_fields(seeded_db):
    row = get_conn().execute(
        "SELECT * FROM detections WHERE id=?", (seeded_db["high"],)
    ).fetchone()
    import json
    assert row["risk_level"] == "HIGH"
    assert json.loads(row["reasons"])
    peak = json.loads(row["peak_event"])
    assert peak["eventName"] == "DescribeInstances"


def test_meta_roundtrip():
    set_meta("unit_test_key", {"a": 1})
    assert get_meta("unit_test_key") == {"a": 1}
    set_meta("unit_test_str", "plain")
    assert get_meta("unit_test_str") == "plain"
    assert get_meta("missing_key", "fallback") == "fallback"


def test_insert_response(seeded_db):
    rid = insert_response(seeded_db["high"], "BLOCK", "SIMULATED", "EXECUTED", "test detail")
    rows = get_conn().execute(
        "SELECT * FROM responses WHERE detection_id=?", (seeded_db["high"],)
    ).fetchall()
    assert any(r["id"] == rid for r in rows)
    assert rows[0]["mode"] == "SIMULATED"


# ---------------- API ----------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "online"
    assert d["backend_status"] == "ONLINE"
    assert d["model_status"] in ("LOADED", "NOT_LOADED")
    assert d["mode"].startswith("SIMULATED")
    assert d["database"]["detections"] >= 3


def test_dashboard_values_come_from_db(client):
    d = client.get("/api/dashboard").json()
    assert d["kpis"]["total_events"] == 1225784  # from meta seeded in fixture
    assert d["kpis"]["unique_principals"] >= 2
    assert d["risk_distribution"] == {"LOW": 1, "MEDIUM": 1, "HIGH": 1}
    assert d["kpis"]["high_risk_incidents"] == 1
    assert len(d["recent_threats"]) == 2  # non-LOW only
    assert all(t["risk_level"] != "LOW" for t in d["recent_threats"])
    assert d["timeline"], "timeline must contain real buckets"
    assert d["mitre_distribution"]


def test_events_pagination_and_filters(client):
    r = client.get("/api/events?limit=2")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 3
    assert len(d["items"]) == 2

    d2 = client.get("/api/events?risk=HIGH").json()
    assert d2["total"] == 1
    assert d2["items"][0]["risk_level"] == "HIGH"

    d3 = client.get("/api/events?q=backup").json()
    assert d3["total"] == 3
    d4 = client.get("/api/events?q=doesnotexist").json()
    assert d4["total"] == 0


def test_event_detail_includes_evidence_and_mitre(client):
    d = client.get(f"/api/events/{client.get('/api/events?risk=HIGH').json()['items'][0]['id']}").json()
    assert d["peak_event"]["eventName"] == "DescribeInstances"
    assert d["reasons"]
    assert d["mitre_rationale"]
    assert d["evidence_summary"]["n_events"] == 32


def test_event_404(client):
    assert client.get("/api/events/999999").status_code == 404


def test_threats_excludes_low_and_sorts_by_score(client):
    d = client.get("/api/threats").json()
    assert d["total"] == 2
    scores = [t["anomaly_score"] for t in d["items"]]
    assert scores == sorted(scores, reverse=True)
    # LOW is not a threat: the threats API rejects it as a filter value
    assert client.get("/api/threats?risk=LOW").status_code == 422


def test_threat_detail_structure(client):
    high_id = client.get("/api/threats?risk=HIGH").json()["items"][0]["id"]
    d = client.get(f"/api/threats/{high_id}").json()
    assert d["mitre"]["technique_id"] == "T1562.008"
    assert d["mitre"]["rationale"]
    assert d["responses"] == []
    assert d["principal"]["display_name"] == "backup"


def test_block_flow_simulated(client):
    high_id = client.get("/api/threats?risk=HIGH").json()["items"][0]["id"]
    r = client.post(f"/api/threats/{high_id}/block")
    assert r.status_code == 200
    d = r.json()
    assert d["simulated"] is True
    assert d["label"] == "SIMULATED RESPONSE"
    assert d["status"] == "EXECUTED"
    assert d["principal_status"] == "BLOCKED"
    assert "No AWS credentials" in d["detail"]

    # persisted
    det = client.get(f"/api/threats/{high_id}").json()
    assert det["response_status"] == "EXECUTED"
    assert len(det["responses"]) == 1
    assert det["principal"]["status"] == "BLOCKED"

    # idempotency guard
    assert client.post(f"/api/threats/{high_id}/block").status_code == 409


def test_block_rejected_for_non_high(client):
    med_id = client.get("/api/threats?risk=MEDIUM").json()["items"][0]["id"]
    assert client.post(f"/api/threats/{med_id}/block").status_code == 400
    low_id = client.get("/api/events?risk=LOW").json()["items"][0]["id"]
    assert client.post(f"/api/threats/{low_id}/block").status_code == 400
    assert client.post("/api/threats/999999/block").status_code == 404


def test_users_list_and_detail(client):
    d = client.get("/api/users?limit=10").json()
    assert d["total"] == 2
    names = {i["display_name"] for i in d["items"]}
    assert names == {"backup", "Level6"}
    # sorted by event_count desc
    assert d["items"][0]["event_count"] >= d["items"][1]["event_count"]

    uid = d["items"][0]["id"]
    detail = client.get(f"/api/users/{uid}").json()
    assert detail["display_name"] == "backup"
    assert detail["risk_distribution"]["HIGH"] == 1
    assert detail["top_threats"]
    assert client.get("/api/users/999999").status_code == 404


def test_input_validation(client):
    assert client.get("/api/events?risk=INVALID").status_code == 422
    assert client.get("/api/events?limit=0").status_code == 422
    assert client.get("/api/events?limit=99999").status_code == 422
