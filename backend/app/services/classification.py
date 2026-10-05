"""Risk classification layer (Phase 7).

This is a TRANSPARENT RULE LAYER over model output - not a trained supervised
classifier. The dataset has no ground-truth threat labels, so no classifier
accuracy is claimed anywhere.

Inputs:
  - anomaly_score: actual GRU-Autoencoder reconstruction error (masked MSE)
  - threshold: calibrated from the training error distribution (quantiles)
  - behavioral evidence extracted from the window contents

Rules (documented, reproducible - see README):
  LOW:    score <  p95(train errors)                    -> normal/low deviation
  MEDIUM: p95 <= score < threshold OR (score >= threshold, no high-risk evidence)
  HIGH:   score >= threshold AND high-risk evidence present
          (sensitive/privileged CloudTrail events, authentication anomalies,
           or score >= p99.9 of train errors regardless of evidence)

Every detection stores its rule trace so the dashboard can show *why*.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import ANOMALY_THRESHOLD

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

# evidence sets loaded from the same config the MITRE layer uses
SENSITIVE_EVENTS = {
    "CreateUser", "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AttachUserPolicy", "PutUserPolicy", "AddUserToGroup", "AttachRolePolicy",
    "PutRolePolicy", "CreateRole", "CreatePolicy", "DeleteUser", "DeleteAccessKey",
    "StopLogging", "StartLogging", "DeleteTrail", "UpdateTrail", "PutBucketAcl",
    "PutBucketPolicy", "AuthorizeSecurityGroupIngress", "ModifySnapshotAttribute",
    "ConsoleLogin", "ImportKeyPair", "RegisterImage", "DeleteBucket",
}
# subset that alone justifies HIGH when combined with anomaly >= threshold
HIGH_RISK_EVENTS = {
    "CreateAccessKey", "CreateLoginProfile", "AttachUserPolicy", "PutUserPolicy",
    "AddUserToGroup", "StopLogging", "DeleteTrail", "UpdateTrail", "DeleteUser",
    "DeleteAccessKey", "PutBucketAcl", "PutBucketPolicy",
    "AuthorizeSecurityGroupIngress", "ModifySnapshotAttribute", "ConsoleLogin",
}
AUTH_ERROR_CODES = {
    "Client.UnauthorizedOperation", "AccessDenied", "AccessDeniedException",
    "UnauthorizedOperation", "AuthFailure", "InvalidClientTokenId",
    "SignatureDoesNotMatch",
}


@dataclass
class Thresholds:
    """Calibrated cut points from the TRAINING reconstruction-error distribution."""

    threshold: float          # p99 - the anomaly decision boundary
    p95: float
    p999: float

    @classmethod
    def load(cls, path: Path | None = None) -> "Thresholds":
        path = path or ANOMALY_THRESHOLD
        data = json.loads(path.read_text())
        stats = data["train_score_stats"]
        return cls(
            threshold=float(data["threshold"]),
            p95=float(stats["p95"]),
            # p99.9 from max/p99 interpolation is unreliable; fall back to max
            p999=float(stats.get("p999") or stats["max"]),
        )


@dataclass
class Evidence:
    """Behavioral evidence extracted from one window's CloudTrail events."""

    event_names: list[str]
    error_codes: list[str]
    n_events: int
    sensitive_events: list[str] = field(default_factory=list)
    high_risk_events: list[str] = field(default_factory=list)
    auth_errors: int = 0
    error_events: int = 0

    @classmethod
    def from_window(cls, event_names: list[str], error_codes: list[str]) -> "Evidence":
        sens = sorted({e for e in event_names if e in SENSITIVE_EVENTS})
        high = sorted({e for e in event_names if e in HIGH_RISK_EVENTS})
        auth = sum(1 for c in error_codes if c in AUTH_ERROR_CODES)
        errs = sum(1 for c in error_codes if c)
        return cls(
            event_names=list(event_names),
            error_codes=list(error_codes),
            n_events=len(event_names),
            sensitive_events=sens,
            high_risk_events=high,
            auth_errors=auth,
            error_events=errs,
        )


@dataclass
class RiskDecision:
    risk_level: str
    anomaly_score: float
    threshold: float
    reasons: list[str]
    evidence_summary: dict


def classify_risk(score: float, evidence: Evidence, th: Thresholds) -> RiskDecision:
    """Apply the documented rule ladder. Pure function - fully testable."""
    reasons: list[str] = []
    over_threshold = score >= th.threshold
    elevated = score >= th.p95

    has_high_risk_evidence = bool(evidence.high_risk_events)
    has_sensitive_evidence = bool(evidence.sensitive_events)
    auth_burst = evidence.auth_errors >= 5
    extreme_score = score >= th.p999

    if over_threshold and (has_high_risk_evidence or auth_burst or extreme_score):
        level = HIGH
        if has_high_risk_evidence:
            reasons.append(
                "score >= calibrated threshold AND high-risk privileged events in "
                f"window: {', '.join(evidence.high_risk_events[:6])}"
            )
        if auth_burst:
            reasons.append(
                f"score >= calibrated threshold AND {evidence.auth_errors} "
                "authorization-failure errors in window"
            )
        if extreme_score:
            reasons.append("reconstruction error in the extreme tail (>= p99.9 of train)")
    elif over_threshold:
        level = MEDIUM
        reasons.append(
            "score >= calibrated threshold; behavioral deviation without "
            "high-risk privileged evidence"
        )
    elif elevated:
        level = MEDIUM
        reasons.append(
            f"score >= p95 of train errors ({th.p95:.6f}) but below anomaly "
            f"threshold ({th.threshold:.6f}); elevated deviation"
        )
        if has_sensitive_evidence:
            reasons.append(
                f"sensitive events present: {', '.join(evidence.sensitive_events[:6])}"
            )
    else:
        level = LOW
        reasons.append(
            f"score {score:.6f} below p95 of train errors ({th.p95:.6f}); "
            "behavior consistent with learned normal pattern"
        )

    return RiskDecision(
        risk_level=level,
        anomaly_score=float(score),
        threshold=th.threshold,
        reasons=reasons,
        evidence_summary={
            "n_events": evidence.n_events,
            "sensitive_events": evidence.sensitive_events,
            "high_risk_events": evidence.high_risk_events,
            "auth_errors": evidence.auth_errors,
            "error_events": evidence.error_events,
            "distinct_event_names": sorted(set(evidence.event_names))[:20],
        },
    )


def decision_to_dict(decision: RiskDecision) -> dict:
    return asdict(decision)
