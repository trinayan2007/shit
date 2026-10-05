"""Response engine (Phase 9) with adapter interface.

DEFAULT MODE IS SAFE SIMULATION. The SimulatedAdapter records the action and
flips a user's status ACTIVE -> BLOCKED in the local SQLite database only.
It never calls AWS.

A real AWS implementation (RealAWSAdapter) can be added by implementing the
same ResponseAdapter interface; it stays disabled unless
AWS_REAL_BLOCKING_ENABLED=true. It is PROPOSED functionality, not implemented
call logic here - it raises NotImplementedError on use.

Response policy (from the PPT):
  LOW    -> log/alert      (recorded, no action)
  MEDIUM -> escalation     (recorded as NOTIFICATION)
  HIGH   -> mitigation     (BLOCK, operator-triggered, simulated)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import AWS_REAL_BLOCKING_ENABLED

logger = logging.getLogger(__name__)

SIMULATED_LABEL = "SIMULATED RESPONSE"


@dataclass
class ResponseResult:
    action: str            # LOG | NOTIFY | BLOCK
    status: str            # EXECUTED | FAILED
    mode: str              # SIMULATED | REAL
    detail: str
    executed_at: str


class ResponseAdapter:
    """Interface - real AWS integration slots in here later."""

    def execute_block(self, principal_key: str, context: dict) -> ResponseResult:
        raise NotImplementedError


class SimulatedAdapter(ResponseAdapter):
    """Local-only simulation: no AWS calls whatsoever."""

    def execute_block(self, principal_key: str, context: dict) -> ResponseResult:
        logger.info("[SIMULATED] block %s (no AWS call)", principal_key)
        return ResponseResult(
            action="BLOCK",
            status="EXECUTED",
            mode="SIMULATED",
            detail=(
                f"{SIMULATED_LABEL}: principal '{principal_key}' marked BLOCKED in "
                "the local database. No AWS credentials were used and no real AWS "
                "account or IAM user was disabled."
            ),
            executed_at=_utcnow(),
        )


class RealAWSAdapter(ResponseAdapter):
    """PROPOSED: real AWS IAM blocking. Disabled and unimplemented by default."""

    def execute_block(self, principal_key: str, context: dict) -> ResponseResult:
        if not AWS_REAL_BLOCKING_ENABLED:
            raise RuntimeError(
                "Real AWS blocking is disabled (AWS_REAL_BLOCKING_ENABLED != true)."
            )
        raise NotImplementedError(
            "RealAWSAdapter is a proposed future integration: implement IAM "
            "actions (DeleteAccessKey / UpdateAccessKey / AttachUserPolicy deny) "
            "here with boto3 and least-privilege credentials."
        )


def get_adapter() -> ResponseAdapter:
    if AWS_REAL_BLOCKING_ENABLED:
        logger.warning("AWS real blocking enabled - RealAWSAdapter selected")
        return RealAWSAdapter()
    return SimulatedAdapter()


def automatic_action_for(risk_level: str) -> tuple[str, str]:
    """Risk policy -> (action, status) applied automatically at detection time."""
    if risk_level == "HIGH":
        return "BLOCK", "PENDING"      # operator must trigger it (simulated)
    if risk_level == "MEDIUM":
        return "NOTIFY", "ESCALATED"
    return "LOG", "LOGGED"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
