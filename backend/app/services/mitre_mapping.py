"""MITRE ATT&CK mapping layer (Phase 8).

SEPARATE from the model: the GRU-Autoencoder only produces a reconstruction
error. This layer inspects the actual CloudTrail events inside the DETECTED
window and matches them against evidence rules in config/mitre_mapping.json
(technique IDs verified against attack.mitre.org).

If no rule matches the evidence -> "No confident MITRE mapping".
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import MITRE_MAPPING_FILE

NO_MAPPING = "No confident MITRE mapping"


@dataclass
class MitreMatch:
    technique_id: str
    technique_name: str
    tactic: str
    rationale: str
    supporting_behavior: list[str]
    matched_count: int


@dataclass
class MitreResult:
    status: str                    # "mapped" | "no_mapping"
    technique_id: str | None
    technique_name: str | None
    tactic: str | None
    rationale: str | None
    supporting_behavior: list[str]
    matched_count: int


class MitreMapper:
    """Rule-based evidence mapper backed by a maintainable JSON config."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or MITRE_MAPPING_FILE
        if not self.config_path.exists():
            raise FileNotFoundError(f"MITRE mapping config missing: {self.config_path}")
        self._cfg = json.loads(self.config_path.read_text())
        self._rules = sorted(self._cfg["rules"], key=lambda r: r["priority"])
        self._lock = threading.Lock()

    @property
    def sensitive_events(self) -> set[str]:
        return set(self._cfg.get("sensitive_event_names", []))

    def map_window(self, event_names: list[str], error_codes: list[str] | None = None) -> MitreResult:
        """Map one detected window's behavior to ATT&CK, or return no-mapping."""
        from collections import Counter

        counts = Counter(event_names)
        best: MitreMatch | None = None
        with self._lock:
            for rule in self._rules:
                matched = {
                    n: counts[n]
                    for n in rule["match_event_names"]
                    if counts.get(n, 0) > 0
                }
                total = sum(matched.values())
                if total >= rule["min_count"]:
                    best = MitreMatch(
                        technique_id=rule["technique_id"],
                        technique_name=rule["technique_name"],
                        tactic=rule["tactic"],
                        rationale=rule["rationale"],
                        supporting_behavior=[
                            f"{n} x{c}" for n, c in sorted(matched.items(), key=lambda x: -x[1])
                        ],
                        matched_count=total,
                    )
                    break  # rules are priority-ordered; take the strongest fit

        if best is None:
            return MitreResult(
                status="no_mapping",
                technique_id=None,
                technique_name=None,
                tactic=None,
                rationale=(
                    "Window contents do not satisfy any evidence rule in "
                    "config/mitre_mapping.json; insufficient evidence for a "
                    "confident ATT&CK mapping."
                ),
                supporting_behavior=[],
                matched_count=0,
            )
        return MitreResult(
            status="mapped",
            technique_id=best.technique_id,
            technique_name=best.technique_name,
            tactic=best.tactic,
            rationale=best.rationale,
            supporting_behavior=best.supporting_behavior,
            matched_count=best.matched_count,
        )

    def list_rules(self) -> list[dict]:
        return list(self._cfg["rules"])


_default: MitreMapper | None = None


def get_mitre_mapper() -> MitreMapper:
    """Process-wide singleton (config parsed once)."""
    global _default
    if _default is None:
        _default = MitreMapper()
    return _default


def result_to_dict(result: MitreResult) -> dict:
    return asdict(result)
