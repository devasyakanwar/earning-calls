"""
Canonical speaker classification for earnings call transcripts.

Single source of truth for all speaker role and segment type heuristics.
Consumed by: segmentation.py, download_transcripts.py, and any future modules.

Roles: ceo, cfo, executive, analyst, operator, other
Segment types: prepared_remarks, analyst_question, management_answer, operator_transition
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Speaker classification constants
# ---------------------------------------------------------------------------

OPERATOR_KEYWORDS = ["operator", "conference call", "moderator"]

EXECUTIVE_TITLES = [
    "ceo", "chief executive", "president",
    "cfo", "chief financial", "treasurer",
    "coo", "chief operating",
    "cto", "chief technology",
    "chairman", "vice president", "vp", "svp", "evp",
    "director", "head of", "general manager", "controller",
    "ir ", "investor relations",
]

ANALYST_KEYWORDS = [
    "analyst", "research", "capital", "securities",
    "partners", "advisors", "bank", "morgan",
    "goldman", "barclays", "citi", "jpmorgan",
    "credit suisse", "ubs", "wells fargo",
    "bofa", "merrill", "deutsche",
]

# Q&A transition markers
QA_MARKERS = [
    "question-and-answer",
    "question and answer",
    "q&a session",
    "q & a",
    "open the line",
    "open it up for questions",
    "take your questions",
    "first question",
]


# ---------------------------------------------------------------------------
# Speaker override support
# ---------------------------------------------------------------------------

_OVERRIDES: dict[str, dict[str, str]] | None = None


def load_overrides(config_path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Load speaker role overrides from configs/speaker_override.yaml.

    Returns a dict of {call_id: {speaker_name_lower: role}}.
    """
    global _OVERRIDES
    if _OVERRIDES is not None:
        return _OVERRIDES

    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "speaker_override.yaml"
        )

    _OVERRIDES = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            raw = data.get("overrides", {}) or {}
            # Normalize: lower-case the speaker names for matching
            for call_id, speakers in raw.items():
                _OVERRIDES[call_id] = {
                    name.lower().strip(): role
                    for name, role in speakers.items()
                }
            if _OVERRIDES:
                logger.info("Loaded speaker overrides for %d calls", len(_OVERRIDES))
        except Exception as e:
            logger.warning("Could not load speaker overrides from %s: %s", config_path, e)

    return _OVERRIDES


# ---------------------------------------------------------------------------
# Speaker role classification
# ---------------------------------------------------------------------------


def classify_speaker_role(
    speaker_name: str,
    segment_index: int,
    call_id: str | None = None,
) -> str:
    """
    Classify a speaker into: ceo, cfo, executive, analyst, operator, other.

    Args:
        speaker_name: raw speaker name from transcript
        segment_index: position of this segment in the call (0-indexed)
        call_id: optional call identifier for override lookup

    Returns:
        One of: "ceo", "cfo", "executive", "analyst", "operator", "other"
    """
    if not speaker_name:
        return "other"

    name_lower = speaker_name.lower().strip()

    # 1. Check manual overrides first
    if call_id is not None:
        overrides = load_overrides()
        call_overrides = overrides.get(call_id, {})
        if name_lower in call_overrides:
            return call_overrides[name_lower]

    # 2. Operator detection
    if any(kw in name_lower for kw in OPERATOR_KEYWORDS):
        return "operator"

    # 3. Executive detection (check specific titles first)
    for title in EXECUTIVE_TITLES:
        if title in name_lower:
            if "ceo" in name_lower or "chief executive" in name_lower or "president" in name_lower:
                return "ceo"
            if "cfo" in name_lower or "chief financial" in name_lower:
                return "cfo"
            return "executive"

    # 4. Analyst detection
    if any(kw in name_lower for kw in ANALYST_KEYWORDS):
        return "analyst"

    # 5. Position-based heuristic: first few speakers are typically executives
    if segment_index < 3:
        return "executive"

    return "other"


# ---------------------------------------------------------------------------
# Q&A transition detection
# ---------------------------------------------------------------------------


def detect_qa_transition(text: str) -> bool:
    """Check if a text segment contains Q&A transition markers."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in QA_MARKERS)


# ---------------------------------------------------------------------------
# Segment type classification
# ---------------------------------------------------------------------------


def classify_segment_type(
    speaker_role: str,
    text: str,
    qa_started: bool,
) -> tuple[str, bool]:
    """
    Classify a segment as one of:
        prepared_remarks, analyst_question, management_answer, operator_transition.

    Returns (segment_type, qa_started_flag).
    """
    # Check for Q&A transition
    if not qa_started and detect_qa_transition(text):
        qa_started = True

    if speaker_role == "operator":
        return "operator_transition", qa_started

    if not qa_started:
        # Before Q&A, everything is prepared remarks
        if speaker_role == "analyst":
            # First analyst appearance triggers Q&A
            return "analyst_question", True
        return "prepared_remarks", False

    # Inside Q&A
    if speaker_role == "analyst":
        return "analyst_question", True

    if speaker_role in ("ceo", "cfo", "executive", "other"):
        return "management_answer", True

    return "management_answer", True
