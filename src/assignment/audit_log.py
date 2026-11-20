"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from assignment._files import write_json
from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[float, str, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the sanitized input and start time for this request."""
        self._open[user_id, request_id] = (
            monotonic(),
            utc_now_iso(),
            content_filter(text)["redacted"],
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the request and record the guarded response and decision."""
        pending = self._open.pop((user_id, request_id), None)
        started, timestamp, input_text = pending if pending is not None else (None, None, None)
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "input": input_text,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((monotonic() - started) * 1_000, 3) if started is not None else None,
        })

    def export_json(self, filepath: str | Path | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        write_json(filepath or default_audit_log_path(), self.logs)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
