"""Lightweight checks for the user-study endpoints.

Exercises the production /study/log and /study/form handlers via
FastAPI TestClient without triggering the lifespan (so SigLIP / SAM /
FAISS are never loaded). Logs are written to a tmp directory and read
back to verify shape and content.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def study_client(tmp_path, monkeypatch):
    """Build a TestClient whose study logs land in a temp dir.

    We patch ``settings.logs_path`` BEFORE importing the server module so
    the first call to ``_study_dir()`` resolves under ``tmp_path``.
    Lifespan startup is skipped (TestClient without ``with`` block does
    not run startup events), so no heavy models load.
    """
    from src import config as cfg

    monkeypatch.setattr(cfg.settings, "logs_path", str(tmp_path))

    from src import retrieval_server_visual as srv

    client = TestClient(srv.app)
    yield client, tmp_path / "study"


def test_study_log_appends_jsonl(study_client):
    client, study_dir = study_client
    pid = "P_smoke"
    sid = "session-abc"
    tid = "cat_eyelid.n.01"

    events = [
        {
            "participant_id": pid,
            "session_id": sid,
            "event_type": "session_start",
            "client_ts": time.time() * 1000,
            "data": {"total_tasks": 1},
        },
        {
            "participant_id": pid,
            "session_id": sid,
            "task_id": tid,
            "event_type": "task_start",
            "client_ts": time.time() * 1000,
            "data": {"query": "lid", "target_path": "data/visual_genome/VG_100K/6.jpg"},
        },
        {
            "participant_id": pid,
            "session_id": sid,
            "task_id": tid,
            "event_type": "search_submitted",
            "client_ts": time.time() * 1000,
            "data": {
                "query": "lid",
                "top_k": 12,
                "latency_ms": 412,
                "result_paths": [
                    "data/visual_genome/VG_100K/14.jpg",
                    "data/visual_genome/VG_100K/6.jpg",
                ],
                "scores": [0.71, 0.69],
            },
        },
        {
            "participant_id": pid,
            "session_id": sid,
            "task_id": tid,
            "event_type": "feedback_applied",
            "client_ts": time.time() * 1000,
            "data": {
                "round": 1,
                "latency_ms": 988,
                "relevant_hint": "close-up of an eye",
                "irrelevant_hint": "",
                "sam_regions": 1,
                "full_labels": 0,
                "result_paths": [
                    "data/visual_genome/VG_100K/6.jpg",
                    "data/visual_genome/VG_100K/14.jpg",
                ],
                "scores": [0.78, 0.66],
                "hard_filter": {
                    "blacklist_size": 0,
                    "boostlist_size": 1,
                    "dropped": 0,
                    "boosted": 1,
                    "overfetch": 72,
                    "top_negative_phrases": [],
                    "top_positive_phrases": [{"phrase": "eyelid", "count": 1}],
                },
            },
        },
        {
            "participant_id": pid,
            "session_id": sid,
            "task_id": tid,
            "event_type": "task_finished",
            "client_ts": time.time() * 1000,
            "data": {
                "task_id": tid,
                "query": "lid",
                "outcome": "found",
                "rounds": 1,
                "duration_ms": 42_000,
                "final_paths": [
                    "data/visual_genome/VG_100K/6.jpg",
                    "data/visual_genome/VG_100K/14.jpg",
                ],
                "target_path": "data/visual_genome/VG_100K/6.jpg",
            },
        },
    ]

    for ev in events:
        r = client.post("/study/log", json=ev)
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}

    log_file = study_dir / f"{pid}.jsonl"
    assert log_file.exists(), f"missing {log_file}"
    lines = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    assert len(lines) == len(events)
    assert [ln["event_type"] for ln in lines] == [
        "session_start",
        "task_start",
        "search_submitted",
        "feedback_applied",
        "task_finished",
    ]
    for ln in lines:
        assert ln["participant_id"] == pid
        assert ln["session_id"] == sid
        assert "server_ts" in ln and isinstance(ln["server_ts"], (int, float))


def test_study_form_writes_json(study_client):
    client, study_dir = study_client
    pid = "P_smoke_form"
    sid = "session-xyz"
    answers = {
        "ease": 4,
        "region_help": 5,
        "text_help": 4,
        "combo_help": 5,
        "no_repeats": 3,
        "responsive": 4,
        "confidence": 4,
        "again": 5,
        "liked": "The point-click felt instant.",
        "improve": "Could explain hard-filter when it kicks in.",
    }
    r = client.post(
        "/study/form",
        json={"participant_id": pid, "session_id": sid, "answers": answers},
    )
    assert r.status_code == 200, r.text

    form_file = study_dir / f"{pid}.form.json"
    assert form_file.exists()
    payload = json.loads(form_file.read_text())
    assert payload["participant_id"] == pid
    assert payload["session_id"] == sid
    assert payload["answers"] == answers


def test_safe_pid_filters_unsafe_chars(study_client):
    """Path-traversal / odd-char IDs should be sanitized to a safe filename."""
    client, study_dir = study_client
    r = client.post(
        "/study/log",
        json={
            "participant_id": "../../etc/passwd",
            "event_type": "session_start",
            "data": {},
        },
    )
    assert r.status_code == 200
    # The sanitizer collapses non-alphanumerics to underscores and trims them.
    assert any(p.name.endswith(".jsonl") for p in study_dir.iterdir())
    for p in study_dir.iterdir():
        assert "/" not in p.name
        assert ".." not in p.name


def test_invalid_payload_rejected(study_client):
    """Missing participant_id should fail pydantic validation (422)."""
    client, _ = study_client
    r = client.post("/study/log", json={"event_type": "session_start"})
    assert r.status_code == 422
