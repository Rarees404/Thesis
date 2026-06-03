"""Analyze human user-study logs into RQ-relevant metrics + LaTeX.

Joins the per-participant event logs (`logs/study/{id}.jsonl`) and questionnaire
files (`logs/study/{id}.form.json`) with the offline ground truth
(`eval/data/queries.json`) so that the same relevant-image sets used by the
simulated eval also score the human runs (precision-at-stop). Objective task
metrics + subjective questionnaire aggregates are written to study_summary.json
and study_tables.tex.

Usage (from server/):
  python -m src.eval.analyze_study \
      --logs-dir ../logs/study --queries src/eval/data/queries.json \
      --out-dir src/eval/study_results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Optional, Set

from src.eval import metrics as M
from src.eval.questions_meta import LIKERT_IDS, TEXT_IDS, RQ_OF  # see fallback below

KS = (5, 10)


def _img_id(path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else None


def _ranked_ids(paths: List[str]) -> List[int]:
    out = []
    for p in paths:
        i = _img_id(p)
        if i is not None:
            out.append(i)
    return out


def load_ground_truth(queries_path: str) -> Dict[str, Set[int]]:
    data = json.load(open(queries_path))
    gt: Dict[str, Set[int]] = {}
    for kind in ("category", "compositional"):
        for q in data.get(kind, []):
            gt[q["qid"]] = {int(k) for k in q.get("relevant", {})}
    return gt


def parse_participant(jsonl_path: str):
    events = [json.loads(line) for line in open(jsonl_path) if line.strip()]
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for e in events:
        by_task[e.get("task_id") or "_none"].append(e)
    return events, by_task


def task_rankings(task_events: List[dict]) -> List[List[str]]:
    """Ordered per-round result path lists: initial search, then each feedback."""
    rankings: List[List[str]] = []
    for e in task_events:
        if e["event_type"] in ("search_submitted", "feedback_applied"):
            paths = e.get("data", {}).get("result_paths")
            if paths:
                rankings.append(paths)
    return rankings


def analyze_task(task_id: str, task_events: List[dict], gt: Dict[str, Set[int]]) -> Optional[dict]:
    finished = [e for e in task_events if e["event_type"] == "task_finished"]
    if not finished:
        return None
    fin = finished[-1]["data"]
    rankings = task_rankings(task_events)
    relevant = gt.get(task_id, set())
    final_ids = _ranked_ids(rankings[-1]) if rankings else []

    res = {
        "task_id": task_id,
        "outcome": fin.get("outcome"),
        "success": fin.get("outcome") == "found",
        "rounds": fin.get("rounds", len(rankings)),
        "duration_s": round(fin.get("duration_ms", 0) / 1000.0, 1),
        "n_rounds_logged": len(rankings),
    }
    if relevant and final_ids:
        for k in KS:
            res[f"P@{k}_at_stop"] = M.precision_at_k(final_ids, relevant, k)
        # target actually retrieved at stop?
        tgt = _img_id(fin.get("target_path", ""))
        res["target_in_results"] = tgt in final_ids if tgt is not None else None
    return res


def aggregate(values: List[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(mean(vals), 3),
            "std": round(pstdev(vals), 3) if len(vals) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    gt = load_ground_truth(args.queries)
    os.makedirs(args.out_dir, exist_ok=True)

    per_task: List[dict] = []
    forms: List[dict] = []
    participants = 0
    for jsonl in sorted(glob.glob(os.path.join(args.logs_dir, "*.jsonl"))):
        participants += 1
        _events, by_task = parse_participant(jsonl)
        for task_id, te in by_task.items():
            if task_id == "_none":
                continue
            r = analyze_task(task_id, te, gt)
            if r:
                r["participant"] = os.path.basename(jsonl)[:-6]
                per_task.append(r)
    for form in sorted(glob.glob(os.path.join(args.logs_dir, "*.form.json"))):
        forms.append(json.load(open(form)))

    # Objective aggregates
    obj = {
        "participants": participants,
        "tasks_completed": len(per_task),
        "success_rate": aggregate([1.0 if t["success"] else 0.0 for t in per_task]),
        "rounds": aggregate([t["rounds"] for t in per_task]),
        "duration_s": aggregate([t["duration_s"] for t in per_task]),
    }
    for k in KS:
        obj[f"P@{k}_at_stop"] = aggregate([t.get(f"P@{k}_at_stop") for t in per_task])
    obj["target_in_results_rate"] = aggregate(
        [1.0 if t.get("target_in_results") else 0.0 for t in per_task
         if t.get("target_in_results") is not None])

    # Subjective aggregates (Likert per item + per RQ)
    likert_by_item: Dict[str, List[float]] = defaultdict(list)
    for fm in forms:
        ans = fm.get("answers", {})
        for qid in LIKERT_IDS:
            if isinstance(ans.get(qid), (int, float)):
                likert_by_item[qid].append(float(ans[qid]))
    subj_items = {qid: aggregate(v) for qid, v in likert_by_item.items()}
    rq_groups: Dict[str, List[float]] = defaultdict(list)
    for qid, vals in likert_by_item.items():
        rq_groups[RQ_OF.get(qid, "usability")].extend(vals)
    subj_by_rq = {rq: aggregate(v) for rq, v in rq_groups.items()}
    open_text = {qid: [fm.get("answers", {}).get(qid) for fm in forms
                       if fm.get("answers", {}).get(qid)] for qid in TEXT_IDS}

    summary = {
        "objective": obj,
        "subjective_items": subj_items,
        "subjective_by_rq": subj_by_rq,
        "open_text": open_text,
        "n_questionnaires": len(forms),
    }
    with open(os.path.join(args.out_dir, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_latex(summary, os.path.join(args.out_dir, "study_tables.tex"))

    print(f"[study] {participants} participants, {len(per_task)} tasks, "
          f"{len(forms)} questionnaires")
    print(f"  success={obj['success_rate'].get('mean')} "
          f"rounds={obj['rounds'].get('mean')} "
          f"dur(s)={obj['duration_s'].get('mean')} "
          f"P@5={obj.get('P@5_at_stop', {}).get('mean')}")
    print(f"[done] wrote study_summary.json + study_tables.tex to {args.out_dir}")


def _row(label: str, agg: dict) -> str:
    if not agg or agg.get("n", 0) == 0:
        return f"{label} & -- & -- \\\\"
    return f"{label} & {agg['mean']} & {agg.get('std', 0.0)} \\\\"


def _write_latex(summary: dict, path: str) -> None:
    obj = summary["objective"]
    lines = [
        "\\begin{table}[t]",
        "\\caption{User study: objective task metrics (mean over completed tasks).}",
        "\\label{tab:study-objective}",
        "\\centering\\small",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Measure & Mean & SD \\\\",
        "\\midrule",
        _row("Success rate", obj["success_rate"]),
        _row("Feedback rounds", obj["rounds"]),
        _row("Completion time (s)", obj["duration_s"]),
        _row("P@5 at stop", obj.get("P@5_at_stop", {})),
        _row("P@10 at stop", obj.get("P@10_at_stop", {})),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\caption{User study: questionnaire (1--5 Likert) by research question.}",
        "\\label{tab:study-subjective}",
        "\\centering\\small",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Dimension & Mean & SD \\\\",
        "\\midrule",
    ]
    for rq, agg in summary["subjective_by_rq"].items():
        lines.append(_row(rq, agg))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
