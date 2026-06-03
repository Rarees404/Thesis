"""Questionnaire metadata, mirroring client-next/src/lib/study/questions.ts.

Kept in sync by hand: if you add/rename a questionnaire item in the frontend,
update it here so analyze_study.py aggregates it correctly.
"""

# question_id -> research question / dimension it informs
RQ_OF = {
    "ease": "usability",
    "region_help": "RQ2",
    "text_help": "RQ1",
    "combo_help": "RQ1",
    "no_repeats": "RQ3",
    "responsive": "usability",
    "confidence": "usability",
    "again": "usability",
}

LIKERT_IDS = list(RQ_OF.keys())
TEXT_IDS = ["liked", "improve"]
