"use client";

import { Loader2, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  STUDY_QUESTIONS,
  LIKERT_LABELS,
  type StudyQuestion,
} from "@/lib/study/questions";
import type { QuestionnaireAnswers } from "@/lib/study/types";

interface QuestionnaireProps {
  answers: QuestionnaireAnswers;
  setAnswer: (id: string, value: number | string) => void;
  onSubmit: () => void;
  submitting?: boolean;
}

export function Questionnaire({
  answers,
  setAnswer,
  onSubmit,
  submitting,
}: QuestionnaireProps) {
  const likertItems = STUDY_QUESTIONS.filter((q) => q.type === "likert");
  const allLikertAnswered = likertItems.every((q) => answers[q.id] != null);

  return (
    <div className="mx-auto max-w-2xl space-y-6 py-6">
      <header className="space-y-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          Almost done
        </p>
        <h1 className="text-2xl font-medium tracking-tight text-foreground">
          How was it?
        </h1>
        <p className="text-[14px] leading-relaxed text-muted-foreground">
          Your answers help us evaluate the system. There are no wrong answers.
        </p>
      </header>

      <div className="space-y-5">
        {STUDY_QUESTIONS.map((q) =>
          q.type === "likert" ? (
            <LikertRow
              key={q.id}
              question={q}
              value={answers[q.id] as number | undefined}
              onChange={(v) => setAnswer(q.id, v)}
            />
          ) : (
            <TextRow
              key={q.id}
              question={q}
              value={(answers[q.id] as string) ?? ""}
              onChange={(v) => setAnswer(q.id, v)}
            />
          ),
        )}
      </div>

      <div className="flex items-center justify-between border-t border-border pt-4">
        <span className="font-mono text-[11px] text-muted-foreground">
          {allLikertAnswered
            ? "All rating questions answered"
            : "Please answer all rating questions"}
        </span>
        <Button
          onClick={onSubmit}
          disabled={!allLikertAnswered || submitting}
          className="h-9 gap-1.5 px-4"
        >
          {submitting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Send className="h-3.5 w-3.5" />
          )}
          Submit
        </Button>
      </div>
    </div>
  );
}

function LikertRow({
  question,
  value,
  onChange,
}: {
  question: StudyQuestion;
  value: number | undefined;
  onChange: (v: number) => void;
}) {
  return (
    <fieldset className="space-y-2 rounded-md border border-border bg-card px-4 py-3">
      <legend className="text-[13.5px] text-foreground/90">
        {question.prompt}
      </legend>
      <div className="flex items-stretch gap-1.5">
        {LIKERT_LABELS.map((label, i) => {
          const score = i + 1;
          const active = value === score;
          return (
            <button
              key={score}
              type="button"
              onClick={() => onChange(score)}
              title={label}
              className={`flex-1 rounded border px-1 py-2 text-center transition-colors duration-100 ${
                active
                  ? "border-foreground/40 bg-foreground/[0.08] text-foreground"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              <span className="block font-mono text-[13px] tabular-nums">
                {score}
              </span>
            </button>
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground/60">
        <span>{LIKERT_LABELS[0]}</span>
        <span>{LIKERT_LABELS[LIKERT_LABELS.length - 1]}</span>
      </div>
    </fieldset>
  );
}

function TextRow({
  question,
  value,
  onChange,
}: {
  question: StudyQuestion;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-2 rounded-md border border-border bg-card px-4 py-3">
      <label className="text-[13.5px] text-foreground/90">{question.prompt}</label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        placeholder="Optional"
        className="w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-[13px] text-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
      />
    </div>
  );
}
