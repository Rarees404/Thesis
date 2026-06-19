import type { ComponentType } from "react";
import { TitleSlide } from "./slides/title-slide";
import { MotivationSlide } from "./slides/motivation-slide";
import { PipelineSlide } from "./slides/pipeline-slide";
import { StudyDesignSlide } from "./slides/study-design-slide";
import { StudyMetricsSlide } from "./slides/study-metrics-slide";
import { StudyObjectiveSlide } from "./slides/study-objective-slide";
import { StudySubjectiveSlide } from "./slides/study-subjective-slide";
import { LimitationsSlide } from "./slides/limitations-slide";
import { ClosingSlide } from "./slides/closing-slide";
import {
  SectionMotivation,
  SectionHowItWorks,
  SectionUserStudy,
  SectionReflections,
} from "./slides/section-slide";

export interface SlideDef {
  id: string;
  component: ComponentType<{ step: number }>;
  steps?: number;
  fullBleed?: boolean;
}

export const SLIDES: SlideDef[] = [
  { id: "title", component: TitleSlide },
  { id: "sec-motivation", component: SectionMotivation, fullBleed: true },
  { id: "motivation", component: MotivationSlide },
  { id: "sec-how", component: SectionHowItWorks, fullBleed: true },
  { id: "pipeline", component: PipelineSlide, steps: 7 },
  { id: "sec-study", component: SectionUserStudy, fullBleed: true },
  { id: "study-design", component: StudyDesignSlide },
  { id: "study-metrics", component: StudyMetricsSlide },
  { id: "study-objective", component: StudyObjectiveSlide, steps: 2 },
  { id: "study-subjective", component: StudySubjectiveSlide, steps: 1 },
  { id: "sec-reflections", component: SectionReflections, fullBleed: true },
  { id: "limitations", component: LimitationsSlide },
  { id: "closing", component: ClosingSlide },
];
