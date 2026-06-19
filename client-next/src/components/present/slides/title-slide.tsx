import { Eyebrow } from "../primitives";

export function TitleSlide() {
  return (
    <div className="space-y-10">
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden
          className="block h-4 w-4 rounded-sm bg-foreground"
          style={{
            boxShadow:
              "inset 0 0 0 1.5px var(--background), 0 0 0 1px var(--foreground)",
          }}
        />
        <span className="text-lg font-medium tracking-tight text-foreground">
          VisualReF
        </span>
        <span className="font-mono text-sm text-muted-foreground">v2</span>
      </div>

      <div className="space-y-5">
        <Eyebrow>Bachelor Thesis · Interactive Image Retrieval</Eyebrow>
        <h1 className="max-w-4xl text-6xl font-medium leading-[1.05] tracking-tight text-foreground lg:text-7xl">
          Fine-Grained Multimodal Relevance Feedback
          <span className="text-muted-foreground">
            {" "}
            for Interactive Image Retrieval
          </span>
        </h1>
      </div>

      <div className="flex flex-col gap-1.5 border-l border-border pl-6">
        <p className="text-2xl font-medium tracking-tight text-foreground">
          Rares Boghean
        </p>
        <p className="text-lg text-muted-foreground">
          Department of Advanced Computing Sciences
        </p>
        <p className="text-lg text-muted-foreground">
          Faculty of Science and Engineering · Maastricht University
        </p>
        <p className="mt-1 font-mono text-base text-muted-foreground">
          r.boghean@student.maastrichtuniversity.nl
        </p>
      </div>
    </div>
  );
}
