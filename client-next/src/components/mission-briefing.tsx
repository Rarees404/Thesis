"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";

interface MissionBriefingProps {
  onComplete: () => void;
}

function useTypewriter(text: string, speed: number, startDelay: number) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => {
      let i = 0;
      const interval = setInterval(() => {
        setDisplayed(text.slice(0, i + 1));
        i++;
        if (i >= text.length) {
          clearInterval(interval);
          setDone(true);
        }
      }, speed);
      return () => clearInterval(interval);
    }, startDelay);
    return () => clearTimeout(timeout);
  }, [text, speed, startDelay]);

  return { displayed, done };
}

const REDACTED_LINES = [
  { w: "w-[70%]", delay: 0.3 },
  { w: "w-[85%]", delay: 0.45 },
  { w: "w-[60%]", delay: 0.6 },
  { w: "w-[90%]", delay: 0.75 },
  { w: "w-[45%]", delay: 0.9 },
];

export function MissionBriefing({ onComplete }: MissionBriefingProps) {
  const [phase, setPhase] = useState(0);

  const title = useTypewriter("OPERATION: VISUALREF", 65, 1200);
  const subtitle = useTypewriter(
    "VISUAL INTELLIGENCE RETRIEVAL & FEEDBACK SYSTEM",
    30,
    3600
  );

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase(1), 400),
      setTimeout(() => setPhase(2), 1200),
      setTimeout(() => setPhase(3), 3400),
      setTimeout(() => setPhase(4), 5600),
      setTimeout(() => setPhase(5), 6400),
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  const handleEnter = useCallback(() => {
    setPhase(6);
    setTimeout(onComplete, 600);
  }, [onComplete]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (phase >= 5 && e.key === "Enter") handleEnter();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase, handleEnter]);

  return (
    <AnimatePresence>
      {phase < 6 && (
        <motion.div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Scan lines overlay */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.04]"
            style={{
              backgroundImage:
                "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.08) 2px, rgba(255,255,255,0.08) 4px)",
            }}
          />

          {/* Vignette */}
          <div
            className="pointer-events-none absolute inset-0"
            style={{
              background:
                "radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.7) 100%)",
            }}
          />

          <div className="relative w-full max-w-2xl px-8">
            {/* CLASSIFIED stamp */}
            <AnimatePresence>
              {phase >= 1 && (
                <motion.div
                  initial={{ opacity: 0, scale: 1.5, rotate: -12 }}
                  animate={{ opacity: [0, 0.9, 0.6], scale: 1, rotate: -8 }}
                  transition={{ duration: 0.3, times: [0, 0.5, 1] }}
                  className="absolute -top-16 -right-4 select-none"
                >
                  <span className="font-rajdhani text-5xl font-bold tracking-[0.2em] text-red-600/60 border-4 border-red-600/50 px-4 py-1">
                    CLASSIFIED
                  </span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Document header line */}
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: phase >= 1 ? 1 : 0 }}
              transition={{ duration: 0.4 }}
              className="mb-6 h-[1px] origin-left bg-red-600/40"
            />

            {/* Document classification */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: phase >= 1 ? 1 : 0 }}
              className="mb-1 font-mono text-[10px] tracking-[0.4em] text-red-500/60 uppercase"
            >
              TOP SECRET // SI // NOFORN
            </motion.p>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: phase >= 1 ? 1 : 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8 font-mono text-[10px] tracking-[0.3em] text-neutral-600"
            >
              DOCUMENT REF: VRF-2026-7734-ALPHA
            </motion.p>

            {/* Title typing */}
            {phase >= 2 && (
              <div className="mb-3">
                <h1 className="font-rajdhani text-4xl font-bold tracking-[0.15em] text-red-500 sm:text-5xl">
                  {title.displayed}
                  {!title.done && (
                    <span className="inline-block w-[3px] h-[1em] bg-red-500 ml-1 animate-pulse" />
                  )}
                </h1>
              </div>
            )}

            {/* Subtitle typing */}
            {phase >= 3 && (
              <div className="mb-8">
                <p className="font-mono text-xs tracking-[0.2em] text-neutral-500 sm:text-sm">
                  {subtitle.displayed}
                  {!subtitle.done && (
                    <span className="inline-block w-[2px] h-[1em] bg-neutral-500 ml-1 animate-pulse" />
                  )}
                </p>
              </div>
            )}

            {/* Redacted lines */}
            {phase >= 3 && (
              <div className="mb-8 space-y-2.5">
                {REDACTED_LINES.map((line, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: line.delay, duration: 0.3 }}
                    className={`${line.w} h-2.5 rounded-[2px] bg-neutral-800`}
                  />
                ))}
              </div>
            )}

            {/* ACCESS GRANTED + Enter */}
            <AnimatePresence>
              {phase >= 4 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.15 }}
                >
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: [0, 1, 0.8, 1], y: 0 }}
                    transition={{
                      duration: 0.6,
                      times: [0, 0.3, 0.6, 1],
                    }}
                    className="mb-6 inline-flex items-center gap-3 border border-green-500/30 bg-green-500/[0.06] px-4 py-2"
                  >
                    <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="font-rajdhani text-sm font-bold tracking-[0.3em] text-green-400">
                      ACCESS GRANTED
                    </span>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {phase >= 5 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <button
                    onClick={handleEnter}
                    className="group flex items-center gap-3 border border-red-600/30 bg-red-600/[0.06] px-6 py-3 transition-all hover:bg-red-600/[0.12] hover:border-red-500/50"
                  >
                    <span className="font-rajdhani text-sm font-bold tracking-[0.25em] text-red-400 group-hover:text-red-300 transition-colors">
                      ENTER MISSION CONTROL
                    </span>
                    <motion.span
                      animate={{ x: [0, 4, 0] }}
                      transition={{ repeat: Infinity, duration: 1.5 }}
                      className="text-red-500"
                    >
                      &rarr;
                    </motion.span>
                  </button>
                  <p className="mt-3 font-mono text-[10px] text-neutral-600 tracking-widest">
                    PRESS ENTER OR CLICK TO PROCEED
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Bottom line */}
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: phase >= 4 ? 1 : 0 }}
              transition={{ duration: 0.6 }}
              className="mt-8 h-[1px] origin-right bg-red-600/30"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
