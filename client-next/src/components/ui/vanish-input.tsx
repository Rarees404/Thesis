"use client";

import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface VanishInputProps {
  placeholders: string[];
  value?: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit?: (e: React.FormEvent<HTMLFormElement>) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  disabled?: boolean;
  className?: string;
  accentColor?: string; // tailwind color class for focus ring e.g. "indigo"
  submitIcon?: React.ReactNode;
  hideSubmitButton?: boolean;
}

export function VanishInput({
  placeholders,
  value: externalValue,
  onChange,
  onSubmit,
  onKeyDown,
  disabled,
  className,
  accentColor = "indigo",
  submitIcon,
  hideSubmitButton = false,
}: VanishInputProps) {
  const [currentPlaceholder, setCurrentPlaceholder] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startAnimation = () => {
    intervalRef.current = setInterval(() => {
      setCurrentPlaceholder((prev) => (prev + 1) % placeholders.length);
    }, 4500);
  };

  const handleVisibilityChange = () => {
    if (document.visibilityState !== "visible" && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    } else if (document.visibilityState === "visible") {
      startAnimation();
    }
  };

  useEffect(() => {
    startAnimation();
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [placeholders]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const newDataRef = useRef<{ x: number; y: number; r: number; color: string }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const [internalValue, setInternalValue] = useState("");
  const [animating, setAnimating] = useState(false);

  const controlled = externalValue !== undefined;
  const displayValue = controlled ? (externalValue ?? "") : internalValue;

  const draw = useCallback(() => {
    if (!inputRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 800;
    canvas.height = 800;
    ctx.clearRect(0, 0, 800, 800);
    const computedStyles = getComputedStyle(inputRef.current);
    const fontSize = parseFloat(computedStyles.getPropertyValue("font-size"));
    ctx.font = `${fontSize * 2}px ${computedStyles.fontFamily}`;
    ctx.fillStyle = "#FFF";
    ctx.fillText(displayValue, 16, 40);

    const imageData = ctx.getImageData(0, 0, 800, 800);
    const pixelData = imageData.data;
    const newData: { x: number; y: number; color: [number, number, number, number] }[] = [];

    for (let t = 0; t < 800; t++) {
      const i = 4 * t * 800;
      for (let n = 0; n < 800; n++) {
        const e = i + 4 * n;
        if (pixelData[e] !== 0 && pixelData[e + 1] !== 0 && pixelData[e + 2] !== 0) {
          newData.push({ x: n, y: t, color: [pixelData[e], pixelData[e + 1], pixelData[e + 2], pixelData[e + 3]] });
        }
      }
    }

    newDataRef.current = newData.map(({ x, y, color }) => ({
      x, y, r: 1,
      color: `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3]})`,
    }));
  }, [displayValue]);

  useEffect(() => {
    draw();
  }, [displayValue, draw]);

  const animate = (start: number) => {
    const animateFrame = (pos: number = 0) => {
      requestAnimationFrame(() => {
        const newArr: typeof newDataRef.current = [];
        for (let i = 0; i < newDataRef.current.length; i++) {
          const current = newDataRef.current[i];
          if (current.x < pos) {
            newArr.push(current);
          } else {
            if (current.r <= 0) { current.r = 0; continue; }
            // slower drift — smaller random displacement, slower fade
            current.x += (Math.random() - 0.48) * 1.2;
            current.y += (Math.random() - 0.5) * 1.2;
            current.r -= 0.025 * Math.random();
            newArr.push(current);
          }
        }
        newDataRef.current = newArr;
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx) {
          ctx.clearRect(pos, 0, 800, 800);
          newDataRef.current.forEach(({ x: n, y: i, r: s, color }) => {
            if (n > pos) {
              ctx.beginPath();
              ctx.rect(n, i, s, s);
              ctx.fillStyle = color;
              ctx.strokeStyle = color;
              ctx.stroke();
            }
          });
        }
        if (newDataRef.current.length > 0) {
          // step size 4 instead of 8 — sweep progresses more slowly
          animateFrame(pos - 4);
        } else {
          if (!controlled) setInternalValue("");
          setAnimating(false);
        }
      });
    };
    animateFrame(start);
  };

  const vanishAndSubmit = () => {
    setAnimating(true);
    draw();
    const val = inputRef.current?.value || "";
    if (val) {
      const maxX = newDataRef.current.reduce((prev, cur) => (cur.x > prev ? cur.x : prev), 0);
      animate(maxX);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !animating && !disabled) {
      vanishAndSubmit();
    }
    onKeyDown?.(e);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    vanishAndSubmit();
    onSubmit?.(e);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (animating || disabled) return;
    if (!controlled) setInternalValue(e.target.value);
    onChange?.(e);
  };

  const focusRingMap: Record<string, string> = {
    indigo: "focus-within:border-indigo-500/50 focus-within:ring-indigo-500/20",
    green: "focus-within:border-green-500/40 focus-within:ring-green-500/15",
    red: "focus-within:border-red-500/40 focus-within:ring-red-500/15",
    violet: "focus-within:border-violet-500/50 focus-within:ring-violet-500/20",
  };
  const ringClass = focusRingMap[accentColor] ?? focusRingMap.indigo;

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        "relative w-full h-12 rounded-xl overflow-hidden",
        "border border-white/[0.08] bg-white/[0.04] backdrop-blur-xl",
        "shadow-[0_0_0_1px_rgba(255,255,255,0.04)_inset]",
        "transition-all duration-200 focus-within:ring-1",
        ringClass,
        disabled && "opacity-50 pointer-events-none",
        className
      )}
    >
      {/* vanish canvas */}
      <canvas
        ref={canvasRef}
        className={cn(
          "absolute pointer-events-none text-base transform scale-50 top-[20%] left-2 sm:left-4 origin-top-left pr-20",
          animating ? "opacity-100" : "opacity-0"
        )}
      />

      <input
        ref={inputRef}
        type="text"
        value={displayValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={cn(
          "w-full h-full bg-transparent border-none outline-none ring-0",
          "pl-4 sm:pl-5 text-base text-white z-50 relative",
          hideSubmitButton ? "pr-4" : "pr-14",
          animating && "text-transparent"
        )}
      />

      {/* animated placeholder */}
      <div className="absolute inset-0 flex items-center pointer-events-none">
        <AnimatePresence mode="wait">
          {!displayValue && (
            <motion.span
              key={`ph-${currentPlaceholder}`}
              initial={{ y: 10, opacity: 0, filter: "blur(4px)" }}
              animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
              exit={{ y: -10, opacity: 0, filter: "blur(4px)" }}
              transition={{ duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
              className={cn(
                "text-sm sm:text-base font-normal text-white/30 truncate",
                "pl-4 sm:pl-5",
                hideSubmitButton ? "pr-4" : "pr-14"
              )}
            >
              {placeholders[currentPlaceholder]}
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* submit button */}
      {!hideSubmitButton && (
        <button
          type="submit"
          disabled={!displayValue || disabled}
          className={cn(
            "absolute right-2 top-1/2 -translate-y-1/2 z-50",
            "h-8 w-8 rounded-lg flex items-center justify-center",
            "transition-all duration-200",
            "disabled:opacity-30 disabled:cursor-not-allowed",
            "bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.08]"
          )}
        >
          {submitIcon ?? (
            <motion.svg
              xmlns="http://www.w3.org/2000/svg"
              width="16" height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-white/60"
            >
              <path stroke="none" d="M0 0h24v24H0z" fill="none" />
              <motion.path
                d="M5 12l14 0"
                initial={{ strokeDasharray: "50%", strokeDashoffset: "50%" }}
                animate={{ strokeDashoffset: displayValue ? 0 : "50%" }}
                transition={{ duration: 0.3, ease: "linear" }}
              />
              <path d="M13 18l6 -6" />
              <path d="M13 6l6 6" />
            </motion.svg>
          )}
        </button>
      )}
    </form>
  );
}
