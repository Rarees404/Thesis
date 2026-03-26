"use client";

import { useEffect, useRef } from "react";

interface Blip {
  x: number;
  y: number;
  age: number;
  maxAge: number;
  size: number;
}

export function NeuralBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let frame = 0;
    const blips: Blip[] = [];
    const GRID_SPACING = 60;
    const BLIP_CHANCE = 0.018;
    const MAX_BLIPS = 30;

    function resize() {
      w = canvas!.width = window.innerWidth;
      h = canvas!.height = window.innerHeight;
    }

    resize();

    function drawGrid() {
      ctx!.strokeStyle = "rgba(220, 38, 38, 0.09)";
      ctx!.lineWidth = 0.5;

      for (let x = 0; x < w; x += GRID_SPACING) {
        ctx!.beginPath();
        ctx!.moveTo(x, 0);
        ctx!.lineTo(x, h);
        ctx!.stroke();
      }
      for (let y = 0; y < h; y += GRID_SPACING) {
        ctx!.beginPath();
        ctx!.moveTo(0, y);
        ctx!.lineTo(w, y);
        ctx!.stroke();
      }

      ctx!.fillStyle = "rgba(220, 38, 38, 0.15)";
      for (let x = 0; x < w; x += GRID_SPACING) {
        for (let y = 0; y < h; y += GRID_SPACING) {
          ctx!.beginPath();
          ctx!.arc(x, y, 1.2, 0, Math.PI * 2);
          ctx!.fill();
        }
      }
    }

    function drawRadar() {
      const cx = w / 2;
      const cy = h / 2;
      const maxR = Math.max(w, h) * 0.7;
      const sweepAngle = (frame * 0.005) % (Math.PI * 2);

      // Concentric rings
      for (let r = 100; r < maxR; r += 140) {
        ctx!.beginPath();
        ctx!.arc(cx, cy, r, 0, Math.PI * 2);
        const ringAlpha = Math.max(0.02, 0.09 - r * 0.00004);
        ctx!.strokeStyle = `rgba(220, 38, 38, ${ringAlpha})`;
        ctx!.lineWidth = 0.8;
        ctx!.stroke();
      }

      // Center crosshair
      const chSize = 20;
      ctx!.strokeStyle = "rgba(220, 38, 38, 0.2)";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(cx - chSize, cy);
      ctx!.lineTo(cx + chSize, cy);
      ctx!.stroke();
      ctx!.beginPath();
      ctx!.moveTo(cx, cy - chSize);
      ctx!.lineTo(cx, cy + chSize);
      ctx!.stroke();

      // Center dot
      ctx!.fillStyle = "rgba(220, 38, 38, 0.3)";
      ctx!.beginPath();
      ctx!.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx!.fill();

      // Sweep trail (conic gradient)
      const grad = ctx!.createConicGradient(sweepAngle, cx, cy);
      grad.addColorStop(0, "rgba(220, 38, 38, 0.18)");
      grad.addColorStop(0.06, "rgba(220, 38, 38, 0.09)");
      grad.addColorStop(0.14, "rgba(220, 38, 38, 0.02)");
      grad.addColorStop(0.2, "rgba(220, 38, 38, 0)");
      grad.addColorStop(1, "rgba(220, 38, 38, 0)");

      ctx!.beginPath();
      ctx!.moveTo(cx, cy);
      ctx!.arc(cx, cy, maxR, sweepAngle - 0.7, sweepAngle);
      ctx!.closePath();
      ctx!.fillStyle = grad;
      ctx!.fill();

      // Sweep leading edge line
      ctx!.beginPath();
      ctx!.moveTo(cx, cy);
      const endX = cx + Math.cos(sweepAngle) * maxR;
      const endY = cy + Math.sin(sweepAngle) * maxR;
      ctx!.lineTo(endX, endY);
      ctx!.strokeStyle = "rgba(220, 38, 38, 0.3)";
      ctx!.lineWidth = 1.5;
      ctx!.stroke();
    }

    function drawBlips() {
      if (Math.random() < BLIP_CHANCE && blips.length < MAX_BLIPS) {
        blips.push({
          x: Math.random() * w,
          y: Math.random() * h,
          age: 0,
          maxAge: 150 + Math.random() * 200,
          size: 2.5 + Math.random() * 4,
        });
      }

      for (let i = blips.length - 1; i >= 0; i--) {
        const b = blips[i];
        b.age++;
        if (b.age > b.maxAge) {
          blips.splice(i, 1);
          continue;
        }

        const life = b.age / b.maxAge;
        const alpha = life < 0.1 ? life / 0.1 : life > 0.7 ? (1 - life) / 0.3 : 1;
        const pulse = 1 + Math.sin(b.age * 0.1) * 0.3;

        // Outer glow
        const glow = ctx!.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.size * 5 * pulse);
        glow.addColorStop(0, `rgba(220, 38, 38, ${0.6 * alpha})`);
        glow.addColorStop(0.3, `rgba(220, 38, 38, ${0.2 * alpha})`);
        glow.addColorStop(0.7, `rgba(220, 38, 38, ${0.05 * alpha})`);
        glow.addColorStop(1, "rgba(220, 38, 38, 0)");
        ctx!.fillStyle = glow;
        ctx!.beginPath();
        ctx!.arc(b.x, b.y, b.size * 5 * pulse, 0, Math.PI * 2);
        ctx!.fill();

        // Core
        ctx!.fillStyle = `rgba(255, 120, 120, ${0.95 * alpha})`;
        ctx!.beginPath();
        ctx!.arc(b.x, b.y, b.size * 0.7, 0, Math.PI * 2);
        ctx!.fill();

        // Ring around blip
        ctx!.strokeStyle = `rgba(220, 38, 38, ${0.3 * alpha})`;
        ctx!.lineWidth = 0.8;
        ctx!.beginPath();
        ctx!.arc(b.x, b.y, b.size * 2 * pulse, 0, Math.PI * 2);
        ctx!.stroke();
      }
    }

    function drawCoordinates() {
      ctx!.font = "9px monospace";
      ctx!.fillStyle = "rgba(220, 38, 38, 0.15)";

      for (let x = GRID_SPACING * 3; x < w; x += GRID_SPACING * 3) {
        ctx!.fillText(`${x}`, x + 2, 12);
      }
      for (let y = GRID_SPACING * 3; y < h; y += GRID_SPACING * 3) {
        ctx!.fillText(`${y}`, 4, y - 2);
      }
    }

    function tick() {
      frame++;

      // Slower fade = longer trails, more visible elements
      ctx!.fillStyle = "rgba(5, 5, 5, 0.08)";
      ctx!.fillRect(0, 0, w, h);

      if (frame % 2 === 0) {
        drawGrid();
      }
      drawRadar();
      drawBlips();

      if (frame % 8 === 0) {
        drawCoordinates();
      }

      animRef.current = requestAnimationFrame(tick);
    }

    ctx.fillStyle = "rgb(5, 5, 5)";
    ctx.fillRect(0, 0, w, h);
    tick();

    const handleResize = () => {
      resize();
      ctx!.fillStyle = "rgb(5, 5, 5)";
      ctx!.fillRect(0, 0, w, h);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0"
    />
  );
}
