"use client";

import { useEffect, useRef } from "react";

interface Neuron {
  x: number;
  y: number;
  vx: number;
  vy: number;
  soma: number;
  hue: number;
  phase: number;
  pulseSpeed: number;
}

interface Signal {
  from: number;
  to: number;
  t: number;
  speed: number;
  hue: number;
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
    const NEURON_COUNT = 80;
    const CONNECT_DIST = 200;
    const SIGNAL_CHANCE = 0.003;
    const neurons: Neuron[] = [];
    const signals: Signal[] = [];

    function resize() {
      w = canvas!.width = window.innerWidth;
      h = canvas!.height = window.innerHeight;
    }

    function seed() {
      resize();
      neurons.length = 0;
      for (let i = 0; i < NEURON_COUNT; i++) {
        neurons.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.35,
          soma: 2.5 + Math.random() * 2,
          hue: 220 + Math.random() * 80,
          phase: Math.random() * Math.PI * 2,
          pulseSpeed: 0.015 + Math.random() * 0.02,
        });
      }
    }

    function drawDendrite(
      ax: number,
      ay: number,
      bx: number,
      by: number,
      alpha: number,
      hueA: number,
      hueB: number,
      width: number
    ) {
      const mx = (ax + bx) / 2;
      const my = (ay + by) / 2;
      const dx = bx - ax;
      const dy = by - ay;
      const perpX = -dy * 0.12;
      const perpY = dx * 0.12;
      const cpx = mx + perpX;
      const cpy = my + perpY;

      const grad = ctx!.createLinearGradient(ax, ay, bx, by);
      grad.addColorStop(0, `hsla(${hueA}, 70%, 60%, ${alpha})`);
      grad.addColorStop(0.5, `hsla(${(hueA + hueB) / 2}, 60%, 55%, ${alpha * 0.7})`);
      grad.addColorStop(1, `hsla(${hueB}, 70%, 60%, ${alpha})`);

      ctx!.beginPath();
      ctx!.moveTo(ax, ay);
      ctx!.quadraticCurveTo(cpx, cpy, bx, by);
      ctx!.strokeStyle = grad;
      ctx!.lineWidth = width;
      ctx!.stroke();
    }

    function drawSoma(n: Neuron, pulse: number) {
      const r = Math.max(n.soma * pulse, 0.5);
      const glowR = r * 5;

      const glow = ctx!.createRadialGradient(n.x, n.y, 0, n.x, n.y, glowR);
      glow.addColorStop(0, `hsla(${n.hue}, 80%, 70%, 0.5)`);
      glow.addColorStop(0.3, `hsla(${n.hue}, 70%, 55%, 0.12)`);
      glow.addColorStop(1, `hsla(${n.hue}, 70%, 50%, 0)`);
      ctx!.fillStyle = glow;
      ctx!.beginPath();
      ctx!.arc(n.x, n.y, glowR, 0, Math.PI * 2);
      ctx!.fill();

      const inner = ctx!.createRadialGradient(n.x, n.y, 0, n.x, n.y, r);
      inner.addColorStop(0, `hsla(${n.hue}, 90%, 85%, 0.95)`);
      inner.addColorStop(0.6, `hsla(${n.hue}, 80%, 65%, 0.8)`);
      inner.addColorStop(1, `hsla(${n.hue}, 70%, 50%, 0.4)`);
      ctx!.fillStyle = inner;
      ctx!.beginPath();
      ctx!.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx!.fill();
    }

    function drawSignal(sig: Signal) {
      const a = neurons[sig.from];
      const b = neurons[sig.to];
      if (!a || !b) return;

      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const cpx = mx + (-dy * 0.12);
      const cpy = my + (dx * 0.12);

      const t = sig.t;
      const sx = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * cpx + t * t * b.x;
      const sy = (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * cpy + t * t * b.y;

      const pulseR = 4 + Math.sin(t * Math.PI) * 3;
      const glow = ctx!.createRadialGradient(sx, sy, 0, sx, sy, pulseR * 3);
      glow.addColorStop(0, `hsla(${sig.hue}, 90%, 80%, 0.9)`);
      glow.addColorStop(0.4, `hsla(${sig.hue}, 80%, 60%, 0.3)`);
      glow.addColorStop(1, `hsla(${sig.hue}, 70%, 50%, 0)`);
      ctx!.fillStyle = glow;
      ctx!.beginPath();
      ctx!.arc(sx, sy, pulseR * 3, 0, Math.PI * 2);
      ctx!.fill();

      ctx!.fillStyle = `hsla(${sig.hue}, 95%, 90%, 1)`;
      ctx!.beginPath();
      ctx!.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx!.fill();
    }

    function tick() {
      frame++;

      ctx!.fillStyle = "rgba(6, 6, 18, 0.15)";
      ctx!.fillRect(0, 0, w, h);

      for (let i = 0; i < neurons.length; i++) {
        const n = neurons[i];
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < -20) n.x = w + 20;
        if (n.x > w + 20) n.x = -20;
        if (n.y < -20) n.y = h + 20;
        if (n.y > h + 20) n.y = -20;
      }

      for (let i = 0; i < neurons.length; i++) {
        const a = neurons[i];
        for (let j = i + 1; j < neurons.length; j++) {
          const b = neurons[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DIST) {
            const strength = 1 - dist / CONNECT_DIST;
            const alpha = strength * 0.28;
            drawDendrite(a.x, a.y, b.x, b.y, alpha, a.hue, b.hue, strength * 1.8);

            if (Math.random() < SIGNAL_CHANCE * strength) {
              signals.push({
                from: i,
                to: j,
                t: 0,
                speed: 0.008 + Math.random() * 0.012,
                hue: (a.hue + b.hue) / 2 + (Math.random() - 0.5) * 20,
              });
            }
          }
        }
      }

      for (let i = signals.length - 1; i >= 0; i--) {
        const sig = signals[i];
        sig.t += sig.speed;
        if (sig.t >= 1) {
          signals.splice(i, 1);
        } else {
          drawSignal(sig);
        }
      }

      for (const n of neurons) {
        const pulse = 0.85 + Math.sin(frame * n.pulseSpeed + n.phase) * 0.15;
        drawSoma(n, pulse);
      }

      animRef.current = requestAnimationFrame(tick);
    }

    seed();
    ctx.fillStyle = "rgb(6, 6, 18)";
    ctx.fillRect(0, 0, w, h);
    tick();

    const handleResize = () => {
      resize();
      ctx!.fillStyle = "rgb(6, 6, 18)";
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
