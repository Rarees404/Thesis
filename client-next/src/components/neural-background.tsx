"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

const NEURON_COUNT = 120;
const CONNECTION_DIST = 2.8;
const SIGNAL_CHANCE = 0.002;
const PULSE_SPEED = 0.6;
const CAMERA_DIST = 9;
const DRIFT_SPEED = 0.00012;

interface NeuronNode {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  phase: number;
  pulseSpeed: number;
  baseScale: number;
}

interface Signal {
  from: number;
  to: number;
  t: number;
  speed: number;
}

export function NeuralBackground() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.06);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0, CAMERA_DIST);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 1);
    container.appendChild(renderer.domElement);

    const neurons: NeuronNode[] = [];
    const signals: Signal[] = [];

    const somaGeometry = new THREE.SphereGeometry(1, 16, 12);
    const somaMaterial = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.9 });
    const somaMeshes: THREE.Mesh[] = [];

    const glowGeometry = new THREE.SphereGeometry(1, 12, 8);
    const glowMaterial = new THREE.MeshBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0.15, side: THREE.BackSide });
    const glowMeshes: THREE.Mesh[] = [];

    const spread = 7;
    for (let i = 0; i < NEURON_COUNT; i++) {
      const pos = new THREE.Vector3(
        (Math.random() - 0.5) * spread * 2,
        (Math.random() - 0.5) * spread * 2,
        (Math.random() - 0.5) * spread * 2
      );
      const vel = new THREE.Vector3(
        (Math.random() - 0.5) * DRIFT_SPEED,
        (Math.random() - 0.5) * DRIFT_SPEED,
        (Math.random() - 0.5) * DRIFT_SPEED
      );

      neurons.push({
        position: pos,
        velocity: vel,
        phase: Math.random() * Math.PI * 2,
        pulseSpeed: 0.01 + Math.random() * 0.015,
        baseScale: 0.04 + Math.random() * 0.03,
      });

      const soma = new THREE.Mesh(somaGeometry, somaMaterial.clone());
      soma.position.copy(pos);
      soma.scale.setScalar(neurons[i].baseScale);
      scene.add(soma);
      somaMeshes.push(soma);

      const glow = new THREE.Mesh(glowGeometry, glowMaterial.clone());
      glow.position.copy(pos);
      glow.scale.setScalar(neurons[i].baseScale * 4);
      scene.add(glow);
      glowMeshes.push(glow);
    }

    const linePositions = new Float32Array(NEURON_COUNT * NEURON_COUNT * 6);
    const lineColors = new Float32Array(NEURON_COUNT * NEURON_COUNT * 6);
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
    const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4 });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    const signalGeometry = new THREE.SphereGeometry(0.03, 8, 6);
    const signalMaterial = new THREE.MeshBasicMaterial({ color: 0xc4b5fd, transparent: true, opacity: 0.95 });
    const signalPool: THREE.Mesh[] = [];
    const MAX_SIGNALS = 80;
    for (let i = 0; i < MAX_SIGNALS; i++) {
      const m = new THREE.Mesh(signalGeometry, signalMaterial.clone());
      m.visible = false;
      scene.add(m);
      signalPool.push(m);
    }

    let frame = 0;
    let animId = 0;
    const clock = new THREE.Clock();

    function animate() {
      animId = requestAnimationFrame(animate);
      const dt = clock.getDelta();
      frame++;

      const time = frame * 0.01;
      camera.position.x = Math.sin(time * 0.15) * 1.5;
      camera.position.y = Math.cos(time * 0.1) * 1.0;
      camera.lookAt(0, 0, 0);

      let lineIdx = 0;
      for (let i = 0; i < NEURON_COUNT; i++) {
        const n = neurons[i];
        n.position.add(n.velocity);

        const bound = spread;
        if (n.position.x < -bound || n.position.x > bound) n.velocity.x *= -1;
        if (n.position.y < -bound || n.position.y > bound) n.velocity.y *= -1;
        if (n.position.z < -bound || n.position.z > bound) n.velocity.z *= -1;

        const pulse = 0.85 + Math.sin(frame * n.pulseSpeed + n.phase) * 0.15;
        somaMeshes[i].position.copy(n.position);
        somaMeshes[i].scale.setScalar(n.baseScale * pulse);
        glowMeshes[i].position.copy(n.position);
        glowMeshes[i].scale.setScalar(n.baseScale * pulse * 5);

        for (let j = i + 1; j < NEURON_COUNT; j++) {
          const dist = n.position.distanceTo(neurons[j].position);
          if (dist < CONNECTION_DIST) {
            const strength = 1 - dist / CONNECTION_DIST;
            const idx = lineIdx * 6;

            linePositions[idx] = n.position.x;
            linePositions[idx + 1] = n.position.y;
            linePositions[idx + 2] = n.position.z;
            linePositions[idx + 3] = neurons[j].position.x;
            linePositions[idx + 4] = neurons[j].position.y;
            linePositions[idx + 5] = neurons[j].position.z;

            const r = 0.35 * strength;
            const g = 0.22 * strength;
            const b = 0.85 * strength;
            lineColors[idx] = r;
            lineColors[idx + 1] = g;
            lineColors[idx + 2] = b;
            lineColors[idx + 3] = r;
            lineColors[idx + 4] = g;
            lineColors[idx + 5] = b;

            lineIdx++;

            if (Math.random() < SIGNAL_CHANCE * strength && signals.length < MAX_SIGNALS) {
              signals.push({
                from: i,
                to: j,
                t: 0,
                speed: PULSE_SPEED + Math.random() * 0.4,
              });
            }
          }
        }
      }

      lineGeometry.setDrawRange(0, lineIdx * 2);
      (lineGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
      (lineGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;

      signalPool.forEach((m) => (m.visible = false));

      for (let i = signals.length - 1; i >= 0; i--) {
        const sig = signals[i];
        sig.t += dt * sig.speed;
        if (sig.t >= 1) {
          signals.splice(i, 1);
          continue;
        }

        if (i < MAX_SIGNALS) {
          const a = neurons[sig.from].position;
          const b = neurons[sig.to].position;
          const mesh = signalPool[i];
          mesh.visible = true;
          mesh.position.lerpVectors(a, b, sig.t);
          const pulseScale = 1 + Math.sin(sig.t * Math.PI) * 1.5;
          mesh.scale.setScalar(pulseScale);
          (mesh.material as THREE.MeshBasicMaterial).opacity = Math.sin(sig.t * Math.PI) * 0.95;
        }
      }

      renderer.render(scene, camera);
    }

    animate();

    function handleResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
    window.addEventListener("resize", handleResize);

    cleanupRef.current = () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animId);
      renderer.dispose();
      somaGeometry.dispose();
      somaMaterial.dispose();
      glowGeometry.dispose();
      glowMaterial.dispose();
      signalGeometry.dispose();
      signalMaterial.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };

    return () => {
      cleanupRef.current?.();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="pointer-events-none fixed inset-0 z-0"
      aria-hidden="true"
    />
  );
}
