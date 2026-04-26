"use client";

import { useMemo, useState, useEffect } from "react";
import {
  Cpu,
  MemoryStick,
  HardDrive,
  Wifi,
  WifiOff,
  Server,
  Database,
  Clock,
  RefreshCw,
  Activity,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import {
  XAxis,
  YAxis,
  Tooltip as RTooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
} from "recharts";
import { useMetricsStore } from "@/lib/metrics-store";
import type { MetricsDataPoint } from "@/lib/types";

const ACCENT = "#6f8cf2"; // primary blue
const ACCENT_2 = "#8ad5c2"; // teal
const WARN = "#e0b96a"; // amber

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function StatusDot({ status }: { status: "healthy" | "error" | "connecting" }) {
  const color =
    status === "healthy"
      ? "bg-emerald-400"
      : status === "connecting"
        ? "bg-amber-400"
        : "bg-rose-400";
  return (
    <span className="relative inline-flex h-2 w-2">
      {status === "healthy" && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/60 opacity-75" />
      )}
      <span className={`relative inline-flex h-2 w-2 rounded-full ${color}`} />
    </span>
  );
}

function Trend({
  data,
  field,
}: {
  data: MetricsDataPoint[];
  field: keyof MetricsDataPoint;
}) {
  const recent = data.slice(-6);
  const values = recent.map((d) => d[field] as number).filter((v) => v > 0);
  if (values.length < 2)
    return <Minus className="h-3 w-3 text-muted-foreground/40" />;
  const diff = values[values.length - 1] - values[0];
  if (Math.abs(diff) < 1)
    return <Minus className="h-3 w-3 text-muted-foreground/40" />;
  return diff > 0 ? (
    <TrendingUp className="h-3 w-3 text-amber-400/80" />
  ) : (
    <TrendingDown className="h-3 w-3 text-emerald-400/80" />
  );
}

function MetricTile({
  icon: Icon,
  title,
  value,
  subtitle,
  percent,
  sparkData,
  sparkKey,
}: {
  icon: React.ElementType;
  title: string;
  value: string;
  subtitle?: string;
  percent?: number;
  sparkData: MetricsDataPoint[];
  sparkKey: keyof MetricsDataPoint;
}) {
  const last12 = sparkData.slice(-12);
  const barColor =
    percent !== undefined && percent > 80
      ? "#e07b6b"
      : percent !== undefined && percent > 60
        ? WARN
        : ACCENT;

  return (
    <div className="rounded-md border border-border bg-card p-4">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5">
            <Icon className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-muted-foreground">
              {title}
            </span>
            <Trend data={sparkData} field={sparkKey} />
          </div>
          <p className="font-mono text-[22px] font-medium tabular-nums tracking-tight text-foreground">
            {value}
          </p>
          {subtitle && (
            <p className="text-[11px] text-muted-foreground">{subtitle}</p>
          )}
        </div>
      </div>

      {/* Sparkline */}
      <div className="mt-2 h-8">
        <ResponsiveContainer width="100%" height={32}>
          <AreaChart
            data={last12}
            margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
          >
            <defs>
              <linearGradient id={`spark-${title}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={ACCENT} stopOpacity={0.25} />
                <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey={sparkKey as string}
              stroke={ACCENT}
              fill={`url(#spark-${title})`}
              strokeWidth={1.25}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {percent !== undefined && (
        <div className="mt-2">
          <div className="flex justify-between font-mono text-[10px] tabular-nums text-muted-foreground">
            <span>load</span>
            <span>{percent.toFixed(1)}%</span>
          </div>
          <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full transition-[width] duration-500 ease-out"
              style={{
                width: `${Math.min(percent, 100)}%`,
                backgroundColor: barColor,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ChartTooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-popover px-2.5 py-1.5 shadow-xl">
      <p className="mb-1 font-mono text-[10px] text-muted-foreground">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-[11px]">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: p.color }}
          />
          <span className="text-muted-foreground">{p.name}</span>
          <span className="ml-auto font-mono font-medium tabular-nums text-foreground">
            {p.value.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function ServerDashboard() {
  const [mounted, setMounted] = useState(false);
  const [secondsAgo, setSecondsAgo] = useState(0);

  useEffect(() => {
    const id = window.setTimeout(() => setMounted(true), 0);
    return () => window.clearTimeout(id);
  }, []);

  const metrics = useMetricsStore((s) => s.metrics);
  const history = useMetricsStore((s) => s.history);
  const status = useMetricsStore((s) => s.status);
  const lastError = useMetricsStore((s) => s.lastError);
  const lastPollTime = useMetricsStore((s) => s.lastPollTime);
  const polling = useMetricsStore((s) => s.polling);
  const setPolling = useMetricsStore((s) => s.setPolling);
  const poll = useMetricsStore((s) => s.poll);

  // Count up seconds since last successful poll so the audience sees live activity
  useEffect(() => {
    setSecondsAgo(0);
    if (!polling) return;
    const id = setInterval(() => setSecondsAgo((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [lastPollTime, polling]);

  const cpuAvg = useMemo(() => {
    const live = history.filter((d) => d.cpu > 0);
    return live.length ? live.reduce((s, d) => s + d.cpu, 0) / live.length : 0;
  }, [history]);

  const memAvg = useMemo(() => {
    const live = history.filter((d) => d.memory > 0);
    return live.length
      ? live.reduce((s, d) => s + d.memory, 0) / live.length
      : 0;
  }, [history]);

  return (
    <div className="space-y-5">
      {/* Status bar */}
      <div className="rounded-md border border-border bg-card">
        <div className="flex flex-col gap-3 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <StatusDot status={status} />
            <div className="space-y-0.5">
              <p className="text-sm font-medium tracking-tight text-foreground">
                {status === "healthy"
                  ? "Server online"
                  : status === "connecting"
                    ? "Connecting"
                    : "Server unreachable"}
              </p>
              {metrics && (
                <p className="font-mono text-[11px] text-muted-foreground">
                  {metrics.system.hostname} · {metrics.system.platform}{" "}
                  {metrics.system.architecture} · Python{" "}
                  {metrics.system.python_version}
                </p>
              )}
              {lastError && (
                <p className="font-mono text-[11px] text-destructive">
                  {lastError}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
            {lastPollTime && (
              <span className="inline-flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 tabular-nums">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {lastPollTime}
                {secondsAgo > 0 && (
                  <span className="text-muted-foreground/60">
                    +{secondsAgo}s
                  </span>
                )}
              </span>
            )}
            {metrics && (
              <span className="inline-flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 tabular-nums">
                <Clock className="h-3 w-3" />
                {formatUptime(metrics.uptime_seconds)}
              </span>
            )}
            {metrics?.model.loaded && (
              <span className="inline-flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 tabular-nums">
                <Database className="h-3 w-3" />
                {metrics.model.index_size.toLocaleString()} vec
              </span>
            )}
            <button
              onClick={() => setPolling(!polling)}
              className="inline-flex items-center gap-1.5 rounded border border-border bg-background px-2 py-1 transition-colors duration-100 hover:text-foreground"
            >
              {polling ? (
                <WifiOff className="h-3 w-3" />
              ) : (
                <Wifi className="h-3 w-3" />
              )}
              {polling ? "pause" : "resume"}
            </button>
            <button
              onClick={poll}
              className="inline-flex items-center justify-center rounded border border-border bg-background p-1.5 transition-colors duration-100 hover:text-foreground"
              aria-label="Refresh"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Metric tiles */}
      {mounted && metrics && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <MetricTile
            icon={Cpu}
            title="CPU"
            value={`${metrics.cpu.percent.toFixed(1)}%`}
            subtitle={`${metrics.cpu.count_physical} cores · ${metrics.cpu.count} threads`}
            percent={metrics.cpu.percent}
            sparkData={history}
            sparkKey="cpu"
          />
          <MetricTile
            icon={MemoryStick}
            title="Memory"
            value={`${(metrics.memory.used_mb / 1024).toFixed(1)} GB`}
            subtitle={`of ${(metrics.memory.total_mb / 1024).toFixed(1)} GB`}
            percent={metrics.memory.percent}
            sparkData={history}
            sparkKey="memory"
          />
          <MetricTile
            icon={Zap}
            title={metrics.gpu.backend === "mps" ? "GPU (MPS)" : "GPU"}
            value={
              metrics.gpu.available
                ? `${metrics.gpu.memory_used_mb.toFixed(0)} MB`
                : "n/a"
            }
            subtitle={
              metrics.gpu.backend === "mps"
                ? "unified memory"
                : metrics.gpu.available
                  ? `of ${metrics.gpu.memory_total_mb.toFixed(0)} MB`
                  : "no GPU detected"
            }
            percent={
              metrics.gpu.available ? metrics.gpu.utilization_pct : undefined
            }
            sparkData={history}
            sparkKey="gpu"
          />
          <MetricTile
            icon={HardDrive}
            title="Disk"
            value={`${metrics.disk.used_gb.toFixed(0)} GB`}
            subtitle={`of ${metrics.disk.total_gb.toFixed(0)} GB`}
            percent={metrics.disk.percent}
            sparkData={history}
            sparkKey="memory"
          />
        </div>
      )}

      {/* Main charts */}
      {mounted && (
        <div className="grid gap-3 lg:grid-cols-2">
          <ChartCard
            title="CPU & Memory"
            icon={Activity}
            polling={polling}
            legend={[
              { color: ACCENT, label: "CPU" },
              { color: ACCENT_2, label: "RAM" },
            ]}
          >
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart
                data={history}
                margin={{ top: 5, right: 8, bottom: 0, left: -18 }}
              >
                <defs>
                  <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={ACCENT} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={ACCENT_2} stopOpacity={0.2} />
                    <stop offset="100%" stopColor={ACCENT_2} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.35)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.35)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <ReferenceLine
                  y={cpuAvg}
                  stroke={ACCENT}
                  strokeDasharray="4 3"
                  strokeOpacity={0.25}
                />
                <ReferenceLine
                  y={memAvg}
                  stroke={ACCENT_2}
                  strokeDasharray="4 3"
                  strokeOpacity={0.25}
                />
                <Area
                  type="monotone"
                  dataKey="cpu"
                  stroke={ACCENT}
                  fill="url(#cpuGrad)"
                  strokeWidth={1.25}
                  name="CPU"
                  dot={false}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="memory"
                  stroke={ACCENT_2}
                  fill="url(#memGrad)"
                  strokeWidth={1.25}
                  name="RAM"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title={
              metrics?.gpu.backend === "mps" ? "GPU (MPS)" : "GPU Utilization"
            }
            icon={Zap}
            polling={polling}
            legend={[
              {
                color: WARN,
                label: metrics?.gpu.backend === "mps" ? "P-core %" : "compute",
              },
              {
                color: "#c97c75",
                label: metrics?.gpu.backend === "mps" ? "memory" : "vram",
                dashed: true,
              },
            ]}
            note={
              metrics?.gpu.backend === "mps"
                ? "Apple Silicon · CPU/GPU share unified memory"
                : undefined
            }
          >
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart
                data={history}
                margin={{ top: 5, right: 8, bottom: 0, left: -18 }}
              >
                <defs>
                  <linearGradient id="gpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={WARN} stopOpacity={0.22} />
                    <stop offset="100%" stopColor={WARN} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gpuMemGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#c97c75" stopOpacity={0.12} />
                    <stop offset="100%" stopColor="#c97c75" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.35)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.35)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <Area
                  type="monotone"
                  dataKey="gpu"
                  stroke={WARN}
                  fill="url(#gpuGrad)"
                  strokeWidth={1.25}
                  name="GPU Util"
                  dot={false}
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="gpuMem"
                  stroke="#c97c75"
                  fill="url(#gpuMemGrad)"
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  name="GPU VRAM"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {/* System info table */}
      {metrics && (
        <div className="rounded-md border border-border bg-card">
          <header className="flex items-center gap-2 border-b border-border px-5 py-3">
            <Server className="h-3.5 w-3.5 text-muted-foreground" />
            <h3 className="text-sm font-medium tracking-tight text-foreground">
              System
            </h3>
          </header>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 px-5 py-4 text-[12.5px] sm:grid-cols-4">
            <InfoRow
              label="Platform"
              value={`${metrics.system.platform} ${metrics.system.architecture}`}
            />
            <InfoRow label="Hostname" value={metrics.system.hostname} />
            <InfoRow label="Python" value={metrics.system.python_version} />
            <InfoRow
              label="Network"
              value={`↑${metrics.network.bytes_sent_mb.toFixed(0)} MB · ↓${metrics.network.bytes_recv_mb.toFixed(0)} MB`}
            />
            <InfoRow
              label="GPU"
              value={metrics.gpu.available ? metrics.gpu.name ?? "—" : "CPU"}
            />
            <InfoRow
              label="FAISS index"
              value={
                metrics.model.loaded
                  ? `${metrics.model.index_size.toLocaleString()} vectors`
                  : "not loaded"
              }
            />
            <InfoRow
              label="Model"
              value={
                metrics.model.loaded ? (
                  <span className="text-emerald-400">loaded</span>
                ) : (
                  <span className="text-rose-400">not loaded</span>
                )
              }
            />
            <InfoRow label="Uptime" value={formatUptime(metrics.uptime_seconds)} />
          </div>
        </div>
      )}
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="space-y-0.5">
      <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="tracking-tight text-foreground">{value}</p>
    </div>
  );
}

function ChartCard({
  title,
  icon: Icon,
  polling,
  legend,
  note,
  children,
}: {
  title: string;
  icon: React.ElementType;
  polling: boolean;
  legend: { color: string; label: string; dashed?: boolean }[];
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border bg-card">
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-2">
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
          <h3 className="text-sm font-medium tracking-tight text-foreground">
            {title}
          </h3>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10.5px] text-muted-foreground">
          {legend.map((l) => (
            <span key={l.label} className="flex items-center gap-1.5">
              {l.dashed ? (
                <span
                  className="h-px w-4 border-t border-dashed"
                  style={{ borderColor: l.color }}
                />
              ) : (
                <span
                  className="h-1 w-4 rounded-full"
                  style={{ backgroundColor: l.color }}
                />
              )}
              {l.label}
            </span>
          ))}
          {polling && (
            <span className="flex items-center gap-1 text-emerald-400/80">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              live
            </span>
          )}
        </div>
      </header>
      {note && (
        <p className="px-5 pt-2 font-mono text-[10.5px] text-muted-foreground">
          {note}
        </p>
      )}
      <div className="px-2 pb-3 pt-2">{children}</div>
    </div>
  );
}
