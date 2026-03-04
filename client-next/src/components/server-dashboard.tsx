"use client";

import { useMemo } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Cpu,
  MemoryStick,
  HardDrive,
  Wifi,
  WifiOff,
  Monitor,
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
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
} from "recharts";
import { useMetricsStore } from "@/lib/metrics-store";
import type { MetricsDataPoint } from "@/lib/types";

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
      ? "bg-emerald-500"
      : status === "connecting"
        ? "bg-yellow-500"
        : "bg-red-500";
  return (
    <span className="relative flex h-3 w-3">
      {status === "healthy" && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
      )}
      <span className={`relative inline-flex h-3 w-3 rounded-full ${color}`} />
    </span>
  );
}

function Trend({ data, field }: { data: MetricsDataPoint[]; field: keyof MetricsDataPoint }) {
  const recent = data.slice(-6);
  const values = recent.map((d) => d[field] as number).filter((v) => v > 0);
  if (values.length < 2) return <Minus className="h-3 w-3 text-white/20" />;
  const diff = values[values.length - 1] - values[0];
  if (Math.abs(diff) < 1) return <Minus className="h-3 w-3 text-white/20" />;
  return diff > 0 ? (
    <TrendingUp className="h-3 w-3 text-amber-400" />
  ) : (
    <TrendingDown className="h-3 w-3 text-emerald-400" />
  );
}

function MetricCard({
  icon: Icon,
  title,
  value,
  subtitle,
  percent,
  color,
  sparkData,
  sparkKey,
}: {
  icon: React.ElementType;
  title: string;
  value: string;
  subtitle?: string;
  percent?: number;
  color: string;
  sparkData: MetricsDataPoint[];
  sparkKey: keyof MetricsDataPoint;
}) {
  const last12 = sparkData.slice(-12);

  return (
    <Card>
      <CardContent className="pt-4 pb-4">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-1.5">
              <p className="text-xs font-medium text-white/40">{title}</p>
              <Trend data={sparkData} field={sparkKey} />
            </div>
            <p className="text-2xl font-bold text-white">{value}</p>
            {subtitle && (
              <p className="text-[11px] text-white/30">{subtitle}</p>
            )}
          </div>
          <div
            className="rounded-lg p-2.5 border border-white/[0.06]"
            style={{ backgroundColor: `${color}12` }}
          >
            <Icon className="h-5 w-5" style={{ color }} />
          </div>
        </div>

        <div className="mt-2 h-8">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={last12} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`spark-${title}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey={sparkKey as string}
                stroke={color}
                fill={`url(#spark-${title})`}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {percent !== undefined && (
          <div className="mt-1">
            <div className="flex justify-between text-[10px] text-white/30 mb-1">
              <span>Usage</span>
              <span>{percent.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-white/[0.06] overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${Math.min(percent, 100)}%`,
                  backgroundColor: percent > 80 ? "#ef4444" : percent > 60 ? "#f59e0b" : color,
                  boxShadow: `0 0 8px ${percent > 80 ? "#ef4444" : percent > 60 ? "#f59e0b" : color}60`,
                }}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ChartTooltipContent({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/[0.08] bg-[rgba(10,10,25,0.92)] px-3 py-2 shadow-xl backdrop-blur-xl">
      <p className="text-[10px] text-white/40 mb-1">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-xs">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-white/60">{p.name}</span>
          <span className="ml-auto font-mono font-semibold text-white">
            {p.value.toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function ServerDashboard() {
  const metrics = useMetricsStore((s) => s.metrics);
  const history = useMetricsStore((s) => s.history);
  const status = useMetricsStore((s) => s.status);
  const lastError = useMetricsStore((s) => s.lastError);
  const polling = useMetricsStore((s) => s.polling);
  const setPolling = useMetricsStore((s) => s.setPolling);
  const poll = useMetricsStore((s) => s.poll);

  const cpuAvg = useMemo(() => {
    const live = history.filter((d) => d.cpu > 0);
    return live.length ? live.reduce((s, d) => s + d.cpu, 0) / live.length : 0;
  }, [history]);

  const memAvg = useMemo(() => {
    const live = history.filter((d) => d.memory > 0);
    return live.length ? live.reduce((s, d) => s + d.memory, 0) / live.length : 0;
  }, [history]);

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <Card>
        <CardContent className="py-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <StatusDot status={status} />
              <div>
                <p className="text-sm font-semibold text-white">
                  {status === "healthy"
                    ? "Server Online"
                    : status === "connecting"
                      ? "Connecting..."
                      : "Server Unreachable"}
                </p>
                {metrics && (
                  <p className="text-xs text-white/30">
                    {metrics.system.hostname} &middot; {metrics.system.platform}{" "}
                    {metrics.system.architecture} &middot; Python{" "}
                    {metrics.system.python_version}
                  </p>
                )}
                {lastError && (
                  <p className="text-xs text-red-400">{lastError}</p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {metrics && (
                <Badge variant="outline" className="gap-1 text-[10px] border-white/10 bg-white/5 text-white/60">
                  <Clock className="h-3 w-3" />
                  {formatUptime(metrics.uptime_seconds)}
                </Badge>
              )}
              {metrics?.model.loaded && (
                <Badge variant="secondary" className="gap-1 text-[10px] bg-white/5 text-white/60 border border-white/[0.06]">
                  <Database className="h-3 w-3" />
                  {metrics.model.index_size.toLocaleString()} images
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPolling(!polling)}
                className="gap-1.5 text-[11px] text-white/50 hover:text-white hover:bg-white/5"
              >
                {polling ? <WifiOff className="h-3.5 w-3.5" /> : <Wifi className="h-3.5 w-3.5" />}
                {polling ? "Pause" : "Resume"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={poll}
                className="h-8 w-8 p-0 text-white/50 hover:text-white hover:bg-white/5"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metric cards */}
      {metrics && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <MetricCard
            icon={Cpu}
            title="CPU Usage"
            value={`${metrics.cpu.percent.toFixed(1)}%`}
            subtitle={`${metrics.cpu.count_physical} cores / ${metrics.cpu.count} threads`}
            percent={metrics.cpu.percent}
            color="#818cf8"
            sparkData={history}
            sparkKey="cpu"
          />
          <MetricCard
            icon={MemoryStick}
            title="RAM Usage"
            value={`${(metrics.memory.used_mb / 1024).toFixed(1)} GB`}
            subtitle={`of ${(metrics.memory.total_mb / 1024).toFixed(1)} GB total`}
            percent={metrics.memory.percent}
            color="#22d3ee"
            sparkData={history}
            sparkKey="memory"
          />
          <MetricCard
            icon={Zap}
            title={
              metrics.gpu.backend === "mps"
                ? "Apple GPU (MPS)"
                : metrics.gpu.available
                  ? (metrics.gpu.name ?? "GPU")
                  : "GPU"
            }
            value={metrics.gpu.available ? `${metrics.gpu.memory_used_mb.toFixed(0)} MB` : "N/A"}
            subtitle={
              metrics.gpu.backend === "mps"
                ? "Unified memory (shared with CPU)"
                : metrics.gpu.available
                  ? `of ${metrics.gpu.memory_total_mb.toFixed(0)} MB`
                  : "No GPU detected"
            }
            percent={metrics.gpu.available ? metrics.gpu.utilization_pct : undefined}
            color="#fbbf24"
            sparkData={history}
            sparkKey="gpu"
          />
          <MetricCard
            icon={HardDrive}
            title="Disk Usage"
            value={`${metrics.disk.used_gb.toFixed(0)} GB`}
            subtitle={`of ${metrics.disk.total_gb.toFixed(0)} GB total`}
            percent={metrics.disk.percent}
            color="#34d399"
            sparkData={history}
            sparkKey="memory"
          />
        </div>
      )}

      {/* Main charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* CPU & Memory */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-white/80">
              <Activity className="h-4 w-4 text-indigo-400" />
              CPU & Memory
              <div className="ml-auto flex items-center gap-3">
                <span className="flex items-center gap-1.5 text-[10px] text-white/30">
                  <span className="h-2 w-5 rounded-sm bg-indigo-400/60" />
                  CPU
                </span>
                <span className="flex items-center gap-1.5 text-[10px] text-white/30">
                  <span className="h-2 w-5 rounded-sm bg-cyan-400/60" />
                  RAM
                </span>
                {polling && (
                  <span className="flex items-center gap-1 text-[10px] text-emerald-400/60">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={history} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                <defs>
                  <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#818cf8" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.2)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.2)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <ReferenceLine y={cpuAvg} stroke="#818cf8" strokeDasharray="6 4" strokeOpacity={0.3} />
                <ReferenceLine y={memAvg} stroke="#22d3ee" strokeDasharray="6 4" strokeOpacity={0.3} />
                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="2 6" strokeOpacity={0.2} label={{ value: "80%", position: "right", fill: "rgba(239,68,68,0.3)", fontSize: 9 }} />
                <Area type="basis" dataKey="cpu" stroke="#818cf8" fill="url(#cpuGrad)" strokeWidth={2} name="CPU" dot={false} isAnimationActive={false} />
                <Area type="basis" dataKey="memory" stroke="#22d3ee" fill="url(#memGrad)" strokeWidth={2} name="RAM" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* GPU */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-white/80">
              <Monitor className="h-4 w-4 text-amber-400" />
              {metrics?.gpu.backend === "mps" ? "Apple GPU (MPS)" : "GPU Utilization"}
              <div className="ml-auto flex items-center gap-3">
                <span className="flex items-center gap-1.5 text-[10px] text-white/30">
                  <span className="h-2 w-5 rounded-sm bg-amber-400/60" />
                  {metrics?.gpu.backend === "mps" ? "P-Core %" : "Compute"}
                </span>
                <span className="flex items-center gap-1.5 text-[10px] text-white/30">
                  <span className="h-0.5 w-5 border-t-2 border-dashed border-rose-400/60" />
                  {metrics?.gpu.backend === "mps" ? "Unified Mem" : "VRAM"}
                </span>
                {polling && (
                  <span className="flex items-center gap-1 text-[10px] text-emerald-400/60">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </CardTitle>
            {metrics?.gpu.backend === "mps" && (
              <p className="text-[10px] text-white/20 -mt-1">
                Apple Silicon shares CPU/GPU memory · P-core load used as compute proxy
              </p>
            )}
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={history} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                <defs>
                  <linearGradient id="gpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gpuMemGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f87171" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#f87171" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.2)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.2)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="2 6" strokeOpacity={0.2} label={{ value: "80%", position: "right", fill: "rgba(239,68,68,0.3)", fontSize: 9 }} />
                <Area type="basis" dataKey="gpu" stroke="#fbbf24" fill="url(#gpuGrad)" strokeWidth={2} name="GPU Util" dot={false} isAnimationActive={false} />
                <Area type="basis" dataKey="gpuMem" stroke="#f87171" fill="url(#gpuMemGrad)" strokeWidth={1.5} strokeDasharray="6 3" name="GPU VRAM" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* System info */}
      {metrics && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-white/80">
              <Server className="h-4 w-4 text-white/40" />
              System Information
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm sm:grid-cols-4">
              <div>
                <p className="text-white/30 text-xs">Platform</p>
                <p className="font-medium text-white/80">
                  {metrics.system.platform} {metrics.system.architecture}
                </p>
              </div>
              <div>
                <p className="text-white/30 text-xs">Hostname</p>
                <p className="font-medium text-white/80">{metrics.system.hostname}</p>
              </div>
              <div>
                <p className="text-white/30 text-xs">Python</p>
                <p className="font-medium text-white/80">{metrics.system.python_version}</p>
              </div>
              <div>
                <p className="text-white/30 text-xs">Network I/O</p>
                <p className="font-medium text-white/80">
                  &uarr; {metrics.network.bytes_sent_mb.toFixed(0)} MB &darr;{" "}
                  {metrics.network.bytes_recv_mb.toFixed(0)} MB
                </p>
              </div>
              <div>
                <p className="text-white/30 text-xs">GPU</p>
                <p className="font-medium text-white/80">
                  {metrics.gpu.available ? metrics.gpu.name : "None (CPU mode)"}
                </p>
              </div>
              <div>
                <p className="text-white/30 text-xs">FAISS Index</p>
                <p className="font-medium text-white/80">
                  {metrics.model.loaded
                    ? `${metrics.model.index_size.toLocaleString()} vectors`
                    : "Not loaded"}
                </p>
              </div>
              <div>
                <p className="text-white/30 text-xs">Model Status</p>
                <p className="font-medium">
                  {metrics.model.loaded ? (
                    <span className="text-emerald-400">Loaded</span>
                  ) : (
                    <span className="text-red-400">Not loaded</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-white/30 text-xs">Uptime</p>
                <p className="font-medium text-white/80">
                  {formatUptime(metrics.uptime_seconds)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
