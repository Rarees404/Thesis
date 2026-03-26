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
      ? "bg-green-500"
      : status === "connecting"
        ? "bg-amber-500"
        : "bg-red-500";
  return (
    <span className="relative flex h-3 w-3">
      {status === "healthy" && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
      )}
      <span className={`relative inline-flex h-3 w-3 rounded-full ${color}`} />
    </span>
  );
}

function Trend({ data, field }: { data: MetricsDataPoint[]; field: keyof MetricsDataPoint }) {
  const recent = data.slice(-6);
  const values = recent.map((d) => d[field] as number).filter((v) => v > 0);
  if (values.length < 2) return <Minus className="h-3 w-3 text-neutral-600" />;
  const diff = values[values.length - 1] - values[0];
  if (Math.abs(diff) < 1) return <Minus className="h-3 w-3 text-neutral-600" />;
  return diff > 0 ? (
    <TrendingUp className="h-3 w-3 text-amber-400" />
  ) : (
    <TrendingDown className="h-3 w-3 text-green-400" />
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
              <p className="text-xs font-rajdhani font-semibold tracking-wider text-neutral-500 uppercase">{title}</p>
              <Trend data={sparkData} field={sparkKey} />
            </div>
            <p className="text-2xl font-bold font-mono text-neutral-200">{value}</p>
            {subtitle && (
              <p className="text-[11px] font-mono text-neutral-600">{subtitle}</p>
            )}
          </div>
          <div
            className="p-2.5 border border-red-600/10"
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
            <div className="flex justify-between text-[10px] font-mono text-neutral-600 mb-1 tracking-wider uppercase">
              <span>Usage</span>
              <span>{percent.toFixed(1)}%</span>
            </div>
            <div className="h-1.5 w-full bg-red-950/30 overflow-hidden">
              <div
                className="h-full transition-all duration-700 ease-out"
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
    <div className="border border-red-600/15 bg-[rgba(5,5,5,0.95)] px-3 py-2 shadow-xl backdrop-blur-xl">
      <p className="text-[10px] font-mono text-neutral-600 mb-1 tracking-wider uppercase">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-xs font-mono">
          <span className="h-2 w-2" style={{ backgroundColor: p.color }} />
          <span className="text-neutral-500 uppercase tracking-wider">{p.name}</span>
          <span className="ml-auto font-semibold text-neutral-200">
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
                <p className="text-sm font-rajdhani font-bold tracking-wider text-neutral-200 uppercase">
                  {status === "healthy"
                    ? "SYSTEMS OPERATIONAL"
                    : status === "connecting"
                      ? "ESTABLISHING LINK..."
                      : "SYSTEMS OFFLINE"}
                </p>
                {metrics && (
                  <p className="text-xs font-mono text-neutral-600 tracking-wider">
                    {metrics.system.hostname} &middot; {metrics.system.platform}{" "}
                    {metrics.system.architecture} &middot; Python{" "}
                    {metrics.system.python_version}
                  </p>
                )}
                {lastError && (
                  <p className="text-xs font-mono text-red-400">{lastError}</p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {metrics && (
                <Badge variant="outline" className="gap-1 text-[10px] font-mono border-red-600/15 bg-red-600/5 text-neutral-500 tracking-widest">
                  <Clock className="h-3 w-3" />
                  {formatUptime(metrics.uptime_seconds)}
                </Badge>
              )}
              {metrics?.model.loaded && (
                <Badge variant="secondary" className="gap-1 text-[10px] font-mono bg-red-600/5 text-neutral-500 border border-red-600/10 tracking-widest">
                  <Database className="h-3 w-3" />
                  {metrics.model.index_size.toLocaleString()} ASSETS
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPolling(!polling)}
                className="gap-1.5 text-[11px] text-neutral-500 hover:text-red-400 hover:bg-red-600/10 font-mono tracking-wider uppercase"
              >
                {polling ? <WifiOff className="h-3.5 w-3.5" /> : <Wifi className="h-3.5 w-3.5" />}
                {polling ? "Pause" : "Resume"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={poll}
                className="h-8 w-8 p-0 text-neutral-500 hover:text-red-400 hover:bg-red-600/10"
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
            title="CPU"
            value={`${metrics.cpu.percent.toFixed(1)}%`}
            subtitle={`${metrics.cpu.count_physical} cores / ${metrics.cpu.count} threads`}
            percent={metrics.cpu.percent}
            color="#dc2626"
            sparkData={history}
            sparkKey="cpu"
          />
          <MetricCard
            icon={MemoryStick}
            title="RAM"
            value={`${(metrics.memory.used_mb / 1024).toFixed(1)} GB`}
            subtitle={`of ${(metrics.memory.total_mb / 1024).toFixed(1)} GB total`}
            percent={metrics.memory.percent}
            color="#f59e0b"
            sparkData={history}
            sparkKey="memory"
          />
          <MetricCard
            icon={Zap}
            title={
              metrics.gpu.backend === "mps"
                ? "APPLE GPU (MPS)"
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
            color="#ef4444"
            sparkData={history}
            sparkKey="gpu"
          />
          <MetricCard
            icon={HardDrive}
            title="STORAGE"
            value={`${metrics.disk.used_gb.toFixed(0)} GB`}
            subtitle={`of ${metrics.disk.total_gb.toFixed(0)} GB total`}
            percent={metrics.disk.percent}
            color="#22c55e"
            sparkData={history}
            sparkKey="memory"
          />
        </div>
      )}

      {/* Main charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-neutral-400">
              <Activity className="h-4 w-4 text-red-500" />
              CPU & MEMORY
              <div className="ml-auto flex items-center gap-3">
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-neutral-600 tracking-wider">
                  <span className="h-2 w-5 bg-red-500/60" />
                  CPU
                </span>
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-neutral-600 tracking-wider">
                  <span className="h-2 w-5 bg-amber-400/60" />
                  RAM
                </span>
                {polling && (
                  <span className="flex items-center gap-1 text-[10px] font-mono text-green-400/60 tracking-widest">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
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
                    <stop offset="0%" stopColor="#dc2626" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#dc2626" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(220,38,38,0.04)" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.15)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.15)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <ReferenceLine y={cpuAvg} stroke="#dc2626" strokeDasharray="6 4" strokeOpacity={0.3} />
                <ReferenceLine y={memAvg} stroke="#f59e0b" strokeDasharray="6 4" strokeOpacity={0.3} />
                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="2 6" strokeOpacity={0.2} label={{ value: "80%", position: "right", fill: "rgba(239,68,68,0.3)", fontSize: 9 }} />
                <Area type="basis" dataKey="cpu" stroke="#dc2626" fill="url(#cpuGrad)" strokeWidth={2} name="CPU" dot={false} isAnimationActive={false} />
                <Area type="basis" dataKey="memory" stroke="#f59e0b" fill="url(#memGrad)" strokeWidth={2} name="RAM" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-neutral-400">
              <Monitor className="h-4 w-4 text-red-400" />
              {metrics?.gpu.backend === "mps" ? "APPLE GPU (MPS)" : "GPU UTILIZATION"}
              <div className="ml-auto flex items-center gap-3">
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-neutral-600 tracking-wider">
                  <span className="h-2 w-5 bg-red-400/60" />
                  {metrics?.gpu.backend === "mps" ? "P-Core %" : "Compute"}
                </span>
                <span className="flex items-center gap-1.5 text-[10px] font-mono text-neutral-600 tracking-wider">
                  <span className="h-0.5 w-5 border-t-2 border-dashed border-amber-400/60" />
                  {metrics?.gpu.backend === "mps" ? "Unified Mem" : "VRAM"}
                </span>
                {polling && (
                  <span className="flex items-center gap-1 text-[10px] font-mono text-green-400/60 tracking-widest">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </CardTitle>
            {metrics?.gpu.backend === "mps" && (
              <p className="text-[10px] font-mono text-neutral-700 -mt-1 tracking-wider">
                Apple Silicon shares CPU/GPU memory &middot; P-core load used as compute proxy
              </p>
            )}
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={history} margin={{ top: 5, right: 5, bottom: 0, left: -15 }}>
                <defs>
                  <linearGradient id="gpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gpuMemGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(220,38,38,0.04)" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.15)" }}
                  axisLine={false}
                  tickLine={false}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 25, 50, 75, 100]}
                  tick={{ fontSize: 9, fill: "rgba(255,255,255,0.15)" }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <RTooltip content={<ChartTooltipContent />} />
                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="2 6" strokeOpacity={0.2} label={{ value: "80%", position: "right", fill: "rgba(239,68,68,0.3)", fontSize: 9 }} />
                <Area type="basis" dataKey="gpu" stroke="#ef4444" fill="url(#gpuGrad)" strokeWidth={2} name="GPU Util" dot={false} isAnimationActive={false} />
                <Area type="basis" dataKey="gpuMem" stroke="#f59e0b" fill="url(#gpuMemGrad)" strokeWidth={1.5} strokeDasharray="6 3" name="GPU VRAM" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* System info */}
      {metrics && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-neutral-400">
              <Server className="h-4 w-4 text-red-500/60" />
              SYSTEM INFORMATION
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm sm:grid-cols-4">
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Platform</p>
                <p className="font-mono font-medium text-neutral-300">
                  {metrics.system.platform} {metrics.system.architecture}
                </p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Hostname</p>
                <p className="font-mono font-medium text-neutral-300">{metrics.system.hostname}</p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Python</p>
                <p className="font-mono font-medium text-neutral-300">{metrics.system.python_version}</p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Network I/O</p>
                <p className="font-mono font-medium text-neutral-300">
                  &uarr; {metrics.network.bytes_sent_mb.toFixed(0)} MB &darr;{" "}
                  {metrics.network.bytes_recv_mb.toFixed(0)} MB
                </p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">GPU</p>
                <p className="font-mono font-medium text-neutral-300">
                  {metrics.gpu.available ? metrics.gpu.name : "None (CPU mode)"}
                </p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">FAISS Index</p>
                <p className="font-mono font-medium text-neutral-300">
                  {metrics.model.loaded
                    ? `${metrics.model.index_size.toLocaleString()} vectors`
                    : "Not loaded"}
                </p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Model Status</p>
                <p className="font-mono font-medium">
                  {metrics.model.loaded ? (
                    <span className="text-green-400">OPERATIONAL</span>
                  ) : (
                    <span className="text-red-400">OFFLINE</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-neutral-600 text-xs font-rajdhani tracking-wider uppercase">Uptime</p>
                <p className="font-mono font-medium text-neutral-300">
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
