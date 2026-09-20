import { useState, useEffect, useMemo } from "react";
import { Activity, TrendingUp, Trophy, DollarSign, ShoppingCart, Eye, Heart, Search, ShoppingBag, CheckCircle, Package, Store, AlertCircle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, CartesianGrid } from "recharts";
import { connectLiveEvents, fetchDashboardMetrics, fetchMarketplaceHistory, fetchFairnessAudit, fetchLatencyMetrics, fetchInventoryPredictions } from "@/api";

interface DashboardMetrics {
  totalRevenue: { value: number; change: number };
  conversionRate: { overall: number; control: number; treatment: number };
  avgOrderValue: { value: number; byVariant?: Record<string, number>; change: number };
  activeSessions: number;
  pageViews: number;
  cartAdds: number;
  wishlistAdds: number;
  purchases: number;
  topProducts: { name: string; count: number; source?: string | null; sourceId?: string | null; avgPrice?: number | null }[];
  topQueries: { query: string; count: number }[];
  sourceDistribution?: Record<string, number>;
  recentEvents: { type: string; productName?: string; timestamp: string; city?: string; device?: string }[];
  avgEngagementScore?: number;
  avgPurchaseIntent?: number;
  topCategoryAffinity?: { category: string; count: number }[];
  abSignificance?: { zScore: number | null; pValue: number | null; significant: boolean; note: string };
  segmentDistribution?: { value_seeker: number; standard: number; premium_intent: number };
  generatedAt?: string;
  latency?: {
    p50: number;
    p99: number;
    avg: number;
    count: number;
    quality: { ndcg: number; hitRate: number };
  };
}

interface LatencyMetrics {
  p50: number;
  p99: number;
  avg: number;
  count: number;
  quality: { ndcg: number; hitRate: number };
}

interface InventoryPrediction {
  name: string;
  dailyVelocity: number;
  currentStock: number;
  daysRemaining: number;
  critical: boolean;
}

interface FairnessFactor {
  factor: string;
  description?: string;
  reason?: string;
}

interface FairnessAudit {
  auditNote: string;
  pricingFactors?: FairnessFactor[];
  excludedFactors?: FairnessFactor[];
  segmentBasis?: string;
  segmentDistribution?: { value_seeker: number; standard: number; premium_intent: number };
  lastAudited: string;
}

const EMPTY_METRICS: DashboardMetrics = {
  totalRevenue: { value: 0, change: 0 },
  conversionRate: { overall: 0, control: 0, treatment: 0 },
  avgOrderValue: { value: 0, change: 0 },
  activeSessions: 0,
  pageViews: 0,
  cartAdds: 0,
  wishlistAdds: 0,
  purchases: 0,
  topProducts: [],
  topQueries: [],
  recentEvents: [],
  latency: { p50: 0, p99: 0, avg: 0, count: 0, quality: { ndcg: 0, hitRate: 0 } }
};

export default function Dashboard() {
  const [events, setEvents] = useState<{ time: string; text: string }[]>([]);
  const [metrics, setMetrics] = useState<DashboardMetrics>(EMPTY_METRICS);
  const [metricsLoading, setMetricsLoading] = useState(true);

  // Marketplace history (real products, deterministic price series)
  type MktProduct = { id: string; name: string; source: string; livePrice: number; history: { hour: string; price: number }[] };
  const [marketplaceHistory, setMarketplaceHistory] = useState<MktProduct[]>([]);
  const [historyLoading, setHistoryLoading]         = useState(true);

  const [latency, setLatency] = useState<LatencyMetrics | null>(null);
  const [predictions, setPredictions] = useState<InventoryPrediction[]>([]);
  const [fairness, setFairness] = useState<FairnessAudit | null>(null);

  const LINE_COLORS = ['#f97316', '#3b82f6', '#22c55e', '#ec4899', '#eab308'];

  // ── Fetch metrics & fairness on mount + refresh every 30s ─────────────────────────────
  useEffect(() => {
    const load = () => {
      fetchDashboardMetrics()
        .then((data) => { setMetrics(prev => ({ ...prev, ...(data as DashboardMetrics) })); setMetricsLoading(false); })
        .catch(() => setMetricsLoading(false));
      
      fetchLatencyMetrics()
        .then(setLatency)
        .catch(() => {});
      
      fetchInventoryPredictions()
        .then(setPredictions)
        .catch(() => {});

      fetchFairnessAudit().then(setFairness).catch(() => {});
    };
    load();
    const interval = setInterval(load, 30_000);
    return () => clearInterval(interval);
  }, []);

  // ── Fetch marketplace price history ───────────────────────────────────────
  useEffect(() => {
    setHistoryLoading(true);
    fetchMarketplaceHistory()
      .then((data: { products?: MktProduct[] }) => setMarketplaceHistory(data.products || []))
      .catch(() => setMarketplaceHistory([]))
      .finally(() => setHistoryLoading(false));
  }, []);

  // ── Real-time SSE events ───────────────────────────────────────────────────
  useEffect(() => {
    const es = connectLiveEvents((event) => {
      setEvents((prev) => [
        { time: new Date().toLocaleTimeString(), text: event.message },
        ...prev.slice(0, 49),
      ]);
    });
    return () => es.close();
  }, []);

  // ── Transform history into Recharts row format ─────────────────────────────
  const chartData = useMemo(() => {
    if (!marketplaceHistory.length) return [];
    const rows = Array.from({ length: 24 }, (_, i) => ({
      hour: marketplaceHistory[0]?.history?.[i]?.hour || `${i}:00`,
    })) as Record<string, string | number>[];
    marketplaceHistory.forEach(product => {
      product.history.forEach((point, i) => {
        rows[i][product.name] = point.price;
      });
    });
    return rows;
  }, [marketplaceHistory]);

  const chartKeys = useMemo(
    () => marketplaceHistory.map(p => p.name),
    [marketplaceHistory]
  );

  // A/B conversion rates (real from DB)
  const controlCR = metrics.conversionRate?.control ?? 0;
  const treatmentCR = metrics.conversionRate?.treatment ?? 0;
  const controlAOVVal = (metrics.avgOrderValue as Record<string, unknown>)?.byVariant
    ? ((metrics.avgOrderValue as Record<string, unknown>).byVariant as Record<string, number>)?.control ?? metrics.avgOrderValue?.value
    : metrics.avgOrderValue?.value;
  const treatmentAOVVal = (metrics.avgOrderValue as Record<string, unknown>)?.byVariant
    ? ((metrics.avgOrderValue as Record<string, unknown>).byVariant as Record<string, number>)?.treatment ?? metrics.avgOrderValue?.value
    : metrics.avgOrderValue?.value;
  const treatmentWins = treatmentCR >= controlCR;

  const abData = [
    { metric: "Conversion Rate", control: controlCR, treatment: treatmentCR },
    { metric: "AOV (₹)", control: controlAOVVal, treatment: treatmentAOVVal },
    { metric: "Rev/Session (₹)", control: Math.round((controlCR * (controlAOVVal ?? 0)) / 100), treatment: Math.round((treatmentCR * (treatmentAOVVal ?? 0)) / 100) },
  ];

  const kpis = [
    {
      label: "REVENUE FROM TRACKED PURCHASES",
      value: `₹${(metrics.totalRevenue?.value ?? 0).toLocaleString("en-IN")}`,
      change: metrics.totalRevenue?.value > 0 ? `+${metrics.totalRevenue?.change ?? 0}%` : "No purchases yet",
      icon: <DollarSign size={20} className="text-green-400" />,
    },
    {
      label: "CONVERSION FROM TRACKED VIEWS",
      value: `${(metrics.conversionRate?.overall ?? 0).toFixed(2)}%`,
      change: `${metrics.purchases} purchases / ${metrics.pageViews} views`,
      icon: <TrendingUp size={20} className="text-blue-400" />,
    },
    {
      label: "AVG ORDER VALUE",
      value: metrics.avgOrderValue?.value > 0 ? `₹${(metrics.avgOrderValue?.value ?? 0).toLocaleString("en-IN")}` : "—",
      change: metrics.purchases > 0 ? `${metrics.purchases} orders` : "No orders yet",
      icon: <ShoppingBag size={20} className="text-purple-400" />,
    },
    {
      label: "LIVE ACTIVE SESSIONS",
      value: (metrics.activeSessions ?? 0).toString(),
      change: "last 5 min",
      icon: <Activity size={20} className="text-orange-400" />,
    },
    {
      label: "PAGE VIEWS",
      value: (metrics.pageViews ?? 0).toLocaleString("en-IN"),
      change: "last 24h",
      icon: <Eye size={20} className="text-gray-400" />,
    },
    {
      label: "CART ADDS",
      value: (metrics.cartAdds ?? 0).toLocaleString("en-IN"),
      change: "last 24h",
      icon: <ShoppingCart size={20} className="text-yellow-400" />,
    },
    {
      label: "PURCHASES",
      value: (metrics.purchases ?? 0).toString(),
      change: "last 24h",
      icon: <CheckCircle size={20} className="text-green-400" />,
    },
    {
      label: "WISHLIST ADDS",
      value: (metrics.wishlistAdds ?? 0).toString(),
      change: "last 24h",
      icon: <Heart size={20} className="text-pink-400" />,
    },
  ];

  return (
    <div className="container py-6 space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-foreground">PriceIQ Analytics — Live</h1>
        <span className="text-xs text-muted-foreground">
          {metricsLoading ? "Loading…" : `Updated ${metrics.generatedAt ? new Date(metrics.generatedAt).toLocaleTimeString() : "just now"}`}
        </span>
      </div>

      {/* Info banner — real data notice */}
      <div className="rounded-md border border-accent/20 bg-accent/5 px-4 py-2 text-xs text-accent">
        📊 Analytics are based on live PriceIQ user events and marketplace product data. &nbsp;·&nbsp; MongoDB Event collection (last 24h)
      </div>

      {/* KPI cards — 4 cols on large screens */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 gap-4">
        {kpis.slice(0, 4).map((k) => (
          <div key={k.label} className="rounded-lg border border-border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">{k.label}</span>
              <span>{k.icon}</span>
            </div>
            <p className="text-2xl font-bold text-card-foreground tabular-nums">{metricsLoading ? "—" : k.value}</p>
            <span className="text-xs text-gray-500">{k.change}</span>
          </div>
        ))}
      </div>

      {/* Secondary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {kpis.slice(4).map((k) => (
          <div key={k.label} className="rounded-lg border border-border bg-card p-3 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">{k.label}</span>
              <span>{k.icon}</span>
            </div>
            <p className="text-xl font-bold text-card-foreground tabular-nums">{metricsLoading ? "—" : k.value}</p>
            <span className="text-xs text-gray-500">{k.change}</span>
          </div>
        ))}
      </div>

      {/* ── Session Engagement KPIs (PS3 real-time behavior tracking) ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Engagement Score */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">Avg Engagement Score</span>
            <Activity size={18} className="text-blue-400" />
          </div>
          <div className="flex items-end gap-2">
            <p className="text-2xl font-bold text-card-foreground tabular-nums">
              {metricsLoading ? "—" : (metrics.avgEngagementScore ?? 0).toFixed(1)}
            </p>
            <span className="text-xs text-muted-foreground mb-1">pts / session</span>
          </div>
        </div>

        {/* Purchase Intent */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">Avg Purchase Intent</span>
            <TrendingUp size={18} className="text-green-400" />
          </div>
          <div className="flex items-end gap-2">
            <p className="text-2xl font-bold text-card-foreground tabular-nums">
              {metricsLoading ? "—" : ((metrics.avgPurchaseIntent ?? 0) * 100).toFixed(1)}%
            </p>
            <span className="text-xs text-muted-foreground mb-1">probability</span>
          </div>
          <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
            <div 
              className="h-full bg-green-500 rounded-full" 
              style={{ width: `${Math.min(100, (metrics.avgPurchaseIntent ?? 0) * 100)}%` }}
            />
          </div>
        </div>

        {/* Category Affinity */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">Top Category Affinity</span>
            <Heart size={18} className="text-pink-400" />
          </div>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {metricsLoading ? (
              <span className="text-sm text-muted-foreground">—</span>
            ) : metrics.topCategoryAffinity && metrics.topCategoryAffinity.length > 0 ? (
              metrics.topCategoryAffinity.slice(0, 3).map((aff, i) => (
                <span key={i} className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-accent/10 border border-accent/20 text-accent">
                  {aff.category} <span className="opacity-70">({aff.count})</span>
                </span>
              ))
            ) : (
              <span className="text-xs text-muted-foreground">Building profiles...</span>
            )}
          </div>
        </div>
      </div>

      {/* ── Evaluation Metrics (NDCG, Hit Rate, P99 Latency) ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* P99 Latency */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">API Latency (p99)</span>
            <Activity size={18} className="text-accent" />
          </div>
          <div className="flex items-end gap-2">
            <p className="text-2xl font-bold text-card-foreground tabular-nums">
              {metrics.latency?.p99 ?? "—"}
            </p>
            <span className="text-xs text-muted-foreground mb-1">ms</span>
          </div>
          <p className="text-[10px] text-muted-foreground">based on last {metrics.latency?.count ?? 0} samples</p>
        </div>

        {/* NDCG@10 */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">Rec Quality (NDCG@10)</span>
            <Trophy size={18} className="text-warning" />
          </div>
          <div className="flex items-end gap-2">
            <p className="text-2xl font-bold text-card-foreground tabular-nums">
              {metrics.latency?.quality?.ndcg ?? "—"}
            </p>
            <span className="text-xs text-muted-foreground mb-1">score</span>
          </div>
          <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
            <div 
              className="h-full bg-warning rounded-full" 
              style={{ width: `${(metrics.latency?.quality?.ndcg ?? 0) * 100}%` }}
            />
          </div>
        </div>

        {/* Hit Rate */}
        <div className="rounded-lg border border-border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide leading-snug">Recommendation Hit Rate</span>
            <CheckCircle size={18} className="text-success" />
          </div>
          <div className="flex items-end gap-2">
            <p className="text-2xl font-bold text-card-foreground tabular-nums">
              {((metrics.latency?.quality?.hitRate ?? 0) * 100).toFixed(1)}%
            </p>
            <span className="text-xs text-muted-foreground mb-1">accuracy</span>
          </div>
          <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
            <div 
              className="h-full bg-success rounded-full" 
              style={{ width: `${(metrics.latency?.quality?.hitRate ?? 0) * 100}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2/3 */}
        <div className="lg:col-span-2 space-y-6">
          {/* A/B Test */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="font-bold text-foreground">A/B Test — Dynamic Pricing Results</h2>
              <span className={`flex items-center gap-1 text-sm font-semibold ${treatmentWins ? "text-warning" : "text-muted-foreground"}`}>
                <Trophy size={16} /> {treatmentWins ? "Variant B Wins" : "Control Leads"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-4 text-center text-sm">
              <div className="rounded-md bg-secondary p-3 space-y-1">
                <p className="text-muted-foreground">Variant A (Control)</p>
                <p className="font-medium text-card-foreground">Rule-based pricing</p>
                <p className="text-muted-foreground">{controlCR}% CR · ₹{(controlAOVVal ?? 0).toLocaleString("en-IN")}/order</p>
              </div>
              <div className="rounded-md bg-accent/10 border border-accent/30 p-3 space-y-1">
                <p className="text-accent font-semibold">Variant B (Treatment)</p>
                <p className="font-medium text-card-foreground">Dynamic pricing</p>
                <p className="text-card-foreground">{treatmentCR}% CR · ₹{(treatmentAOVVal ?? 0).toLocaleString("en-IN")}/order</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={abData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="metric" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, color: "hsl(var(--card-foreground))" }} />
                <Bar dataKey="control" fill="hsl(var(--muted-foreground))" radius={[4, 4, 0, 0]} name="Control" />
                <Bar dataKey="treatment" fill="hsl(var(--accent))" radius={[4, 4, 0, 0]} name="Treatment" />
              </BarChart>
            </ResponsiveContainer>
            
            {/* Statistical Significance Indicator */}
            {metrics.abSignificance && (
              <div className={`mt-2 p-3 border rounded-md text-sm flex items-start gap-2 ${metrics.abSignificance.significant ? 'bg-success/10 border-success/30 text-success' : 'bg-secondary border-border text-muted-foreground'}`}>
                {metrics.abSignificance.significant ? <CheckCircle size={16} className="mt-0.5 flex-shrink-0" /> : <Activity size={16} className="mt-0.5 flex-shrink-0" />}
                <div>
                  <p className="font-semibold">{metrics.abSignificance.significant ? 'Statistically Significant' : 'Collecting Data'}</p>
                  <p className="text-xs opacity-80 mt-0.5">{metrics.abSignificance.note}</p>
                </div>
              </div>
            )}
          </div>

          {/* Price History — Live Marketplace Products */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-4">
            <div className="flex items-start justify-between flex-wrap gap-2">
              <div>
                <h2 className="font-bold text-foreground">Price History — Last 24 Hours</h2>
                <p className="text-xs text-muted-foreground mt-0.5">Based on live marketplace products</p>
              </div>
              {!historyLoading && marketplaceHistory.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap">
                  {marketplaceHistory.filter(p => p.source === 'amazon').length > 0 && (
                    <span className="inline-flex items-center gap-1 text-[10px] text-amber-400">
                      <Package size={10} /> {marketplaceHistory.filter(p => p.source === 'amazon').length} Amazon
                    </span>
                  )}
                  {marketplaceHistory.filter(p => p.source === 'flipkart').length > 0 && (
                    <span className="inline-flex items-center gap-1 text-[10px] text-blue-400">
                      <Store size={10} /> {marketplaceHistory.filter(p => p.source === 'flipkart').length} Flipkart
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Loading skeleton */}
            {historyLoading && (
              <div className="h-[300px] rounded-lg bg-secondary/50 animate-pulse flex items-center justify-center">
                <p className="text-xs text-muted-foreground">Loading marketplace price data…</p>
              </div>
            )}

            {/* Empty state */}
            {!historyLoading && chartData.length === 0 && (
              <div className="h-[300px] rounded-lg bg-secondary/50 flex flex-col items-center justify-center gap-3">
                <Activity size={32} className="text-muted-foreground/30" />
                <p className="text-sm font-semibold text-muted-foreground">No marketplace price data available</p>
                <p className="text-xs text-muted-foreground/60">Price history will appear when marketplace products are loaded</p>
              </div>
            )}

            {/* Chart */}
            {!historyLoading && chartData.length > 0 && (
              <>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="hour" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      interval={5} tickFormatter={(v: string) => v} />
                    <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      tickFormatter={(v: number) => `₹${(v/1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 8, color: "hsl(var(--card-foreground))" }}
                      formatter={(value: number) => [`₹${value.toLocaleString('en-IN')}`, '']}
                    />
                    <Legend />
                    {chartKeys.map((key, i) => (
                      <Line key={key} type="monotone" dataKey={key} stroke={LINE_COLORS[i % LINE_COLORS.length]}
                        strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                {/* Source legend chips under chart */}
                <div className="flex flex-wrap gap-2 pt-1">
                  {marketplaceHistory.map((p, i) => (
                    <div key={p.id} className="flex items-center gap-1.5">
                      <span className="inline-block h-2 w-4 rounded-full flex-shrink-0" style={{ backgroundColor: LINE_COLORS[i % LINE_COLORS.length] }} />
                      <span className="text-[10px] text-muted-foreground truncate max-w-[120px]">{p.name}</span>
                      {p.source === 'amazon'   && <Package size={9} className="text-amber-400 flex-shrink-0" />}
                      {p.source === 'flipkart' && <Store   size={9} className="text-blue-400 flex-shrink-0"  />}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Top Products — from real MongoDB aggregates */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-3">
            <h2 className="font-bold text-foreground">Top Clicked / Purchased Products</h2>
            <p className="text-xs text-muted-foreground">From real MongoDB cart &amp; purchase event aggregates</p>
            {!metricsLoading && metrics.topProducts && metrics.topProducts.length > 0 ? (
              <div className="space-y-2">
                {metrics.topProducts.map((p, i) => (
                  <div key={i} className="flex items-center justify-between rounded-md bg-secondary px-3 py-2 text-sm gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      {p.source === 'amazon' && (
                        <span className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[8px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 flex-shrink-0">
                          <Package size={7} /> Amazon
                        </span>
                      )}
                      {p.source === 'flipkart' && (
                        <span className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[8px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30 flex-shrink-0">
                          <Store size={7} /> Flipkart
                        </span>
                      )}
                      <span className="text-card-foreground truncate">{p.name}</span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {p.avgPrice && <span className="text-green-400 text-xs tabular-nums">₹{p.avgPrice.toLocaleString('en-IN')}</span>}
                      <span className="text-accent font-semibold text-xs">{p.count}×</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-md bg-secondary/50 px-5 py-8 text-center space-y-2">
                <ShoppingCart size={28} className="mx-auto text-muted-foreground/30" />
                <p className="text-sm font-semibold text-muted-foreground">No purchases yet</p>
                <p className="text-xs text-muted-foreground/60">Purchase events will appear here after users complete checkout</p>
              </div>
            )}
          </div>

          {/* Source Distribution — always visible */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-3">
            <h2 className="font-bold text-foreground">Marketplace Source Distribution</h2>
            <p className="text-xs text-muted-foreground">Breakdown of interactions by product source</p>
            {metrics.sourceDistribution && Object.keys(metrics.sourceDistribution).length > 0 ? (
              <div className="flex gap-4 flex-wrap">
                {Object.entries(metrics.sourceDistribution).map(([src, cnt]) => (
                  <div key={src} className="flex items-center gap-2 rounded-lg border border-border bg-secondary px-4 py-3">
                    {src === 'amazon'   && <Package size={16} className="text-amber-400" />}
                    {src === 'flipkart' && <Store   size={16} className="text-blue-400"  />}
                    {src === 'local'    && <Eye     size={16} className="text-gray-400"  />}
                    <div>
                      <p className="text-sm font-bold text-card-foreground tabular-nums">{(cnt as number).toLocaleString('en-IN')}</p>
                      <p className="text-[10px] text-muted-foreground capitalize">{src} events</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-md bg-secondary/50 px-5 py-8 text-center space-y-2">
                <Activity size={28} className="mx-auto text-muted-foreground/30" />
                <p className="text-sm font-semibold text-muted-foreground">No marketplace interaction yet</p>
                <p className="text-xs text-muted-foreground/60">Amazon and Flipkart event counts will appear after users interact with marketplace products</p>
              </div>
            )}
          </div>

          {/* Top Searched Queries — from real MongoDB search event aggregates */}
          <div className="rounded-lg border border-border bg-card p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Search size={16} className="text-accent" />
              <h2 className="font-bold text-foreground">Top Searched Queries</h2>
            </div>
            <p className="text-xs text-muted-foreground">From real MongoDB search event aggregates (last 24h)</p>
            {!metricsLoading && metrics.topQueries && metrics.topQueries.length > 0 ? (
              <div className="space-y-2">
                {metrics.topQueries.map((q, i) => (
                  <div key={i} className="flex items-center justify-between rounded-md bg-secondary px-3 py-2 text-sm">
                    <span className="flex items-center gap-2 text-card-foreground">
                      <span className="text-muted-foreground text-xs w-5">#{i + 1}</span>
                      {q.query}
                    </span>
                    <span className="text-accent font-semibold text-xs">{q.count} searches</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-md bg-secondary/50 px-5 py-8 text-center space-y-2">
                <Search size={28} className="mx-auto text-muted-foreground/30" />
                <p className="text-sm font-semibold text-muted-foreground">No search activity yet</p>
                <p className="text-xs text-muted-foreground/60">Search queries will appear here as users search the marketplace</p>
              </div>
            )}
          </div>
        </div>

        {/* Live Events feed */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-3">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-success" />
            </span>
            <h2 className="font-bold text-foreground text-sm">Real-time Events (Live SSE)</h2>
          </div>
          <div className="space-y-2 max-h-[600px] overflow-y-auto scrollbar-hide">
            {events.length === 0 && <p className="text-xs text-muted-foreground">Connecting to live feed…</p>}
            {events.map((e, i) => (
              <div key={i} className="rounded-md bg-secondary px-3 py-2 text-xs animate-fade-in">
                <span className="text-muted-foreground">{e.time}</span>
                <p className="text-card-foreground mt-0.5">{e.text}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Fairness & Transparency Audit Panel ── */}
      {fairness && (
        <div className="rounded-lg border border-border bg-card p-5 space-y-4 col-span-full">
          <div className="flex items-center gap-2">
            <CheckCircle size={20} className="text-success" />
            <h2 className="font-bold text-foreground">Fair Pricing Principles & Audit Statement</h2>
          </div>
          <p className="text-sm border-l-2 border-accent pl-3 text-muted-foreground italic">
            "{fairness.auditNote}"
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-success mb-3">Included Factors (Behavioral)</h3>
              <ul className="space-y-2">
                {fairness.pricingFactors?.map((f) => (
                  <li key={f.factor} className="flex items-start gap-2 text-sm">
                    <CheckCircle size={14} className="text-success mt-0.5" />
                    <div>
                      <span className="font-medium text-foreground">{f.factor}:</span> <span className="text-muted-foreground">{f.description}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-destructive mb-3">Excluded Factors (Demographic)</h3>
              <ul className="space-y-2">
                {fairness.excludedFactors?.map((f) => (
                  <li key={f.factor} className="flex items-start gap-2 text-sm">
                    <AlertCircle size={14} className="text-destructive mt-0.5" />
                    <div>
                      <span className="font-medium text-foreground">{f.factor}:</span> <span className="text-muted-foreground">{f.reason}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t border-border">
            <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-3">User Segment Distribution (Non-Demographic)</h3>
            <p className="text-xs text-muted-foreground mb-3">{fairness.segmentBasis}</p>
            <div className="flex gap-4 p-3 bg-secondary rounded-md">
              <div className="flex-1 text-center border-r border-border/50">
                <span className="block text-xl font-bold text-foreground">{fairness.segmentDistribution?.value_seeker || 0}</span>
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Value Seekers</span>
              </div>
              <div className="flex-1 text-center border-r border-border/50">
                <span className="block text-xl font-bold text-foreground">{fairness.segmentDistribution?.standard || 0}</span>
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Standard</span>
              </div>
              <div className="flex-1 text-center">
                <span className="block text-xl font-bold text-foreground">{fairness.segmentDistribution?.premium_intent || 0}</span>
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Premium Intent</span>
              </div>
            </div>
          </div>
          
          <div className="mt-2 text-[10px] text-muted-foreground text-center">
            Last Audited: {new Date(fairness.lastAudited).toLocaleString()} · PriceIQ Global Fairness Standard v2.4
          </div>
        </div>
      )}

      {/* ── Predictive Inventory & System Health (Phase 4) ── */}
      <div className="space-y-6 pt-6 border-t border-border/40 pb-10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-foreground">Operational Intelligence</h2>
            <p className="text-sm text-muted-foreground">Predictive inventory alerts and ML system health</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-accent/10 border border-accent/20 text-accent text-xs font-semibold">
            <Activity size={14} /> Real-time Monitoring Active
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Inventory Predictions */}
          <div className="lg:col-span-3 rounded-xl border border-border bg-card overflow-hidden">
            <div className="p-4 border-b border-border bg-muted/30">
              <h3 className="font-semibold text-sm flex items-center gap-2">
                <Package size={16} className="text-warning" /> 
                Inventory Stock-out Predictions
              </h3>
            </div>
            <div className="p-0 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/10 text-muted-foreground">
                    <th className="px-4 py-3 font-medium">Product</th>
                    <th className="px-4 py-3 font-medium">Daily Velocity</th>
                    <th className="px-4 py-3 font-medium">Stock</th>
                    <th className="px-4 py-3 font-medium">Est. Days to Zero</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {predictions.length > 0 ? predictions.slice(0, 5).map((p, i) => (
                    <tr key={i} className="hover:bg-muted/5 transition-colors">
                      <td className="px-4 py-3 font-medium text-foreground">{p.name}</td>
                      <td className="px-4 py-3 tabular-nums">{p.dailyVelocity} units/day</td>
                      <td className="px-4 py-3 tabular-nums">{p.currentStock}</td>
                      <td className={`px-4 py-3 font-bold tabular-nums ${p.daysRemaining < 3 ? 'text-red-500' : 'text-warning'}`}>
                        {p.daysRemaining} days
                      </td>
                      <td className="px-4 py-3">
                        {p.critical ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/10 text-red-500 font-bold border border-red-500/20">
                            <AlertCircle size={10} /> Critical
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-warning/10 text-warning font-bold border border-warning/20 text-[10px]">
                            Stable
                          </span>
                        )}
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground italic">
                        Insufficient data for predictions...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* ML System Health Alerts */}
          <div className="rounded-xl border border-border bg-card p-4 space-y-4 flex flex-col">
            <h3 className="font-semibold text-sm flex items-center gap-2">
              <Activity size={16} className="text-accent" />
              Model Vital Signs
            </h3>
            
            <div className="space-y-3 flex-1">
              {/* Latency Health */}
              <div className="p-3 rounded-lg border border-border bg-muted/20 space-y-2">
                <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  <span>P99 Latency</span>
                  <span className={latency?.p99 < 200 ? 'text-success' : 'text-warning'}>
                    {latency?.p99 < 200 ? 'Healthy' : 'Degraded'}
                  </span>
                </div>
                <div className="flex items-end gap-2">
                  <span className="text-xl font-bold tabular-nums">{latency?.p99 ?? '—'}</span>
                  <span className="text-[10px] text-muted-foreground mb-1">ms</span>
                </div>
              </div>

              {/* Training Status */}
              <div className="p-3 rounded-lg border border-border bg-muted/20 space-y-2">
                <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  <span>Model Updates</span>
                  <span className="text-accent underline cursor-help">Throttled Trigger</span>
                </div>
                <div className="flex items-center gap-2 py-1">
                  <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
                  <span className="text-xs font-semibold text-foreground">Models Active</span>
                </div>
                <p className="text-[10px] text-muted-foreground">Auto-retrain threshold: 500 events</p>
              </div>

              {/* Quality Alert */}
              {latency?.quality?.ndcg < 0.6 && latency?.quality?.ndcg > 0 && (
                <div className="p-3 rounded-lg border border-warning/30 bg-warning/5 flex items-start gap-2">
                  <AlertCircle size={14} className="text-warning mt-0.5" />
                  <div>
                    <p className="text-[10px] font-bold text-warning uppercase">Quality Alert</p>
                    <p className="text-[10px] text-muted-foreground leading-tight">NDCG below target (0.6). Automated retrain pending more data.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
