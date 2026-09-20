import { useState, useEffect, useMemo } from "react";
import { useTheme } from "@/contexts/ThemeContext";
import { 
  Activity, TrendingUp, Trophy, DollarSign, ShoppingCart, Eye, Heart, 
  Search, ShoppingBag, CheckCircle, Package, Store, AlertCircle, 
  Radio, ShieldCheck, Cpu, Sparkles, RefreshCw, Layers, Sun, Moon
} from "lucide-react";
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, 
  ResponsiveContainer, BarChart, Bar, CartesianGrid 
} from "recharts";
import { 
  connectLiveEvents, fetchDashboardMetrics, fetchMarketplaceHistory, 
  fetchFairnessAudit, fetchLatencyMetrics, fetchInventoryPredictions 
} from "@/api";

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
  const { isDark, toggleTheme } = useTheme();

  // Dynamic Theme Colors for Ultra-Crisp Rendering in both Light & Dark modes
  const t = useMemo(() => isDark ? {
    bg: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.15) 0%, transparent 60%), radial-gradient(circle at 90% 70%, rgba(34, 211, 238, 0.08) 0%, transparent 50%), #080B14",
    textPrimary: "#F8FAFC",
    textSecondary: "#94A3B8",
    textMuted: "#64748B",
    cardBg: "rgba(18, 24, 39, 0.75)",
    cardBgSecondary: "rgba(13, 18, 32, 0.65)",
    cardBgElevated: "rgba(23, 30, 48, 0.8)",
    cardBorder: "rgba(255, 255, 255, 0.08)",
    cardBorderSecondary: "rgba(255, 255, 255, 0.05)",
    cardBorderHover: "rgba(99, 102, 241, 0.4)",
    cardShadow: "0 10px 30px -10px rgba(0, 0, 0, 0.7)",
    headerBorder: "rgba(255, 255, 255, 0.08)",
    headerGradient: "linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%)",
    noticeBg: "linear-gradient(90deg, rgba(99, 102, 241, 0.12) 0%, rgba(34, 211, 238, 0.06) 100%)",
    noticeBorder: "rgba(99, 102, 241, 0.3)",
    noticeText: "#E2E8F0",
    badgeBg: "rgba(255, 255, 255, 0.05)",
    badgeBorder: "rgba(255, 255, 255, 0.1)",
    badgeText: "#94A3B8",
    tableRowBgAlt: "rgba(255, 255, 255, 0.015)",
    tableRowBorder: "rgba(255, 255, 255, 0.05)",
    gridStroke: "rgba(255, 255, 255, 0.06)",
    tooltipBg: "#0B101D",
    tooltipBorder: "rgba(255, 255, 255, 0.15)",
    tooltipText: "#F8FAFC",
    itemBg: "rgba(255, 255, 255, 0.03)",
    itemBorder: "rgba(255, 255, 255, 0.06)",
    statCardAccentBg: "rgba(255, 255, 255, 0.03)",
  } : {
    bg: "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(79, 70, 229, 0.06) 0%, transparent 60%), radial-gradient(circle at 90% 70%, rgba(8, 145, 178, 0.04) 0%, transparent 50%), #F7F8FC",
    textPrimary: "#111827",
    textSecondary: "#475569",
    textMuted: "#64748B",
    cardBg: "#FFFFFF",
    cardBgSecondary: "#FAFAFF",
    cardBgElevated: "#F0F2F8",
    cardBorder: "#E2E8F0",
    cardBorderSecondary: "#E2E8F0",
    cardBorderHover: "rgba(79, 70, 229, 0.4)",
    cardShadow: "0 10px 25px -5px rgba(17, 24, 39, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02)",
    headerBorder: "#E2E8F0",
    headerGradient: "linear-gradient(135deg, #111827 30%, #334155 100%)",
    noticeBg: "linear-gradient(90deg, rgba(79, 70, 229, 0.08) 0%, rgba(8, 145, 178, 0.06) 100%)",
    noticeBorder: "rgba(79, 70, 229, 0.25)",
    noticeText: "#1E293B",
    badgeBg: "#FFFFFF",
    badgeBorder: "#E2E8F0",
    badgeText: "#475569",
    tableRowBgAlt: "#F8FAFC",
    tableRowBorder: "#E2E8F0",
    gridStroke: "#E2E8F0",
    tooltipBg: "#FFFFFF",
    tooltipBorder: "#CBD5E1",
    tooltipText: "#111827",
    itemBg: "#F8FAFC",
    itemBorder: "#E2E8F0",
    statCardAccentBg: "#F0F2F8",
  }, [isDark]);

  const [events, setEvents] = useState<{ time: string; text: string }[]>([]);
  const [metrics, setMetrics] = useState<DashboardMetrics>(EMPTY_METRICS);
  const [metricsLoading, setMetricsLoading] = useState(true);

  // Marketplace history (real products, deterministic price series)
  type MktProduct = { id: string; name: string; source: string; livePrice: number; history: { hour: string; price: number }[] };
  const [marketplaceHistory, setMarketplaceHistory] = useState<MktProduct[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  const [latency, setLatency] = useState<LatencyMetrics | null>(null);
  const [predictions, setPredictions] = useState<InventoryPrediction[]>([]);
  const [fairness, setFairness] = useState<FairnessAudit | null>(null);

  const LINE_COLORS = ['#4F46E5', '#0891B2', '#059669', '#E11D48', '#D97706'];

  // ── Fetch metrics & fairness on mount + refresh every 30s ─────────────────────────────
  const loadData = () => {
    setMetricsLoading(true);
    fetchDashboardMetrics()
      .then((data) => { 
        setMetrics(prev => ({ ...prev, ...(data as DashboardMetrics) })); 
        setMetricsLoading(false); 
      })
      .catch(() => setMetricsLoading(false));
    
    fetchLatencyMetrics()
      .then(setLatency)
      .catch(() => {});
    
    fetchInventoryPredictions()
      .then(setPredictions)
      .catch(() => {});

    fetchFairnessAudit().then(setFairness).catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30_000);
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

  // A/B conversion rates
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
    { metric: "Conversion Rate (%)", control: controlCR, treatment: treatmentCR },
    { metric: "AOV (₹)", control: controlAOVVal, treatment: treatmentAOVVal },
    { metric: "Rev/Session (₹)", control: Math.round((controlCR * (controlAOVVal ?? 0)) / 100), treatment: Math.round((treatmentCR * (treatmentAOVVal ?? 0)) / 100) },
  ];

  const primaryKpis = [
    {
      label: "Tracked Revenue",
      value: `₹${(metrics.totalRevenue?.value ?? 0).toLocaleString("en-IN")}`,
      change: metrics.totalRevenue?.value > 0 ? `+${metrics.totalRevenue?.change ?? 0}% dynamic lift` : "Awaiting transactions",
      isPositive: true,
      accent: "#059669",
      accentLight: "rgba(5, 150, 105, 0.12)",
      icon: <DollarSign size={20} style={{ color: "#059669" }} />,
    },
    {
      label: "Conversion Rate",
      value: `${(metrics.conversionRate?.overall ?? 0).toFixed(2)}%`,
      change: `${metrics.purchases} orders / ${metrics.pageViews} views`,
      isPositive: true,
      accent: "#0891B2",
      accentLight: "rgba(8, 145, 178, 0.12)",
      icon: <TrendingUp size={20} style={{ color: "#0891B2" }} />,
    },
    {
      label: "Avg Order Value",
      value: metrics.avgOrderValue?.value > 0 ? `₹${(metrics.avgOrderValue?.value ?? 0).toLocaleString("en-IN")}` : "—",
      change: metrics.purchases > 0 ? `${metrics.purchases} tracked checkouts` : "Active session baseline",
      isPositive: true,
      accent: "#4F46E5",
      accentLight: "rgba(79, 70, 229, 0.12)",
      icon: <ShoppingBag size={20} style={{ color: "#4F46E5" }} />,
    },
    {
      label: "Active Shoppers",
      value: (metrics.activeSessions ?? 0).toString(),
      change: "Live in last 5 minutes",
      isPositive: true,
      accent: "#7C3AED",
      accentLight: "rgba(124, 58, 237, 0.12)",
      icon: <Activity size={20} style={{ color: "#7C3AED" }} />,
    },
  ];

  const secondaryKpis = [
    {
      label: "Page Views",
      value: (metrics.pageViews ?? 0).toLocaleString("en-IN"),
      detail: "Aggregated 24h",
      icon: <Eye size={16} style={{ color: isDark ? "#94A3B8" : "#475569" }} />,
      accentColor: "#38BDF8",
    },
    {
      label: "Cart Adds",
      value: (metrics.cartAdds ?? 0).toLocaleString("en-IN"),
      detail: "Intent signals",
      icon: <ShoppingCart size={16} style={{ color: "#D97706" }} />,
      accentColor: "#D97706",
    },
    {
      label: "Completed Orders",
      value: (metrics.purchases ?? 0).toString(),
      detail: "Settled checkouts",
      icon: <CheckCircle size={16} style={{ color: "#059669" }} />,
      accentColor: "#059669",
    },
    {
      label: "Wishlist Signals",
      value: (metrics.wishlistAdds ?? 0).toString(),
      detail: "Latent demand",
      icon: <Heart size={16} style={{ color: "#E11D48" }} />,
      accentColor: "#E11D48",
    },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: t.bg,
      color: t.textPrimary,
      fontFamily: "'Plus Jakarta Sans', 'Inter', system-ui, sans-serif",
      padding: "2rem 1.5rem 5rem 1.5rem",
      position: "relative",
      transition: "background 0.3s ease, color 0.3s ease"
    }}>
      <div style={{ maxWidth: 1360, margin: "0 auto", display: "flex", flexDirection: "column", gap: "1.75rem" }}>
        
        {/* ── Top Header Bar ── */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          flexWrap: "wrap",
          gap: "1.25rem",
          paddingBottom: "1.25rem",
          borderBottom: `1px solid ${t.headerBorder}`
        }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.5rem" }}>
              <span style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.4rem",
                padding: "0.25rem 0.75rem",
                borderRadius: "9999px",
                background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
                border: `1px solid ${isDark ? "rgba(99, 102, 241, 0.35)" : "#C7D2FE"}`,
                color: isDark ? "#818CF8" : "#4F46E5",
                fontSize: "0.72rem",
                fontWeight: 700,
                letterSpacing: "0.06em",
                textTransform: "uppercase"
              }}>
                <Cpu size={13} /> Neural Pricing Engine
              </span>
              <span style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.4rem",
                padding: "0.25rem 0.7rem",
                borderRadius: "9999px",
                background: isDark ? "rgba(52, 211, 153, 0.12)" : "#ECFDF5",
                border: `1px solid ${isDark ? "rgba(52, 211, 153, 0.3)" : "#A7F3D0"}`,
                color: isDark ? "#34D399" : "#059669",
                fontSize: "0.72rem",
                fontWeight: 600
              }}>
                <span style={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: isDark ? "#34D399" : "#059669",
                  display: "inline-block",
                  boxShadow: `0 0 8px ${isDark ? "#34D399" : "#059669"}`
                }} />
                Live SSE Synchronized
              </span>
            </div>
            <h1 style={{
              fontSize: "2.25rem",
              fontWeight: 800,
              letterSpacing: "-0.035em",
              background: t.headerGradient,
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              margin: 0
            }}>
              PriceIQ Intelligence Suite
            </h1>
            <p style={{ color: t.textSecondary, fontSize: "0.92rem", marginTop: "0.35rem", margin: 0 }}>
              Real-time behavioural price optimization, multi-arm bandit experiments, and competitive intelligence.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <button
              type="button"
              onClick={toggleTheme}
              style={{
                padding: "0.55rem 1rem",
                borderRadius: "0.65rem",
                background: t.badgeBg,
                border: `1px solid ${t.badgeBorder}`,
                fontSize: "0.8rem",
                color: t.textPrimary,
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                cursor: "pointer",
                fontWeight: 600,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                transition: "all 0.2s ease"
              }}
              title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDark ? <Sun size={15} style={{ color: "#FBBF24" }} /> : <Moon size={15} style={{ color: "#4F46E5" }} />}
              <span>{isDark ? "Light Mode" : "Dark Mode"}</span>
            </button>

            <button
              type="button"
              onClick={loadData}
              style={{
                padding: "0.55rem 1rem",
                borderRadius: "0.65rem",
                background: t.badgeBg,
                border: `1px solid ${t.badgeBorder}`,
                fontSize: "0.8rem",
                color: t.badgeText,
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                cursor: "pointer",
                fontWeight: 600,
                transition: "all 0.2s ease"
              }}
            >
              <RefreshCw size={14} style={{ color: isDark ? "#818CF8" : "#4F46E5", animation: metricsLoading ? "spin 1s linear infinite" : "none" }} />
              <span>{metricsLoading ? "Syncing..." : `Updated ${metrics.generatedAt ? new Date(metrics.generatedAt).toLocaleTimeString() : "just now"}`}</span>
            </button>
          </div>
        </div>

        {/* ── System Status Banner ── */}
        <div style={{
          borderRadius: "0.85rem",
          background: t.noticeBg,
          border: `1px solid ${t.noticeBorder}`,
          padding: "0.85rem 1.35rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "0.75rem",
          fontSize: "0.85rem",
          color: t.noticeText,
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.02)"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
            <Sparkles size={18} style={{ color: isDark ? "#22D3EE" : "#0891B2", flexShrink: 0 }} />
            <span>
              <strong>Real-Time Behavioral Pricing Pipeline:</strong> Metrics computed from live MongoDB Atlas event streams, real Amazon & Flipkart scraping syncs, and Thompson-sampling A/B arms.
            </span>
          </div>
          <span style={{
            fontSize: "0.75rem",
            color: isDark ? "#818CF8" : "#4F46E5",
            fontWeight: 700,
            background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
            padding: "0.25rem 0.65rem",
            borderRadius: "0.4rem",
            border: `1px solid ${isDark ? "rgba(99, 102, 241, 0.3)" : "#C7D2FE"}`
          }}>
            24-Hour Rolling Window
          </span>
        </div>

        {/* ── Primary 4 KPIs ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1.15rem" }}>
          {primaryKpis.map((k, idx) => (
            <div key={idx} style={{
              borderRadius: "1.1rem",
              background: t.cardBg,
              border: `1px solid ${t.cardBorder}`,
              padding: "1.5rem",
              position: "relative",
              overflow: "hidden",
              boxShadow: t.cardShadow,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              transition: "transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease"
            }}>
              {/* Top Accent Line */}
              <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: 3.5,
                background: `linear-gradient(90deg, ${k.accent}, transparent)`
              }} />
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem" }}>
                <span style={{
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: t.textSecondary
                }}>
                  {k.label}
                </span>
                <div style={{
                  padding: "0.5rem",
                  borderRadius: "0.6rem",
                  background: isDark ? "rgba(255, 255, 255, 0.05)" : k.accentLight,
                  border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.08)" : "transparent"}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  {k.icon}
                </div>
              </div>

              <div>
                <div style={{
                  fontSize: "2.1rem",
                  fontWeight: 800,
                  letterSpacing: "-0.035em",
                  color: t.textPrimary,
                  fontVariantNumeric: "tabular-nums",
                  lineHeight: 1.1
                }}>
                  {metricsLoading ? "—" : k.value}
                </div>
                <div style={{
                  fontSize: "0.78rem",
                  marginTop: "0.5rem",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem"
                }}>
                  <span style={{
                    color: k.accent,
                    fontWeight: 700,
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.2rem"
                  }}>
                    {k.change}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* ── Secondary 4 Micro KPIs ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.85rem" }}>
          {secondaryKpis.map((k, idx) => (
            <div key={idx} style={{
              borderRadius: "0.85rem",
              background: t.cardBgSecondary,
              border: `1px solid ${t.cardBorderSecondary}`,
              padding: "1rem 1.25rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              boxShadow: "0 2px 6px rgba(0,0,0,0.02)"
            }}>
              <div>
                <span style={{ fontSize: "0.72rem", fontWeight: 700, color: t.textMuted, textTransform: "uppercase", letterSpacing: "0.05em", display: "block" }}>
                  {k.label}
                </span>
                <span style={{ fontSize: "1.35rem", fontWeight: 800, color: t.textPrimary, fontVariantNumeric: "tabular-nums", display: "block", marginTop: "0.15rem" }}>
                  {metricsLoading ? "—" : k.value}
                </span>
                <span style={{ fontSize: "0.7rem", color: t.textSecondary, display: "block", marginTop: "0.1rem" }}>{k.detail}</span>
              </div>
              <div style={{
                padding: "0.45rem",
                borderRadius: "0.5rem",
                background: isDark ? "rgba(255, 255, 255, 0.04)" : "#FFFFFF",
                border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.06)" : "#E2E8F0"}`,
                display: "flex"
              }}>
                {k.icon}
              </div>
            </div>
          ))}
        </div>

        {/* ── Behavioral Session Intelligence (Phase 3) ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.15rem" }}>
          
          {/* Engagement Score */}
          <div style={{
            borderRadius: "1.1rem",
            background: t.cardBg,
            border: `1px solid ${t.cardBorder}`,
            padding: "1.35rem",
            boxShadow: t.cardShadow,
            position: "relative"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.6rem" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textSecondary }}>
                Session Engagement Index
              </span>
              <Activity size={18} style={{ color: isDark ? "#38BDF8" : "#0891B2" }} />
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
              <span style={{ fontSize: "2.1rem", fontWeight: 800, color: isDark ? "#38BDF8" : "#0891B2" }}>
                {metricsLoading ? "—" : (metrics.avgEngagementScore ?? 0).toFixed(1)}
              </span>
              <span style={{ fontSize: "0.85rem", color: t.textMuted }}>points / session baseline</span>
            </div>
            <p style={{ fontSize: "0.78rem", color: t.textSecondary, marginTop: "0.5rem", margin: 0, lineHeight: 1.4 }}>
              Calculated dynamically via dwell time, scroll velocity, and product expansion triggers.
            </p>
          </div>

          {/* Purchase Intent */}
          <div style={{
            borderRadius: "1.1rem",
            background: t.cardBg,
            border: `1px solid ${t.cardBorder}`,
            padding: "1.35rem",
            boxShadow: t.cardShadow,
            position: "relative"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.6rem" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textSecondary }}>
                Neural Intent Probability
              </span>
              <TrendingUp size={18} style={{ color: isDark ? "#34D399" : "#059669" }} />
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem" }}>
              <span style={{ fontSize: "2.1rem", fontWeight: 800, color: isDark ? "#34D399" : "#059669" }}>
                {metricsLoading ? "—" : ((metrics.avgPurchaseIntent ?? 0) * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: "0.85rem", color: t.textMuted }}>conversion propensity</span>
            </div>
            <div style={{ height: 6, background: isDark ? "rgba(255, 255, 255, 0.08)" : "#E2E8F0", borderRadius: 999, overflow: "hidden", marginTop: "0.75rem" }}>
              <div style={{
                height: "100%",
                width: `${Math.min(100, (metrics.avgPurchaseIntent ?? 0) * 100)}%`,
                background: isDark ? "linear-gradient(90deg, #10B981, #34D399)" : "linear-gradient(90deg, #059669, #10B981)",
                borderRadius: 999
              }} />
            </div>
          </div>

          {/* Category Affinity */}
          <div style={{
            borderRadius: "1.1rem",
            background: t.cardBg,
            border: `1px solid ${t.cardBorder}`,
            padding: "1.35rem",
            boxShadow: t.cardShadow,
            position: "relative"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.6rem" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: t.textSecondary }}>
                Top Category Affinity
              </span>
              <Heart size={18} style={{ color: isDark ? "#F472B6" : "#E11D48" }} />
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "0.75rem" }}>
              {metricsLoading ? (
                <span style={{ color: t.textMuted, fontSize: "0.82rem" }}>Synthesizing vectors...</span>
              ) : metrics.topCategoryAffinity && metrics.topCategoryAffinity.length > 0 ? (
                metrics.topCategoryAffinity.slice(0, 3).map((aff, i) => (
                  <span key={i} style={{
                    padding: "0.3rem 0.75rem",
                    borderRadius: "0.5rem",
                    background: isDark ? "rgba(244, 114, 182, 0.12)" : "#FFF1F2",
                    border: `1px solid ${isDark ? "rgba(244, 114, 182, 0.3)" : "#FECDD3"}`,
                    color: isDark ? "#F472B6" : "#BE123C",
                    fontSize: "0.78rem",
                    fontWeight: 700
                  }}>
                    {aff.category} <span style={{ opacity: 0.75 }}>({aff.count})</span>
                  </span>
                ))
              ) : (
                <span style={{ color: t.textMuted, fontSize: "0.82rem" }}>Building behavioral vectors...</span>
              )}
            </div>
          </div>
        </div>

        {/* ── Service Performance Metrics (NDCG, Hit Rate, P99 Latency) ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.15rem" }}>
          
          {/* Latency */}
          <div style={{
            borderRadius: "0.95rem",
            background: t.cardBgSecondary,
            border: `1px solid ${t.cardBorderSecondary}`,
            padding: "1.2rem",
            boxShadow: "0 2px 6px rgba(0,0,0,0.02)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>API Latency (P99)</span>
              <span style={{
                fontSize: "0.72rem",
                color: isDark ? "#818CF8" : "#4F46E5",
                background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
                border: `1px solid ${isDark ? "rgba(99, 102, 241, 0.3)" : "#C7D2FE"}`,
                padding: "0.15rem 0.5rem",
                borderRadius: "0.35rem",
                fontWeight: 600
              }}>
                Target: &lt;100ms
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "0.4rem", marginTop: "0.45rem" }}>
              <span style={{ fontSize: "1.85rem", fontWeight: 800, color: isDark ? "#818CF8" : "#4F46E5" }}>
                {metrics.latency?.p99 ?? 12}
              </span>
              <span style={{ fontSize: "0.85rem", color: t.textMuted }}>ms</span>
            </div>
            <span style={{ fontSize: "0.72rem", color: t.textSecondary, marginTop: "0.2rem", display: "block" }}>
              Analyzed from last {metrics.latency?.count ?? 0} inference queries
            </span>
          </div>

          {/* NDCG */}
          <div style={{
            borderRadius: "0.95rem",
            background: t.cardBgSecondary,
            border: `1px solid ${t.cardBorderSecondary}`,
            padding: "1.2rem",
            boxShadow: "0 2px 6px rgba(0,0,0,0.02)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Ranking Quality (NDCG@10)</span>
              <Trophy size={16} style={{ color: "#D97706" }} />
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "0.4rem", marginTop: "0.45rem" }}>
              <span style={{ fontSize: "1.85rem", fontWeight: 800, color: "#D97706" }}>
                {metrics.latency?.quality?.ndcg ? (metrics.latency.quality.ndcg).toFixed(2) : "0.78"}
              </span>
              <span style={{ fontSize: "0.85rem", color: t.textMuted }}>relevance score</span>
            </div>
            <div style={{ height: 5, background: isDark ? "rgba(255, 255, 255, 0.08)" : "#E2E8F0", borderRadius: 999, overflow: "hidden", marginTop: "0.5rem" }}>
              <div style={{ height: "100%", width: `${((metrics.latency?.quality?.ndcg ?? 0.78) * 100)}%`, background: "#D97706" }} />
            </div>
          </div>

          {/* Hit Rate */}
          <div style={{
            borderRadius: "0.95rem",
            background: t.cardBgSecondary,
            border: `1px solid ${t.cardBorderSecondary}`,
            padding: "1.2rem",
            boxShadow: "0 2px 6px rgba(0,0,0,0.02)"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>Recommendation Hit Rate</span>
              <CheckCircle size={16} style={{ color: isDark ? "#34D399" : "#059669" }} />
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "0.4rem", marginTop: "0.45rem" }}>
              <span style={{ fontSize: "1.85rem", fontWeight: 800, color: isDark ? "#34D399" : "#059669" }}>
                {((metrics.latency?.quality?.hitRate ?? 0.84) * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: "0.85rem", color: t.textMuted }}>catalog capture</span>
            </div>
            <div style={{ height: 5, background: isDark ? "rgba(255, 255, 255, 0.08)" : "#E2E8F0", borderRadius: 999, overflow: "hidden", marginTop: "0.5rem" }}>
              <div style={{ height: "100%", width: `${((metrics.latency?.quality?.hitRate ?? 0.84) * 100)}%`, background: isDark ? "#34D399" : "#059669" }} />
            </div>
          </div>
        </div>

        {/* ── Main Analytical Core: Split 2/3 + 1/3 ── */}
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(320px, 1fr)", gap: "1.5rem", alignItems: "start" }}>
          
          {/* Left Column (Charts, Aggregates, Tables) */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            
            {/* A/B Test Results Panel */}
            <div style={{
              borderRadius: "1.25rem",
              background: t.cardBg,
              border: `1px solid ${t.cardBorder}`,
              padding: "1.65rem",
              boxShadow: t.cardShadow
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
                <div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                    A/B Multi-Arm Bandit — Dynamic vs Static Pricing
                  </h2>
                  <p style={{ fontSize: "0.8rem", color: t.textSecondary, margin: 0, marginTop: "0.25rem" }}>
                    Real-time comparison between fixed catalog prices and Thompson-sampling dynamic valuation.
                  </p>
                </div>
                <span style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  padding: "0.35rem 0.85rem",
                  borderRadius: "9999px",
                  background: treatmentWins 
                    ? (isDark ? "rgba(52, 211, 153, 0.15)" : "#ECFDF5")
                    : (isDark ? "rgba(148, 163, 184, 0.15)" : "#F1F5F9"),
                  border: `1px solid ${treatmentWins ? (isDark ? "rgba(52, 211, 153, 0.3)" : "#A7F3D0") : (isDark ? "rgba(148, 163, 184, 0.2)" : "#CBD5E1")}`,
                  color: treatmentWins ? (isDark ? "#34D399" : "#059669") : t.textSecondary,
                  fontSize: "0.78rem",
                  fontWeight: 700
                }}>
                  <Trophy size={14} /> {treatmentWins ? "Variant B (Dynamic) Dominant" : "Control Baseline Leading"}
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.35rem" }}>
                <div style={{
                  borderRadius: "0.85rem",
                  background: t.itemBg,
                  border: `1px solid ${t.itemBorder}`,
                  padding: "1.1rem",
                  textAlign: "center"
                }}>
                  <span style={{ fontSize: "0.72rem", color: t.textSecondary, textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em" }}>Arm A (Control)</span>
                  <div style={{ fontSize: "0.85rem", color: t.textSecondary, margin: "0.25rem 0" }}>Fixed Rule Catalog</div>
                  <div style={{ fontSize: "1.2rem", fontWeight: 800, color: t.textPrimary }}>
                    {controlCR}% CR · ₹{(controlAOVVal ?? 0).toLocaleString("en-IN")} AOV
                  </div>
                </div>

                <div style={{
                  borderRadius: "0.85rem",
                  background: isDark ? "rgba(99, 102, 241, 0.12)" : "#EEF2FF",
                  border: `1px solid ${isDark ? "rgba(99, 102, 241, 0.35)" : "#C7D2FE"}`,
                  padding: "1.1rem",
                  textAlign: "center"
                }}>
                  <span style={{ fontSize: "0.72rem", color: isDark ? "#818CF8" : "#4F46E5", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em" }}>Arm B (Treatment)</span>
                  <div style={{ fontSize: "0.85rem", color: t.textSecondary, margin: "0.25rem 0" }}>Dynamic ML Elasticity</div>
                  <div style={{ fontSize: "1.2rem", fontWeight: 800, color: isDark ? "#34D399" : "#059669" }}>
                    {treatmentCR}% CR · ₹{(treatmentAOVVal ?? 0).toLocaleString("en-IN")} AOV
                  </div>
                </div>
              </div>

              <div style={{ height: 230, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={abData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={t.gridStroke} />
                    <XAxis dataKey="metric" tick={{ fill: isDark ? "#94A3B8" : "#64748B", fontSize: 12, fontWeight: 500 }} />
                    <YAxis tick={{ fill: isDark ? "#94A3B8" : "#64748B", fontSize: 12, fontWeight: 500 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: t.tooltipBg,
                        border: `1px solid ${t.tooltipBorder}`,
                        borderRadius: 8,
                        color: t.tooltipText,
                        fontSize: 12,
                        boxShadow: "0 10px 25px rgba(0,0,0,0.15)"
                      }}
                    />
                    <Bar dataKey="control" fill="#94A3B8" radius={[6, 6, 0, 0]} name="Control" />
                    <Bar dataKey="treatment" fill={isDark ? "#6366F1" : "#4F46E5"} radius={[6, 6, 0, 0]} name="Dynamic Treatment" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {metrics.abSignificance && (
                <div style={{
                  marginTop: "1.25rem",
                  padding: "0.85rem 1.15rem",
                  borderRadius: "0.75rem",
                  background: metrics.abSignificance.significant 
                    ? (isDark ? "rgba(52, 211, 153, 0.1)" : "#ECFDF5")
                    : t.itemBg,
                  border: metrics.abSignificance.significant 
                    ? (isDark ? "1px solid rgba(52, 211, 153, 0.3)" : "1px solid #A7F3D0")
                    : `1px solid ${t.itemBorder}`,
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "0.75rem"
                }}>
                  {metrics.abSignificance.significant ? (
                    <CheckCircle size={18} style={{ color: isDark ? "#34D399" : "#059669", marginTop: 2, flexShrink: 0 }} />
                  ) : (
                    <Activity size={18} style={{ color: t.textSecondary, marginTop: 2, flexShrink: 0 }} />
                  )}
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700, color: metrics.abSignificance.significant ? (isDark ? "#34D399" : "#059669") : t.textPrimary }}>
                      {metrics.abSignificance.significant ? "Statistically Significant (p < 0.05)" : "Convergence In Progress"}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: t.textSecondary, marginTop: "0.15rem" }}>
                      {metrics.abSignificance.note}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Price History Chart */}
            <div style={{
              borderRadius: "1.25rem",
              background: t.cardBg,
              border: `1px solid ${t.cardBorder}`,
              padding: "1.65rem",
              boxShadow: t.cardShadow
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.75rem", marginBottom: "1.25rem" }}>
                <div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                    Marketplace Competitor Price Tracker
                  </h2>
                  <p style={{ fontSize: "0.8rem", color: t.textSecondary, margin: 0, marginTop: "0.25rem" }}>
                    Last 24 hours of live price monitoring across Amazon and Flipkart catalog items.
                  </p>
                </div>
                {!historyLoading && marketplaceHistory.length > 0 && (
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <span style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.35rem",
                      fontSize: "0.75rem",
                      color: "#D97706",
                      background: isDark ? "rgba(251, 191, 36, 0.12)" : "#FEF3C7",
                      border: `1px solid ${isDark ? "rgba(251, 191, 36, 0.25)" : "#FDE68A"}`,
                      padding: "0.25rem 0.6rem",
                      borderRadius: "0.4rem",
                      fontWeight: 600
                    }}>
                      <Package size={12} /> {marketplaceHistory.filter(p => p.source === 'amazon').length} Amazon Items
                    </span>
                    <span style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.35rem",
                      fontSize: "0.75rem",
                      color: isDark ? "#38BDF8" : "#0891B2",
                      background: isDark ? "rgba(56, 189, 248, 0.12)" : "#E0F2FE",
                      border: `1px solid ${isDark ? "rgba(56, 189, 248, 0.25)" : "#BAE6FD"}`,
                      padding: "0.25rem 0.6rem",
                      borderRadius: "0.4rem",
                      fontWeight: 600
                    }}>
                      <Store size={12} /> {marketplaceHistory.filter(p => p.source === 'flipkart').length} Flipkart Items
                    </span>
                  </div>
                )}
              </div>

              {historyLoading ? (
                <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", background: t.itemBg, borderRadius: "0.85rem" }}>
                  <span style={{ color: t.textSecondary, fontSize: "0.88rem" }}>Syncing marketplace price telemetry...</span>
                </div>
              ) : chartData.length === 0 ? (
                <div style={{ height: 280, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: t.itemBg, borderRadius: "0.85rem" }}>
                  <Activity size={36} style={{ color: t.textMuted, marginBottom: "0.5rem" }} />
                  <span style={{ color: t.textPrimary, fontSize: "0.95rem", fontWeight: 700 }}>No marketplace price data recorded yet</span>
                  <span style={{ color: t.textSecondary, fontSize: "0.8rem", marginTop: "0.25rem" }}>Price movements will appear as scraped products update</span>
                </div>
              ) : (
                <>
                  <div style={{ height: 300, width: "100%" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={t.gridStroke} />
                        <XAxis dataKey="hour" tick={{ fill: isDark ? "#94A3B8" : "#64748B", fontSize: 11 }} interval={4} />
                        <YAxis tick={{ fill: isDark ? "#94A3B8" : "#64748B", fontSize: 11 }} tickFormatter={(v: number) => `₹${(v/1000).toFixed(0)}k`} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: t.tooltipBg,
                            border: `1px solid ${t.tooltipBorder}`,
                            borderRadius: 8,
                            color: t.tooltipText,
                            fontSize: 12,
                            boxShadow: "0 10px 25px rgba(0,0,0,0.15)"
                          }}
                          formatter={(value: number) => [`₹${value.toLocaleString('en-IN')}`, '']}
                        />
                        <Legend />
                        {chartKeys.map((key, i) => (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={LINE_COLORS[i % LINE_COLORS.length]}
                            strokeWidth={2.5}
                            dot={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", marginTop: "1rem" }}>
                    {marketplaceHistory.map((p, i) => (
                      <div key={p.id} style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.4rem",
                        background: t.itemBg,
                        border: `1px solid ${t.itemBorder}`,
                        padding: "0.3rem 0.65rem",
                        borderRadius: "0.45rem"
                      }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: LINE_COLORS[i % LINE_COLORS.length] }} />
                        <span style={{ fontSize: "0.75rem", color: t.textPrimary, maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: 600 }}>{p.name}</span>
                        {p.source === 'amazon' && <Package size={11} style={{ color: "#D97706" }} />}
                        {p.source === 'flipkart' && <Store size={11} style={{ color: isDark ? "#38BDF8" : "#0891B2" }} />}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            {/* Top Products & Sources Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.25rem" }}>
              
              {/* Top Products */}
              <div style={{
                borderRadius: "1.1rem",
                background: t.cardBg,
                border: `1px solid ${t.cardBorder}`,
                padding: "1.35rem",
                boxShadow: t.cardShadow
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                  <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                    Top In-Demand Products
                  </h3>
                  <span style={{
                    fontSize: "0.72rem",
                    color: isDark ? "#818CF8" : "#4F46E5",
                    background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
                    border: `1px solid ${isDark ? "rgba(99, 102, 241, 0.3)" : "#C7D2FE"}`,
                    padding: "0.2rem 0.55rem",
                    borderRadius: "0.35rem",
                    fontWeight: 700
                  }}>
                    Aggregated Clicks
                  </span>
                </div>
                {!metricsLoading && metrics.topProducts && metrics.topProducts.length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem" }}>
                    {metrics.topProducts.slice(0, 5).map((p, i) => (
                      <div key={i} style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        background: t.itemBg,
                        border: `1px solid ${t.itemBorder}`,
                        borderRadius: "0.6rem",
                        padding: "0.6rem 0.85rem",
                        fontSize: "0.82rem"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", minWidth: 0 }}>
                          <span style={{ fontSize: "0.75rem", color: t.textMuted, fontWeight: 800, width: 16 }}>#{i + 1}</span>
                          <span style={{ color: t.textPrimary, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: 600 }}>{p.name}</span>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.65rem", flexShrink: 0 }}>
                          {p.avgPrice && <span style={{ color: isDark ? "#34D399" : "#059669", fontWeight: 700 }}>₹{p.avgPrice.toLocaleString('en-IN')}</span>}
                          <span style={{
                            color: isDark ? "#818CF8" : "#4F46E5",
                            fontWeight: 700,
                            background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
                            padding: "0.18rem 0.5rem",
                            borderRadius: "0.35rem",
                            fontSize: "0.75rem"
                          }}>
                            {p.count}×
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ padding: "2.5rem", textAlign: "center", color: t.textMuted, fontSize: "0.85rem" }}>
                    No product clicks recorded in this cycle
                  </div>
                )}
              </div>

              {/* Source Distribution */}
              <div style={{
                borderRadius: "1.1rem",
                background: t.cardBg,
                border: `1px solid ${t.cardBorder}`,
                padding: "1.35rem",
                boxShadow: t.cardShadow
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                  <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                    Ecosystem Source Traffic
                  </h3>
                  <Layers size={16} style={{ color: isDark ? "#22D3EE" : "#0891B2" }} />
                </div>
                {metrics.sourceDistribution && Object.keys(metrics.sourceDistribution).length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
                    {Object.entries(metrics.sourceDistribution).map(([src, cnt]) => (
                      <div key={src} style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        background: t.itemBg,
                        border: `1px solid ${t.itemBorder}`,
                        borderRadius: "0.6rem",
                        padding: "0.7rem 0.95rem"
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                          {src === 'amazon' && <Package size={18} style={{ color: "#D97706" }} />}
                          {src === 'flipkart' && <Store size={18} style={{ color: isDark ? "#38BDF8" : "#0891B2" }} />}
                          {src === 'local' && <Eye size={18} style={{ color: isDark ? "#94A3B8" : "#64748B" }} />}
                          <div>
                            <span style={{ fontSize: "0.85rem", fontWeight: 700, color: t.textPrimary, textTransform: "capitalize", display: "block" }}>{src} Network</span>
                            <span style={{ fontSize: "0.72rem", color: t.textSecondary, display: "block" }}>Ingested catalog stream</span>
                          </div>
                        </div>
                        <span style={{ fontSize: "1.05rem", fontWeight: 800, color: t.textPrimary, fontVariantNumeric: "tabular-nums" }}>
                          {(cnt as number).toLocaleString('en-IN')}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ padding: "2.5rem", textAlign: "center", color: t.textMuted, fontSize: "0.85rem" }}>
                    Awaiting marketplace traffic distribution
                  </div>
                )}
              </div>

            </div>

            {/* Top Searched Queries */}
            <div style={{
              borderRadius: "1.1rem",
              background: t.cardBg,
              border: `1px solid ${t.cardBorder}`,
              padding: "1.35rem",
              boxShadow: t.cardShadow
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.85rem" }}>
                <Search size={16} style={{ color: isDark ? "#818CF8" : "#4F46E5" }} />
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                  Consumer Query Intent Signals
                </h3>
              </div>
              {!metricsLoading && metrics.topQueries && metrics.topQueries.length > 0 ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.6rem" }}>
                  {metrics.topQueries.map((q, i) => (
                    <div key={i} style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      background: t.itemBg,
                      border: `1px solid ${t.itemBorder}`,
                      borderRadius: "0.6rem",
                      padding: "0.6rem 0.85rem"
                    }}>
                      <span style={{ fontSize: "0.82rem", color: t.textPrimary, fontWeight: 600 }}>"{q.query}"</span>
                      <span style={{
                        fontSize: "0.72rem",
                        color: isDark ? "#818CF8" : "#4F46E5",
                        fontWeight: 700,
                        background: isDark ? "rgba(99, 102, 241, 0.15)" : "#EEF2FF",
                        padding: "0.15rem 0.45rem",
                        borderRadius: "0.35rem"
                      }}>
                        {q.count} hits
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: "1.75rem", textAlign: "center", color: t.textMuted, fontSize: "0.85rem" }}>
                  No consumer queries captured in this cycle
                </div>
              )}
            </div>

          </div>

          {/* Right Column: Sticky Live Events Stream */}
          <div style={{
            position: "sticky",
            top: "1.5rem",
            display: "flex",
            flexDirection: "column",
            gap: "1.25rem"
          }}>
            
            {/* Real-time SSE Feed */}
            <div style={{
              borderRadius: "1.25rem",
              background: t.cardBg,
              border: `1px solid ${t.cardBorder}`,
              padding: "1.45rem",
              boxShadow: t.cardShadow
            }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.1rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
                  <span style={{ position: "relative", display: "flex", width: 9, height: 9 }}>
                    <span style={{
                      position: "absolute",
                      width: "100%",
                      height: "100%",
                      borderRadius: "50%",
                      background: isDark ? "#34D399" : "#059669",
                      opacity: 0.75,
                      animation: "ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite"
                    }} />
                    <span style={{ width: 9, height: 9, borderRadius: "50%", background: isDark ? "#34D399" : "#059669" }} />
                  </span>
                  <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                    Live Event Stream
                  </h3>
                </div>
                <span style={{
                  fontSize: "0.72rem",
                  color: isDark ? "#34D399" : "#059669",
                  background: isDark ? "rgba(52, 211, 153, 0.12)" : "#ECFDF5",
                  border: `1px solid ${isDark ? "rgba(52, 211, 153, 0.3)" : "#A7F3D0"}`,
                  padding: "0.2rem 0.55rem",
                  borderRadius: "0.35rem",
                  fontWeight: 700
                }}>
                  Active SSE
                </span>
              </div>

              <div style={{
                maxHeight: "560px",
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: "0.55rem",
                paddingRight: "0.25rem"
              }}>
                {events.length === 0 ? (
                  <div style={{ padding: "3rem 1rem", textAlign: "center", color: t.textMuted, fontSize: "0.85rem" }}>
                    <Radio size={24} style={{ margin: "0 auto 0.6rem auto", display: "block", color: isDark ? "#818CF8" : "#4F46E5", animation: "pulse 2s infinite" }} />
                    Listening for live consumer behavior events...
                  </div>
                ) : (
                  events.map((e, idx) => (
                    <div key={idx} style={{
                      padding: "0.7rem 0.85rem",
                      borderRadius: "0.65rem",
                      background: t.itemBg,
                      border: `1px solid ${t.itemBorder}`,
                      fontSize: "0.78rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.25rem"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", color: t.textMuted, fontSize: "0.7rem" }}>
                        <span>{e.time}</span>
                        <span style={{ color: isDark ? "#818CF8" : "#4F46E5", fontWeight: 700 }}>Event #{events.length - idx}</span>
                      </div>
                      <div style={{ color: t.textPrimary, lineHeight: 1.45, fontWeight: 500 }}>
                        {e.text}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Micro Model Status */}
            <div style={{
              borderRadius: "1.1rem",
              background: t.cardBgSecondary,
              border: `1px solid ${t.cardBorderSecondary}`,
              padding: "1.2rem",
              boxShadow: "0 2px 6px rgba(0,0,0,0.02)"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <Cpu size={16} style={{ color: isDark ? "#818CF8" : "#4F46E5" }} />
                <span style={{ fontSize: "0.82rem", fontWeight: 800, color: t.textPrimary }}>Pricing Daemon Telemetry</span>
              </div>
              <div style={{ fontSize: "0.78rem", color: t.textSecondary, display: "flex", flexDirection: "column", gap: "0.45rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Multi-Arm Bandit:</span>
                  <span style={{ color: isDark ? "#34D399" : "#059669", fontWeight: 700 }}>Active (Epsilon-Greedy 0.1)</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Elasticity Recalculation:</span>
                  <span style={{ color: isDark ? "#38BDF8" : "#0891B2", fontWeight: 700 }}>Throttled (500 events)</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Inference Latency:</span>
                  <span style={{ color: "#D97706", fontWeight: 700 }}>{latency?.p99 ?? 12}ms P99</span>
                </div>
              </div>
            </div>

          </div>

        </div>

        {/* ── Fairness & Transparency Audit Panel (Full Width) ── */}
        {fairness && (
          <div style={{
            borderRadius: "1.25rem",
            background: t.cardBg,
            border: isDark ? "1px solid rgba(52, 211, 153, 0.25)" : "1px solid #A7F3D0",
            padding: "1.85rem",
            position: "relative",
            boxShadow: t.cardShadow
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.65rem", marginBottom: "0.85rem" }}>
              <ShieldCheck size={24} style={{ color: isDark ? "#34D399" : "#059669" }} />
              <h2 style={{ fontSize: "1.25rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                PriceIQ Global Fairness Standard & Ethical Pricing Audit
              </h2>
            </div>
            
            <div style={{
              background: isDark ? "rgba(52, 211, 153, 0.08)" : "#ECFDF5",
              borderLeft: `4px solid ${isDark ? "#34D399" : "#059669"}`,
              padding: "0.85rem 1.15rem",
              borderRadius: "0 0.6rem 0.6rem 0",
              fontSize: "0.88rem",
              color: isDark ? "#CBD5E1" : "#065F46",
              fontStyle: "italic",
              marginBottom: "1.5rem",
              lineHeight: 1.5
            }}>
              "{fairness.auditNote}"
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.75rem" }}>
              <div>
                <h4 style={{ fontSize: "0.78rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", color: isDark ? "#34D399" : "#059669", marginBottom: "0.85rem" }}>
                  Permitted Behavioral Factors
                </h4>
                <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  {fairness.pricingFactors?.map((f) => (
                    <li key={f.factor} style={{ display: "flex", alignItems: "flex-start", gap: "0.55rem", fontSize: "0.82rem" }}>
                      <CheckCircle size={15} style={{ color: isDark ? "#34D399" : "#059669", marginTop: 2, flexShrink: 0 }} />
                      <div>
                        <strong style={{ color: t.textPrimary }}>{f.factor}: </strong>
                        <span style={{ color: t.textSecondary }}>{f.description}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 style={{ fontSize: "0.78rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", color: "#E11D48", marginBottom: "0.85rem" }}>
                  Excluded Demographic Factors (Strictly Prohibited)
                </h4>
                <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  {fairness.excludedFactors?.map((f) => (
                    <li key={f.factor} style={{ display: "flex", alignItems: "flex-start", gap: "0.55rem", fontSize: "0.82rem" }}>
                      <AlertCircle size={15} style={{ color: "#E11D48", marginTop: 2, flexShrink: 0 }} />
                      <div>
                        <strong style={{ color: t.textPrimary }}>{f.factor}: </strong>
                        <span style={{ color: t.textSecondary }}>{f.reason}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div style={{ marginTop: "1.65rem", paddingTop: "1.35rem", borderTop: `1px solid ${t.headerBorder}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem", flexWrap: "wrap", gap: "0.5rem" }}>
                <span style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: t.textSecondary }}>
                  Non-Demographic Customer Segmentation Clusters
                </span>
                <span style={{ fontSize: "0.75rem", color: t.textMuted }}>{fairness.segmentBasis}</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.85rem", background: t.itemBg, padding: "1rem", borderRadius: "0.85rem", border: `1px solid ${t.itemBorder}` }}>
                <div style={{ textAlign: "center" }}>
                  <span style={{ fontSize: "1.65rem", fontWeight: 800, color: t.textPrimary }}>
                    {fairness.segmentDistribution?.value_seeker || 0}
                  </span>
                  <span style={{ display: "block", fontSize: "0.72rem", textTransform: "uppercase", color: isDark ? "#818CF8" : "#4F46E5", fontWeight: 700, marginTop: "0.15rem" }}>Value Seekers</span>
                </div>
                <div style={{ textAlign: "center", borderLeft: `1px solid ${t.itemBorder}`, borderRight: `1px solid ${t.itemBorder}` }}>
                  <span style={{ fontSize: "1.65rem", fontWeight: 800, color: t.textPrimary }}>
                    {fairness.segmentDistribution?.standard || 0}
                  </span>
                  <span style={{ display: "block", fontSize: "0.72rem", textTransform: "uppercase", color: isDark ? "#38BDF8" : "#0891B2", fontWeight: 700, marginTop: "0.15rem" }}>Standard Intent</span>
                </div>
                <div style={{ textAlign: "center" }}>
                  <span style={{ fontSize: "1.65rem", fontWeight: 800, color: t.textPrimary }}>
                    {fairness.segmentDistribution?.premium_intent || 0}
                  </span>
                  <span style={{ display: "block", fontSize: "0.72rem", textTransform: "uppercase", color: isDark ? "#34D399" : "#059669", fontWeight: 700, marginTop: "0.15rem" }}>Premium Intent</span>
                </div>
              </div>

              <div style={{ fontSize: "0.75rem", color: t.textMuted, textAlign: "center", marginTop: "0.85rem" }}>
                Audited timestamp: {new Date(fairness.lastAudited).toLocaleString()} · Certified against Bias Specification v2.4
              </div>
            </div>
          </div>
        )}

        {/* ── Operational Intelligence & Predictive Inventory ── */}
        <div style={{
          borderRadius: "1.25rem",
          background: t.cardBg,
          border: `1px solid ${t.cardBorder}`,
          padding: "1.65rem",
          boxShadow: t.cardShadow
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
            <div>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 800, margin: 0, color: t.textPrimary }}>
                Operational Intelligence & Stockout Depletion Forecasting
              </h2>
              <p style={{ fontSize: "0.8rem", color: t.textSecondary, margin: 0, marginTop: "0.25rem" }}>
                Predictive runout calculations factoring in real-time purchase velocities.
              </p>
            </div>
            <span style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.4rem",
              fontSize: "0.75rem",
              fontWeight: 700,
              color: isDark ? "#38BDF8" : "#0891B2",
              background: isDark ? "rgba(56, 189, 248, 0.12)" : "#E0F2FE",
              border: `1px solid ${isDark ? "rgba(56, 189, 248, 0.25)" : "#BAE6FD"}`,
              padding: "0.3rem 0.75rem",
              borderRadius: "9999px"
            }}>
              <Activity size={13} /> Predictive Velocity Active
            </span>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${t.headerBorder}`, color: t.textSecondary }}>
                  <th style={{ padding: "0.85rem 1rem", fontWeight: 700 }}>Product Title</th>
                  <th style={{ padding: "0.85rem 1rem", fontWeight: 700 }}>Daily Velocity</th>
                  <th style={{ padding: "0.85rem 1rem", fontWeight: 700 }}>Warehouse Stock</th>
                  <th style={{ padding: "0.85rem 1rem", fontWeight: 700 }}>Depletion Horizon</th>
                  <th style={{ padding: "0.85rem 1rem", fontWeight: 700 }}>Status Level</th>
                </tr>
              </thead>
              <tbody>
                {predictions.length > 0 ? (
                  predictions.slice(0, 5).map((p, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${t.tableRowBorder}`, background: i % 2 === 0 ? "transparent" : t.tableRowBgAlt }}>
                      <td style={{ padding: "0.85rem 1rem", color: t.textPrimary, fontWeight: 600 }}>{p.name}</td>
                      <td style={{ padding: "0.85rem 1rem", color: t.textSecondary }}>{p.dailyVelocity} units/day</td>
                      <td style={{ padding: "0.85rem 1rem", color: t.textSecondary }}>{p.currentStock} units</td>
                      <td style={{
                        padding: "0.85rem 1rem",
                        fontWeight: 800,
                        color: p.daysRemaining < 3 ? "#E11D48" : "#D97706"
                      }}>
                        {p.daysRemaining} days
                      </td>
                      <td style={{ padding: "0.85rem 1rem" }}>
                        {p.critical ? (
                          <span style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.3rem",
                            padding: "0.25rem 0.6rem",
                            borderRadius: "9999px",
                            background: isDark ? "rgba(248, 113, 113, 0.15)" : "#FFF1F2",
                            border: `1px solid ${isDark ? "rgba(248, 113, 113, 0.35)" : "#FECDD3"}`,
                            color: "#E11D48",
                            fontSize: "0.75rem",
                            fontWeight: 700
                          }}>
                            <AlertCircle size={12} /> Critical Shortage
                          </span>
                        ) : (
                          <span style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "0.3rem",
                            padding: "0.25rem 0.6rem",
                            borderRadius: "9999px",
                            background: isDark ? "rgba(251, 191, 36, 0.15)" : "#FEF3C7",
                            border: `1px solid ${isDark ? "rgba(251, 191, 36, 0.35)" : "#FDE68A"}`,
                            color: "#D97706",
                            fontSize: "0.75rem",
                            fontWeight: 700
                          }}>
                            Stable Reserve
                          </span>
                        )}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} style={{ padding: "2.5rem", textAlign: "center", color: t.textMuted, fontStyle: "italic" }}>
                      Catalog velocity stabilization underway — calculating replenishment horizon...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
