import {
  ArrowRight,
  ArrowUpRight,
  Activity,
  Check,
  GitBranch,
  Lock,
  Network,
  Radio,
  ShieldCheck,
  Sliders,
  Sparkles,
  TrendingUp,
  Zap,
  Layers,
  Cpu,
  X,
  Clock,
} from "lucide-react";
import { Link } from "react-router-dom";
import { useMemo, useState, useRef, useEffect } from "react";
import LandingNavbar from "@/components/landing/LandingNavbar";
import { useTheme } from "@/contexts/ThemeContext";
import {
  METHOD_NOTES,
  PREVIEWS,
  PRICING_RULES,
  REPOSITORY_URL,
  SNAPSHOT,
  TECH_STACK,
  WORKED_PRICE,
  type BenchmarkRow,
} from "@/landing/claims";
import { useLandingSignals } from "@/landing/useLandingSignals";
import { useReveal } from "@/landing/useReveal";

type PreviewTab = keyof typeof PREVIEWS;

const formatPercent = (value: number) => `${(value * 100).toFixed(2).replace(/\.00$/, "")}%`;
const formatMilliseconds = (value: number) =>
  `${value.toFixed(value < 1 ? 2 : 2).replace(/0+$/, "").replace(/\.$/, "")} ms`;
const formatIndianCurrency = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);

// ── Scroll-Driven Sticky Card Reveal ──────────────────────────────────────────
// ── Scroll-Driven Sticky Card Reveal (Optimized & Butter-Smooth) ────────────
function ScrollRevealCards({ cards }: { cards: { icon: React.ReactNode; tag: string; val: string; desc: string; accent: string }[] }) {
  const { isDark } = useTheme();
  const sectionRef = useRef<HTMLDivElement>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [modalCard, setModalCard] = useState<number | null>(null);
  const [countdown, setCountdown] = useState(30);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const frontCardRef = useRef<HTMLDivElement>(null);

  const openModal = (idx: number) => {
    setModalCard(idx);
    setCountdown(30);
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownRef.current!);
          setModalCard(null);
          return 30;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const closeModal = () => {
    setModalCard(null);
    if (countdownRef.current) clearInterval(countdownRef.current);
    setCountdown(30);
  };

  const handleFrontMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePos({
      x: ((e.clientX - rect.left) / rect.width) * 100,
      y: ((e.clientY - rect.top) / rect.height) * 100,
    });
  };

  useEffect(() => {
    return () => { if (countdownRef.current) clearInterval(countdownRef.current); };
  }, []);

  useEffect(() => {
    let ticking = false;
    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const section = sectionRef.current;
          if (!section) {
            ticking = false;
            return;
          }
          const rect = section.getBoundingClientRect();
          const totalScrollable = section.offsetHeight - window.innerHeight;
          if (totalScrollable > 0) {
            const scrolled = -rect.top;
            const progress = Math.max(0, Math.min(1, scrolled / totalScrollable));
            const idx = Math.min(cards.length - 1, Math.floor(progress * cards.length * 0.999));
            setActiveIndex(idx);
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [cards.length]);

  const pct = ((activeIndex + 1) / cards.length) * 100;

  return (
    <div ref={sectionRef} style={{ height: `${cards.length * 65}vh`, position: "relative" }}>
      <div style={{
        position: "sticky",
        top: 0,
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        zIndex: 2,
      }}>
        {/* Progress bar */}
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "rgba(37,45,64,0.5)", zIndex: 5 }}>
          <div style={{ height: "100%", background: "linear-gradient(90deg, #6366F1, #22D3EE)", width: `${pct}%`, transition: "width 300ms ease" }} />
        </div>

        {/* Quick-Select Clickable Card Tabs */}
        <div style={{ position: "absolute", top: 28, right: 28, display: "flex", gap: 8, zIndex: 12 }}>
          {cards.map((c, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setActiveIndex(i)}
              style={{
                padding: "6px 14px",
                borderRadius: 9999,
                background: i === activeIndex ? `${c.accent}25` : "rgba(255,255,255,0.04)",
                border: `1px solid ${i === activeIndex ? c.accent : "rgba(255,255,255,0.1)"}`,
                color: i === activeIndex ? "#FFFFFF" : "#94A3B8",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                fontWeight: 700,
                cursor: "pointer",
                transition: "all 200ms ease",
                display: "flex",
                alignItems: "center",
                gap: 6
              }}
            >
              <span style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: i === activeIndex ? c.accent : "#64748B",
                boxShadow: i === activeIndex ? `0 0 8px ${c.accent}` : "none"
              }} />
              0{i + 1}
            </button>
          ))}
        </div>

        {/* Scroll hint */}
        <div style={{ position: "absolute", bottom: 36, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 6, zIndex: 5, opacity: activeIndex === cards.length - 1 ? 0 : 0.6, transition: "opacity 300ms" }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: "#6B7280", letterSpacing: "0.1em", textTransform: "uppercase" }}>Scroll or click tabs</span>
          <div style={{ width: 1, height: 36, background: "linear-gradient(to bottom, #6366F1, transparent)", animation: "scroll-pulse 1.5s ease-in-out infinite" }} />
        </div>

        {/* 3D Stacked Modular Command Deck */}
        <div style={{
          position: "relative",
          width: "min(920px, 94vw)",
          height: 500,
          perspective: 1000,
          perspectiveOrigin: "50% 35%",
        }}>
          {cards.map((c, i) => {
            const offset = i - activeIndex;

            // Hardware accelerated transforms
            let transform = "translate3d(0, 0, 0) scale(1)";
            let opacity = 0;
            let zIndex = 1;
            let filter = "none";
            let pointerEvents: "auto" | "none" = "none";

            if (offset < 0) {
              transform = "translate3d(0, -90px, 0) scale(0.95)";
              opacity = 0;
              zIndex = 1;
            } else if (offset === 0) {
              transform = "translate3d(0, 0, 0) scale(1)";
              opacity = 1;
              zIndex = 10;
              pointerEvents = "auto";
              filter = "none";
            } else if (offset === 1) {
              transform = "translate3d(0, -44px, 0) scale(0.96)";
              opacity = 0.85;
              zIndex = 9;
              filter = "brightness(0.85)";
            } else if (offset === 2) {
              transform = "translate3d(0, -84px, 0) scale(0.92)";
              opacity = 0.60;
              zIndex = 8;
              filter = "brightness(0.70)";
            } else if (offset === 3) {
              transform = "translate3d(0, -120px, 0) scale(0.88)";
              opacity = 0.38;
              zIndex = 7;
              filter = "brightness(0.55)";
            } else {
              opacity = 0;
              zIndex = 1;
            }

            // High-impact domain data tailored for each card
            const cardDetails = [
              {
                eyebrow: "CATALOG INGESTION ENGINE",
                title: "Live Multi-Channel Catalog Sync",
                unit: "ACTIVE SKUs MONITORED",
                chips: ["MongoDB Atlas M10", "Amazon & Flipkart Ingestion", "Real-Time Stock Velocity"],
                renderWidget: () => (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#6366F1", fontWeight: 700, letterSpacing: "0.08em" }}>
                        ● REAL-TIME ATLAS SYNC PIPELINE
                      </span>
                      <span style={{ fontSize: 10, color: "#34D399", background: "rgba(52,211,153,0.12)", padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>
                        99.98% UP
                      </span>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8", marginBottom: 4 }}>
                          <span>Amazon Live Price Feed</span>
                          <span style={{ color: "#FBBF24", fontWeight: 700 }}>184 Products (100%)</span>
                        </div>
                        <div style={{ height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: "100%", background: "linear-gradient(90deg, #6366F1, #FBBF24)", borderRadius: 999 }} />
                        </div>
                      </div>

                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8", marginBottom: 4 }}>
                          <span>Flipkart Marketplace Channel</span>
                          <span style={{ color: "#38BDF8", fontWeight: 700 }}>31 Products (100%)</span>
                        </div>
                        <div style={{ height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: "100%", background: "linear-gradient(90deg, #6366F1, #38BDF8)", borderRadius: 999 }} />
                        </div>
                      </div>

                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8", marginBottom: 4 }}>
                          <span>In-Memory Redis Cache RTT</span>
                          <span style={{ color: "#34D399", fontWeight: 700 }}>0.42 ms latency</span>
                        </div>
                        <div style={{ height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: "95%", background: "linear-gradient(90deg, #6366F1, #34D399)", borderRadius: 999 }} />
                        </div>
                      </div>
                    </div>

                    <div style={{ background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "#CBD5E1" }}>
                      ⚡ <strong>Direct Ingestion:</strong> Zero spreadsheets. Real catalog products continuously update before pricing calculations execute.
                    </div>
                  </div>
                ),
              },
              {
                eyebrow: "DEEP LEARNING EMBEDDING MATRIX",
                title: "High-Dimensional Behavioral Latent Space",
                unit: "LEARNED SEMANTIC TOKENS",
                chips: ["128-Dim Embedding Space", "Cosine Intent Matching", "Category Graph Affinity"],
                renderWidget: () => (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#22D3EE", fontWeight: 700, letterSpacing: "0.08em" }}>
                        ● LATENT CLUSTER EMBEDDINGS (128-D)
                      </span>
                      <span style={{ fontSize: 10, color: "#22D3EE", background: "rgba(34,211,238,0.12)", padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>
                        50,000 VECTORS
                      </span>
                    </div>

                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      <span style={{ background: "rgba(34,211,238,0.12)", border: "1px solid rgba(34,211,238,0.3)", color: "#22D3EE", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                        👟 [Running Shoes: 0.94 cosine]
                      </span>
                      <span style={{ background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.3)", color: "#818CF8", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                        ⚡ [Speed Boost: 0.89 affinity]
                      </span>
                      <span style={{ background: "rgba(52,211,153,0.12)", border: "1px solid rgba(52,211,153,0.3)", color: "#34D399", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                        🏷️ [Price Elasticity: -1.24]
                      </span>
                      <span style={{ background: "rgba(244,114,182,0.12)", border: "1px solid rgba(244,114,182,0.3)", color: "#F472B6", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                        🌧️ [Weather Proof: 0.82]
                      </span>
                      <span style={{ background: "rgba(251,191,36,0.12)", border: "1px solid rgba(251,191,36,0.3)", color: "#FBBF24", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
                        🔍 [High-Intent Shopper: 0.91]
                      </span>
                    </div>

                    <div style={{ background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.2)", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "#CBD5E1" }}>
                      🧠 <strong>Dense Representation:</strong> Captures shopper affinity vectors across 50,000 items in real time for precision pricing.
                    </div>
                  </div>
                ),
              },
              {
                eyebrow: "RECURRENT INFERENCE BENCHMARK",
                title: "Sub-Millisecond Neural Clickstream Inference",
                unit: "P95 BENCHMARKED LATENCY",
                chips: ["Sub-Frame Render Delivery", "Sequential GRU4Rec Engine", "14× Faster Than Batch"],
                renderWidget: () => (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#A78BFA", fontWeight: 700, letterSpacing: "0.08em" }}>
                        ● INFERENCE LATENCY OSCILLOSCOPE
                      </span>
                      <span style={{ fontSize: 10, color: "#A78BFA", background: "rgba(167,139,250,0.12)", padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>
                        14× FASTER
                      </span>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "#94A3B8" }}>
                        <span>Session Clickstream Ingest</span>
                        <span style={{ color: "#FFFFFF", fontWeight: 700 }}>2.1 ms</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "#94A3B8" }}>
                        <span>GRU4Rec Recurrent Forward Pass</span>
                        <span style={{ color: "#FFFFFF", fontWeight: 700 }}>8.4 ms</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: "#94A3B8" }}>
                        <span>MRP Guardrail Policy Check</span>
                        <span style={{ color: "#FFFFFF", fontWeight: 700 }}>3.5 ms</span>
                      </div>
                      <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 6, display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12, fontWeight: 700 }}>
                        <span style={{ color: "#CBD5E1" }}>Total End-to-End Decision</span>
                        <span style={{ color: "#A78BFA", fontSize: 14 }}>14.0 ms (P95)</span>
                      </div>
                    </div>

                    <div style={{ background: "rgba(167,139,250,0.08)", border: "1px solid rgba(167,139,250,0.2)", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "#CBD5E1" }}>
                      ⏱️ <strong>Instant Delivery:</strong> Calculated and delivered inside browser request cycle before viewport paint completes.
                    </div>
                  </div>
                ),
              },
              {
                eyebrow: "AUTONOMOUS SAFETY GOVERNANCE",
                title: "Mathematical Anti-Gouging Guardrails",
                unit: "BOUNDED COMPLIANCE GUARANTEE",
                chips: ["Strict 70% MRP Floor", "Hard 100% MRP Ceiling", "Certified Ethical Standard"],
                renderWidget: () => (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                      <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#34D399", fontWeight: 700, letterSpacing: "0.08em" }}>
                        ● BOUNDED PRICING CORRIDOR
                      </span>
                      <span style={{ fontSize: 10, color: "#34D399", background: "rgba(52,211,153,0.12)", padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>
                        100% COMPLIANT
                      </span>
                    </div>

                    <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "12px", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#64748B", marginBottom: 6, textTransform: "uppercase" }}>
                        <span>Hard Floor (70% MRP)</span>
                        <span style={{ color: "#34D399", fontWeight: 700 }}>Active Price</span>
                        <span>Hard Ceiling (100% MRP)</span>
                      </div>
                      <div style={{ height: 8, background: "rgba(255,255,255,0.08)", borderRadius: 999, position: "relative" }}>
                        <div style={{ position: "absolute", left: "20%", right: "15%", top: 0, bottom: 0, background: "linear-gradient(90deg, #6366F1, #34D399)", borderRadius: 999 }} />
                        <div style={{ position: "absolute", left: "62%", top: -4, width: 16, height: 16, borderRadius: "50%", background: "#34D399", border: "2px solid #FFFFFF", boxShadow: "0 0 10px #34D399" }} />
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8", marginTop: 8 }}>
                        <span>₹1,609 (Min Floor)</span>
                        <span style={{ color: "#34D399", fontWeight: 800 }}>₹2,099 (+14% Surge)</span>
                        <span>₹2,299 (Max Cap)</span>
                      </div>
                    </div>

                    <div style={{ background: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.2)", borderRadius: 8, padding: "8px 12px", fontSize: 11, color: "#CBD5E1" }}>
                      🛡️ <strong>Zero Exceptions:</strong> Strict mathematical boundary checks lock every price strictly inside non-predatory parameters.
                    </div>
                  </div>
                ),
              },
            ];

            const detail = cardDetails[i] || cardDetails[0];

            return (
              <div
                key={i}
                style={{
                  position: "absolute",
                  inset: 0,
                  transform,
                  opacity,
                  zIndex,
                  filter,
                  pointerEvents,
                  transition: "transform 650ms cubic-bezier(0.16, 1, 0.3, 1), opacity 550ms cubic-bezier(0.16, 1, 0.3, 1), filter 550ms ease",
                  transformOrigin: "center top",
                }}
              >
                {/* ── If Card is Behind: Render Elevated Architectural Command Tab ── */}
                {offset > 0 ? (
                  <div
                    onClick={() => openModal(i)}
                    onMouseEnter={() => setHoveredCard(i)}
                    onMouseLeave={() => setHoveredCard(null)}
                    style={{
                    width: "100%",
                    height: hoveredCard === i ? 96 : 80,
                    borderRadius: "20px 20px 0 0",
                    background: isDark
                      ? (hoveredCard === i ? `linear-gradient(180deg, ${c.accent}18 0%, rgba(10, 14, 28, 0.98) 100%)` : `linear-gradient(180deg, rgba(20, 26, 46, 0.98) 0%, rgba(10, 14, 28, 0.98) 100%)`)
                      : (hoveredCard === i ? `linear-gradient(180deg, #FAFAFF 0%, #FFFFFF 100%)` : `linear-gradient(180deg, #FFFFFF 0%, #F0F2F8 100%)`),
                    borderTop: `2px solid ${hoveredCard === i ? c.accent : isDark ? c.accent + "99" : "#E2E8F0"}`,
                    borderLeft: `1px solid ${hoveredCard === i ? c.accent + "80" : isDark ? c.accent + "40" : "#E2E8F0"}`,
                    borderRight: `1px solid ${hoveredCard === i ? c.accent + "80" : isDark ? c.accent + "40" : "#E2E8F0"}`,
                    boxShadow: isDark
                      ? (hoveredCard === i ? `0 -20px 50px rgba(0,0,0,0.8), 0 0 60px ${c.accent}35, 0 0 20px ${c.accent}20` : `0 -10px 30px rgba(0,0,0,0.7), 0 0 25px ${c.accent}15`)
                      : (hoveredCard === i ? `0 -10px 25px rgba(17, 24, 39, 0.08), 0 0 30px ${c.accent}15` : `0 -4px 15px rgba(17, 24, 39, 0.04)`),
                    padding: "14px 28px",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    cursor: "pointer",
                    transition: "all 300ms cubic-bezier(0.16, 1, 0.3, 1)",
                    transform: hoveredCard === i ? "translateY(-6px)" : "translateY(0)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                      <span style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "4px 10px",
                        borderRadius: 9999,
                        background: hoveredCard === i ? `${c.accent}35` : `${c.accent}20`,
                        border: `1px solid ${hoveredCard === i ? c.accent : c.accent + "50"}`,
                        color: c.accent,
                        fontSize: 11,
                        fontWeight: 700,
                        fontFamily: "'JetBrains Mono', monospace",
                        letterSpacing: "0.08em",
                        boxShadow: hoveredCard === i ? `0 0 12px ${c.accent}60` : "none",
                        transition: "all 250ms ease",
                      }}>
                        <span style={{
                          width: 6, height: 6, borderRadius: "50%", background: c.accent,
                          boxShadow: hoveredCard === i ? `0 0 14px ${c.accent}` : `0 0 6px ${c.accent}`,
                          animation: hoveredCard === i ? "pulse-dot 1s ease-in-out infinite" : "none",
                        }} />
                        LAYER 0{i + 1}
                      </span>
                      <span style={{ fontSize: 13, fontWeight: 700, color: isDark ? (hoveredCard === i ? "#FFFFFF" : "#CBD5E1") : (hoveredCard === i ? "#111827" : "#475569"), letterSpacing: "0.04em", textTransform: "uppercase", fontFamily: "'Outfit', sans-serif", transition: "color 200ms ease" }}>
                        {c.tag}
                      </span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                      <span style={{
                        fontSize: hoveredCard === i ? 16 : 14, fontWeight: 800, color: c.accent,
                        fontFamily: "'Plus Jakarta Sans', sans-serif",
                        textShadow: hoveredCard === i ? `0 0 20px ${c.accent}` : "none",
                        transition: "all 250ms ease",
                      }}>
                        {c.val}
                      </span>
                      <span style={{
                        fontSize: 11,
                        color: hoveredCard === i ? "rgba(148,163,184,0.9)" : "rgba(148, 163, 184, 0.6)",
                        fontFamily: "'JetBrains Mono', monospace",
                        letterSpacing: "0.08em",
                        transition: "color 200ms ease",
                      }}>
                        {hoveredCard === i ? "↑ CLICK TO EXPAND" : "[QUEUED IN STACK]"}
                      </span>
                    </div>
                  </div>
                ) : offset === 0 ? (
                  /* ── If Card is Front: Full High-Tech Command Console — click to open modal ── */
                  <div
                    ref={frontCardRef}
                    onClick={() => openModal(i)}
                    onMouseEnter={() => setHoveredCard(i)}
                    onMouseLeave={() => { setHoveredCard(null); setMousePos({ x: 50, y: 50 }); }}
                    onMouseMove={handleFrontMouseMove}
                    style={{
                    width: "100%",
                    height: "100%",
                    borderRadius: 24,
                    background: isDark
                      ? (hoveredCard === i
                          ? `radial-gradient(ellipse at ${mousePos.x}% ${mousePos.y}%, ${c.accent}18 0%, rgba(17,23,42,0.97) 55%), linear-gradient(145deg, rgba(17, 23, 42, 0.96) 0%, rgba(9, 13, 26, 0.98) 100%)`
                          : "linear-gradient(145deg, rgba(17, 23, 42, 0.96) 0%, rgba(9, 13, 26, 0.98) 100%)")
                      : (hoveredCard === i
                          ? `radial-gradient(ellipse at ${mousePos.x}% ${mousePos.y}%, ${c.accent}10 0%, #FAFAFF 55%), linear-gradient(145deg, #FFFFFF 0%, #FAFAFF 100%)`
                          : "linear-gradient(145deg, #FFFFFF 0%, #FAFAFF 100%)"),
                    border: `1px solid ${hoveredCard === i ? c.accent : isDark ? c.accent + "60" : "#E2E8F0"}`,
                    boxShadow: isDark
                      ? (hoveredCard === i
                          ? `0 45px 100px -10px rgba(0,0,0,0.9), 0 0 80px ${c.accent}40, 0 0 30px ${c.accent}20, inset 0 1px 0 rgba(255,255,255,0.18)`
                          : `0 35px 80px -10px rgba(0,0,0,0.85), 0 0 50px ${c.accent}25, inset 0 1px 0 rgba(255,255,255,0.12)`)
                      : (hoveredCard === i
                          ? `0 25px 60px -10px rgba(17, 24, 39, 0.12), 0 0 40px ${c.accent}20, inset 0 1px 0 rgba(255,255,255,0.8)`
                          : `0 15px 40px -10px rgba(17, 24, 39, 0.06), 0 0 20px ${c.accent}10, inset 0 1px 0 rgba(255,255,255,0.8)`),
                    backdropFilter: "blur(30px)",
                    position: "relative",
                    overflow: "hidden",
                    padding: "36px 44px",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    cursor: "pointer",
                    transition: "box-shadow 350ms ease, border-color 350ms ease, transform 350ms cubic-bezier(0.16,1,0.3,1)",
                    transform: hoveredCard === i ? "translateY(-4px) scale(1.005)" : "translateY(0) scale(1)",
                  }}>
                    {/* Top neon accent line — brighter on hover */}
                    <div style={{
                      position: "absolute",
                      top: 0,
                      left: hoveredCard === i ? 16 : 32,
                      right: hoveredCard === i ? 16 : 32,
                      height: hoveredCard === i ? 3 : 2,
                      background: `linear-gradient(90deg, transparent, ${c.accent}, transparent)`,
                      boxShadow: hoveredCard === i ? `0 0 30px ${c.accent}, 0 0 60px ${c.accent}60` : `0 0 15px ${c.accent}`,
                      transition: "all 350ms ease",
                      borderRadius: 999,
                    }} />

                    {/* Mouse-tracking radial glow */}
                    {hoveredCard === i && (
                      <div style={{
                        position: "absolute",
                        left: `${mousePos.x}%`,
                        top: `${mousePos.y}%`,
                        width: 280,
                        height: 280,
                        borderRadius: "50%",
                        background: c.accent,
                        opacity: 0.08,
                        filter: "blur(60px)",
                        transform: "translate(-50%, -50%)",
                        pointerEvents: "none",
                        transition: "left 80ms ease, top 80ms ease",
                      }} />
                    )}

                    {/* Corner ambient glow — expands on hover */}
                    <div style={{
                      position: "absolute",
                      top: -60,
                      right: -60,
                      width: hoveredCard === i ? 340 : 260,
                      height: hoveredCard === i ? 340 : 260,
                      borderRadius: "50%",
                      background: c.accent,
                      opacity: hoveredCard === i ? 0.18 : 0.12,
                      filter: "blur(80px)",
                      pointerEvents: "none",
                      transition: "all 400ms ease",
                    }} />

                    {/* Header Bar */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          padding: "5px 12px",
                          borderRadius: 9999,
                          background: `${c.accent}18`,
                          border: `1px solid ${c.accent}45`,
                          color: c.accent,
                          fontFamily: "'JetBrains Mono', monospace",
                          fontSize: 11,
                          fontWeight: 700,
                          letterSpacing: "0.08em",
                          textTransform: "uppercase",
                        }}>
                          <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.accent, boxShadow: `0 0 8px ${c.accent}` }} />
                          {detail.eyebrow}
                        </span>
                        <span style={{ fontSize: 11, color: "#64748B", fontFamily: "'JetBrains Mono', monospace" }}>
                          CORE ENGINE 0{i + 1} OF 04
                        </span>
                      </div>

                      <div style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "4px 12px",
                        borderRadius: 8,
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.06)",
                        fontSize: 11,
                        color: "#94A3B8",
                        fontFamily: "'JetBrains Mono', monospace"
                      }}>
                        <span style={{ color: "#34D399" }}>● LIVE ENGINE</span>
                        <span style={{ opacity: 0.3 }}>|</span>
                        <span>LAYER 0{i + 1}/04</span>
                      </div>
                    </div>

                    {/* 2-Column Command Console Grid */}
                    <div style={{ display: "grid", gridTemplateColumns: "1.15fr 0.85fr", gap: 32, alignItems: "center", margin: "16px 0" }}>
                      {/* Left Side: Narrative & Big Number */}
                      <div>
                        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 8 }}>
                          <span style={{
                            fontFamily: "'Plus Jakarta Sans', sans-serif",
                            fontSize: "clamp(3.2rem, 5.5vw, 4.4rem)",
                            fontWeight: 900,
                            color: isDark ? "#FFFFFF" : "#111827",
                            letterSpacing: "-0.04em",
                            lineHeight: 1,
                          }}>
                            {c.val}
                          </span>
                          <span style={{
                            fontSize: 12,
                            fontWeight: 700,
                            color: c.accent,
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                            fontFamily: "'JetBrains Mono', monospace",
                            background: `${c.accent}15`,
                            padding: "3px 8px",
                            borderRadius: 4,
                            border: `1px solid ${c.accent}30`
                          }}>
                            {detail.unit}
                          </span>
                        </div>

                        <h3 style={{
                          fontSize: "1.25rem",
                          fontWeight: 700,
                          color: isDark ? "#F8FAFC" : "#111827",
                          margin: "0 0 10px 0",
                          letterSpacing: "-0.01em",
                          lineHeight: 1.3,
                        }}>
                          {detail.title}
                        </h3>

                        <p style={{
                          color: isDark ? "#94A3B8" : "#475569",
                          fontSize: 14,
                          lineHeight: 1.6,
                          margin: "0 0 16px 0",
                          maxWidth: 460,
                        }}>
                          {c.desc}
                        </p>

                        {/* Metadata Pills */}
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                          {detail.chips.map((ch, chIdx) => (
                            <span key={chIdx} style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 5,
                              padding: "4px 9px",
                              borderRadius: 6,
                              background: isDark ? "rgba(255,255,255,0.03)" : "#FFFFFF",
                              border: isDark ? "1px solid rgba(255,255,255,0.07)" : "1px solid #E2E8F0",
                              fontSize: 11,
                              color: isDark ? "#CBD5E1" : "#475569",
                              fontFamily: "'JetBrains Mono', monospace"
                            }}>
                              <span style={{ width: 4, height: 4, borderRadius: "50%", background: c.accent }} />
                              {ch}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Right Side: High-Tech Telemetry HUD */}
                      <div style={{
                        height: 230,
                        background: isDark
                          ? (hoveredCard === i ? `rgba(0,0,0,0.35)` : "rgba(0, 0, 0, 0.4)")
                          : (hoveredCard === i ? "#F0F2F8" : "#F7F8FC"),
                        borderRadius: 16,
                        border: hoveredCard === i ? `1px solid ${c.accent}40` : isDark ? "1px solid rgba(255, 255, 255, 0.08)" : "1px solid #E2E8F0",
                        padding: 16,
                        boxShadow: hoveredCard === i
                          ? `inset 0 2px 10px rgba(0,0,0,0.5), 0 0 20px ${c.accent}10`
                          : "inset 0 2px 10px rgba(0,0,0,0.5)",
                        transition: "all 300ms ease",
                      }}>
                        {detail.renderWidget()}
                      </div>
                    </div>

                    {/* Bottom Status Bar */}
                    <div style={{
                      paddingTop: 12,
                      borderTop: `1px solid ${hoveredCard === i ? c.accent + "25" : "rgba(255, 255, 255, 0.06)"}`,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono', monospace",
                      color: hoveredCard === i ? "rgba(148, 163, 184, 0.9)" : "rgba(148, 163, 184, 0.6)",
                      transition: "all 300ms ease",
                    }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{
                          width: 6, height: 6, borderRadius: "50%", background: c.accent,
                          boxShadow: hoveredCard === i ? `0 0 10px ${c.accent}` : "none",
                          animation: hoveredCard === i ? "pulse-dot 1s ease-in-out infinite" : "none",
                        }} />
                        PRICEIQ AUTONOMOUS PRICING ENGINE v2.4
                      </span>
                      <span style={{ color: hoveredCard === i ? "#4ADE80" : "#34D399", fontWeight: 600, transition: "color 200ms ease" }}>
                        {hoveredCard === i ? "↗ CLICK TO EXPAND DETAILS" : "● CONTINUOUS STREAMING ACTIVE"}
                      </span>
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>

        {/* ── Card Modal Overlay ── */}
        {modalCard !== null && (() => {
          const mc = cards[modalCard];
          const md = [
            {
              eyebrow: "CATALOG INGESTION ENGINE",
              title: "Live Multi-Channel Catalog Sync",
              unit: "ACTIVE SKUs MONITORED",
              chips: ["MongoDB Atlas M10", "Amazon & Flipkart Ingestion", "Real-Time Stock Velocity"],
              renderWidget: () => (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#6366F1", fontWeight: 700, letterSpacing: "0.08em" }}>● REAL-TIME ATLAS SYNC PIPELINE</span>
                    <span style={{ fontSize: 10, color: "#34D399", background: "rgba(52,211,153,0.12)", padding: "2px 8px", borderRadius: 4, fontWeight: 700 }}>99.98% UPTIME</span>
                  </div>
                  {[["Amazon Live Price Feed","184 Products (100%)","#FBBF24"],["Flipkart Marketplace Channel","31 Products (100%)","#38BDF8"],["In-Memory Redis Cache RTT","0.42 ms latency","#34D399"]].map(([label,val,col],idx) => (
                    <div key={idx}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#94A3B8", marginBottom: 5 }}><span>{label}</span><span style={{ color: col as string, fontWeight: 700 }}>{val}</span></div>
                      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 999, overflow: "hidden" }}><div style={{ height: "100%", width: idx === 2 ? "95%" : "100%", background: `linear-gradient(90deg, #6366F1, ${col})`, borderRadius: 999 }} /></div>
                    </div>
                  ))}
                  <div style={{ background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#CBD5E1", marginTop: 8 }}>⚡ <strong>Direct Ingestion:</strong> Zero spreadsheets. Real catalog products continuously update before pricing calculations execute.</div>
                </div>
              ),
            },
            {
              eyebrow: "DEEP LEARNING EMBEDDING MATRIX",
              title: "High-Dimensional Behavioral Latent Space",
              unit: "LEARNED SEMANTIC TOKENS",
              chips: ["128-Dim Embedding Space", "Cosine Intent Matching", "Category Graph Affinity"],
              renderWidget: () => (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#22D3EE", fontWeight: 700 }}>● LATENT CLUSTER EMBEDDINGS (128-D)</span>
                    <span style={{ fontSize: 10, color: "#22D3EE", background: "rgba(34,211,238,0.12)", padding: "2px 8px", borderRadius: 4, fontWeight: 700 }}>50,000 VECTORS</span>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {[["👟 Running Shoes","0.94 cosine","#22D3EE"],["⚡ Speed Boost","0.89 affinity","#818CF8"],["🏷️ Price Elasticity","-1.24","#34D399"],["🌧️ Weather Proof","0.82","#F472B6"],["🔍 High-Intent","0.91","#FBBF24"]].map(([label,val,col],idx) => (
                      <span key={idx} style={{ background: `rgba(${col === "#22D3EE" ? "34,211,238" : col === "#818CF8" ? "129,140,248" : col === "#34D399" ? "52,211,153" : col === "#F472B6" ? "244,114,182" : "251,191,36"},0.12)`, border: `1px solid ${col}40`, color: col as string, padding: "5px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600 }}>{label}: {val}</span>
                    ))}
                  </div>
                  <div style={{ background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.2)", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#CBD5E1", marginTop: 8 }}>🧠 <strong>Dense Representation:</strong> Captures shopper affinity vectors across 50,000 items in real time for precision pricing.</div>
                </div>
              ),
            },
            {
              eyebrow: "RECURRENT INFERENCE BENCHMARK",
              title: "Sub-Millisecond Neural Clickstream Inference",
              unit: "P95 BENCHMARKED LATENCY",
              chips: ["Sub-Frame Render Delivery", "Sequential GRU4Rec Engine", "14× Faster Than Batch"],
              renderWidget: () => (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#A78BFA", fontWeight: 700 }}>● INFERENCE LATENCY OSCILLOSCOPE</span>
                    <span style={{ fontSize: 10, color: "#A78BFA", background: "rgba(167,139,250,0.12)", padding: "2px 8px", borderRadius: 4, fontWeight: 700 }}>14× FASTER</span>
                  </div>
                  {[["Session Clickstream Ingest","2.1 ms"],["GRU4Rec Recurrent Forward Pass","8.4 ms"],["MRP Guardrail Policy Check","3.5 ms"]].map(([label,val],idx) => (
                    <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13, color: "#94A3B8" }}><span>{label}</span><span style={{ color: "#FFFFFF", fontWeight: 700 }}>{val}</span></div>
                  ))}
                  <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 8, display: "flex", justifyContent: "space-between", fontSize: 14, fontWeight: 700 }}><span style={{ color: "#CBD5E1" }}>Total End-to-End</span><span style={{ color: "#A78BFA" }}>14.0 ms (P95)</span></div>
                  <div style={{ background: "rgba(167,139,250,0.08)", border: "1px solid rgba(167,139,250,0.2)", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#CBD5E1" }}>⏱️ <strong>Instant Delivery:</strong> Calculated and delivered inside browser request cycle before viewport paint completes.</div>
                </div>
              ),
            },
            {
              eyebrow: "AUTONOMOUS SAFETY GOVERNANCE",
              title: "Mathematical Anti-Gouging Guardrails",
              unit: "BOUNDED COMPLIANCE GUARANTEE",
              chips: ["Strict 70% MRP Floor", "Hard 100% MRP Ceiling", "Certified Ethical Standard"],
              renderWidget: () => (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#34D399", fontWeight: 700 }}>● BOUNDED PRICING CORRIDOR</span>
                    <span style={{ fontSize: 10, color: "#34D399", background: "rgba(52,211,153,0.12)", padding: "2px 8px", borderRadius: 4, fontWeight: 700 }}>100% COMPLIANT</span>
                  </div>
                  <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: 8, padding: "14px", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#64748B", marginBottom: 8, textTransform: "uppercase" }}><span>Hard Floor (70% MRP)</span><span style={{ color: "#34D399", fontWeight: 700 }}>Active Price</span><span>Hard Ceiling (100% MRP)</span></div>
                    <div style={{ height: 10, background: "rgba(255,255,255,0.08)", borderRadius: 999, position: "relative" }}>
                      <div style={{ position: "absolute", left: "20%", right: "15%", top: 0, bottom: 0, background: "linear-gradient(90deg, #6366F1, #34D399)", borderRadius: 999 }} />
                      <div style={{ position: "absolute", left: "62%", top: -4, width: 18, height: 18, borderRadius: "50%", background: "#34D399", border: "2px solid #FFFFFF", boxShadow: "0 0 12px #34D399" }} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#94A3B8", marginTop: 10 }}><span>₹1,609 (Min)</span><span style={{ color: "#34D399", fontWeight: 800 }}>₹2,099 (+14%)</span><span>₹2,299 (Max)</span></div>
                  </div>
                  <div style={{ background: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.2)", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "#CBD5E1" }}>🛡️ <strong>Zero Exceptions:</strong> Strict mathematical boundary checks lock every price strictly inside non-predatory parameters.</div>
                </div>
              ),
            },
          ];
          const mDetail = md[modalCard] || md[0];
          const circumference = 2 * Math.PI * 22;
          const dashOffset = circumference * (1 - countdown / 30);

          return (
            <>
              {/* Backdrop */}
              <div
                onClick={closeModal}
                style={{
                  position: "fixed",
                  inset: 0,
                  background: "rgba(0, 0, 0, 0.72)",
                  backdropFilter: "blur(8px)",
                  zIndex: 999,
                  animation: "fadeIn 200ms ease",
                }}
              />
              {/* Modal Panel */}
              <div
                style={{
                  position: "fixed",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  zIndex: 1000,
                  width: "min(860px, 94vw)",
                  maxHeight: "90vh",
                  overflowY: "auto",
                  borderRadius: 28,
                  background: "linear-gradient(145deg, rgba(14, 20, 40, 0.98) 0%, rgba(7, 11, 22, 0.99) 100%)",
                  border: `1px solid ${mc.accent}70`,
                  boxShadow: `0 40px 100px -10px rgba(0,0,0,0.9), 0 0 80px ${mc.accent}30, inset 0 1px 0 rgba(255,255,255,0.1)`,
                  backdropFilter: "blur(40px)",
                  animation: "modalScaleIn 280ms cubic-bezier(0.16, 1, 0.3, 1)",
                }}
              >
                {/* Top neon line */}
                <div style={{ position: "absolute", top: 0, left: 40, right: 40, height: 2, background: `linear-gradient(90deg, transparent, ${mc.accent}, transparent)`, boxShadow: `0 0 20px ${mc.accent}`, borderRadius: 999 }} />
                {/* Corner glow */}
                <div style={{ position: "absolute", top: -80, right: -80, width: 300, height: 300, borderRadius: "50%", background: mc.accent, opacity: 0.1, filter: "blur(90px)", pointerEvents: "none" }} />

                {/* Modal Header */}
                <div style={{ padding: "28px 36px 0", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "5px 12px", borderRadius: 9999, background: `${mc.accent}18`, border: `1px solid ${mc.accent}45`, color: mc.accent, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", width: "fit-content" }}>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: mc.accent, boxShadow: `0 0 8px ${mc.accent}` }} />
                      {mDetail.eyebrow}
                    </span>
                    <h3 style={{ fontSize: "1.5rem", fontWeight: 800, color: "#F8FAFC", margin: 0, letterSpacing: "-0.02em", lineHeight: 1.2 }}>{mDetail.title}</h3>
                  </div>

                  {/* Close button with countdown ring */}
                  <div style={{ display: "flex", alignItems: "center", gap: 14, flexShrink: 0, marginLeft: 16 }}>
                    <div style={{ position: "relative", width: 48, height: 48, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <svg width="48" height="48" style={{ position: "absolute", top: 0, left: 0, transform: "rotate(-90deg)" }}>
                        <circle cx="24" cy="24" r="22" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="2.5" />
                        <circle cx="24" cy="24" r="22" fill="none" stroke={mc.accent} strokeWidth="2.5"
                          strokeDasharray={circumference}
                          strokeDashoffset={dashOffset}
                          strokeLinecap="round"
                          style={{ transition: "stroke-dashoffset 900ms linear" }}
                        />
                      </svg>
                      <span style={{ fontSize: 13, fontWeight: 800, color: mc.accent, fontFamily: "'JetBrains Mono', monospace", zIndex: 1 }}>{countdown}</span>
                    </div>
                    <button
                      onClick={closeModal}
                      style={{
                        width: 40, height: 40, borderRadius: "50%",
                        background: "rgba(255,255,255,0.06)",
                        border: "1px solid rgba(255,255,255,0.12)",
                        color: "#94A3B8", cursor: "pointer",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 18, lineHeight: 1,
                        transition: "all 200ms ease",
                      }}
                      onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(239,68,68,0.15)"; (e.currentTarget as HTMLButtonElement).style.color = "#EF4444"; }}
                      onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.06)"; (e.currentTarget as HTMLButtonElement).style.color = "#94A3B8"; }}
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Modal Body */}
                <div style={{ padding: "24px 36px 32px", display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 28, alignItems: "start" }}>
                  {/* Left: Big value + description + chips */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                      <span style={{ fontFamily: "'Plus Jakarta Sans', sans-serif", fontSize: "clamp(3rem, 5vw, 4rem)", fontWeight: 900, color: "#FFFFFF", letterSpacing: "-0.04em", lineHeight: 1 }}>{mc.val}</span>
                      <span style={{ fontSize: 11, fontWeight: 700, color: mc.accent, letterSpacing: "0.08em", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace", background: `${mc.accent}15`, padding: "3px 8px", borderRadius: 4, border: `1px solid ${mc.accent}30` }}>{mDetail.unit}</span>
                    </div>
                    <p style={{ color: "#94A3B8", fontSize: 14, lineHeight: 1.7, margin: 0 }}>{mc.desc}</p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {mDetail.chips.map((ch: string, ci: number) => (
                        <span key={ci} style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "5px 11px", borderRadius: 6, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", fontSize: 12, color: "#CBD5E1", fontFamily: "'JetBrains Mono', monospace" }}>
                          <span style={{ width: 5, height: 5, borderRadius: "50%", background: mc.accent }} />
                          {ch}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Right: Widget */}
                  <div style={{ background: "rgba(0,0,0,0.35)", borderRadius: 16, border: "1px solid rgba(255,255,255,0.07)", padding: 18, boxShadow: "inset 0 2px 12px rgba(0,0,0,0.5)" }}>
                    {mDetail.renderWidget()}
                  </div>
                </div>

                {/* Bottom auto-close bar */}
                <div style={{ margin: "0 36px 28px", borderRadius: 10, overflow: "hidden", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                  <div style={{ height: 3, background: `linear-gradient(90deg, ${mc.accent}, ${mc.accent}50)`, width: `${(countdown / 30) * 100}%`, transition: "width 900ms linear", borderRadius: 999 }} />
                  <div style={{ padding: "8px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "rgba(148,163,184,0.5)" }}>
                    <span>Auto-closing in {countdown}s — click anywhere outside to close</span>
                    <span style={{ color: mc.accent, fontWeight: 700 }}>● PREVIEW ACTIVE</span>
                  </div>
                </div>
              </div>
            </>
          );
        })()}
      </div>
    </div>
  );
}

// Interactive SVG Telemetry Curve
function TelemetryPriceCurve({ isSurge }: { isSurge: boolean }) {
  const pathD = isSurge
    ? "M 10 75 Q 80 80, 140 60 T 260 30 T 360 18 L 360 95 L 10 95 Z"
    : "M 10 75 Q 90 72, 180 70 T 290 68 T 360 65 L 360 95 L 10 95 Z";

  const lineD = isSurge
    ? "M 10 75 Q 80 80, 140 60 T 260 30 T 360 18"
    : "M 10 75 Q 90 72, 180 70 T 290 68 T 360 65";

  return (
    <div className="console-curve-wrap" aria-label="Dynamic Pricing Curve Telemetry">
      <div className="flex items-center justify-between text-[11px] font-mono text-[#A8B1C2] mb-2">
        <span className="flex items-center gap-1.5 text-[#22D3EE]">
          <TrendingUp size={13} />
          <span>REAL-TIME TRAJECTORY (24H)</span>
        </span>
        <span className="text-[#6B7280]">FLOOR ₹1,609 · CEIL ₹2,805</span>
      </div>
      <svg viewBox="0 0 370 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id="piq-curve-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366F1" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#6366F1" stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="cyan-gradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#6366F1" />
            <stop offset="100%" stopColor="#22D3EE" />
          </linearGradient>
        </defs>

        {/* Horizontal grid lines */}
        <line x1="0" y1="20" x2="370" y2="20" className="curve-axis" />
        <line x1="0" y1="50" x2="370" y2="50" className="curve-axis" />
        <line x1="0" y1="80" x2="370" y2="80" className="curve-axis" />

        {/* Modeled reference line */}
        <line x1="0" y1="58" x2="370" y2="58" className="curve-reference-line" />

        {/* Filled curve area */}
        <path d={pathD} className="curve-path-fill" />

        {/* Main curve line */}
        <path d={lineD} className="curve-path-line" style={{ stroke: "url(#cyan-gradient)" }} />

        {/* Interactive target point */}
        <circle cx="360" cy={isSurge ? 18 : 65} r="5" fill="#22D3EE" />
        <circle cx="360" cy={isSurge ? 18 : 65} r="9" fill="none" stroke="#22D3EE" strokeWidth="1.5" opacity="0.6">
          <animate attributeName="r" values="6;12;6" dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.8;0.1;0.8" dur="2s" repeatCount="indefinite" />
        </circle>
      </svg>
      <div className="flex justify-between items-center text-[10px] font-mono text-[#6B7280] mt-1">
        <span>00:00 (Base: ₹2,299)</span>
        <span>12:00 (Signal Ingestion)</span>
        <span className="text-[#22D3EE] font-bold">Now: {isSurge ? "₹2,805 (+22%)" : "₹2,299 (Base)"}</span>
      </div>
    </div>
  );
}

export default function Landing() {
  const { isDark } = useTheme();
  const [previewTab, setPreviewTab] = useState<PreviewTab>("pricing");
  const [inventoryMode, setInventoryMode] = useState<"low" | "steady">("low");
  const [demandMode, setDemandMode] = useState<"elevated" | "normal">("elevated");
  const [activeProblemModal, setActiveProblemModal] = useState<"static" | "smart" | null>(null);
  const [modalTimer, setModalTimer] = useState<number>(30);
  const signals = useLandingSignals();
  useReveal();

  // 30-second auto-close timer when modal is opened
  useEffect(() => {
    if (!activeProblemModal) return;
    setModalTimer(30);

    const interval = setInterval(() => {
      setModalTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setActiveProblemModal(null);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActiveProblemModal(null);
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      clearInterval(interval);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeProblemModal]);

  const isSurgeActive = inventoryMode === "low" && demandMode === "elevated";

  const simulatorCalculation = useMemo(() => {
    const multiplier = isSurgeActive ? 1 + PRICING_RULES.lowStockDemandAdjustment : 1;
    const computedPrice = Math.round(PRICING_RULES.basePrice * multiplier);
    return {
      active: isSurgeActive,
      price: computedPrice,
      diff: computedPrice - PRICING_RULES.basePrice,
      ruleName: isSurgeActive ? "Rule 5: Behavioral Scarcity Surge" : "Rule 1: Standard Equilibrium Base",
      ruleTag: isSurgeActive ? "+22% Applied" : "0% Adjustment",
    };
  }, [isSurgeActive]);

  const gru = signals.benchmark.find((row) => row.key === "gru4rec") ?? signals.benchmark[0];
  const statusCopy = signals.isLoading
    ? "Connecting to pricing telemetry…"
    : signals.source === "live"
      ? `Live Benchmark Stream · ${signals.updatedAt}`
      : `Committed Telemetry Snapshot · ${signals.updatedAt}`;

  return (
    <div className="landing-page" id="top">
      {/* Reimagined Navigation */}
      <LandingNavbar />

      <main id="main-content">
        {/* ── 1. Immersive Command Hero ── */}
        <section className="landing-hero" aria-labelledby="hero-heading">
          <div className="landing-shell landing-hero__grid">
            {/* Left Hero Column */}
            <div className="landing-hero__copy" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Next-Gen Commerce Telemetry</span>
              </div>

              <h1 id="hero-heading">
                Dynamic Pricing with <em>Mathematical Certainty.</em>
              </h1>

              <p className="landing-lede">
                PriceIQ continuously synthesizes interaction telemetry, inventory depletion velocity, and market context
                into auditable, strictly bounded real-time price decisions.
              </p>

              <div className="landing-hero__actions">
                <Link className="landing-button landing-button--primary" to="/shop">
                  <span>Explore Live Platform</span>
                  <ArrowRight size={17} aria-hidden="true" />
                </Link>
                <a className="landing-button landing-button--outline" href="#interactive-simulator">
                  <span>Simulate Algorithm</span>
                  <Sliders size={16} className="text-[#22D3EE]" aria-hidden="true" />
                </a>
              </div>

              <div className="landing-trust">
                <span className="trust-item">
                  <Check size={15} />
                  <span>Sub-millisecond latency</span>
                </span>
                <span className="divider" />
                <span className="trust-item">
                  <Check size={15} />
                  <span>MRP Floor/Ceiling Guardrails</span>
                </span>
                <span className="divider" />
                <span className="trust-item">
                  <Check size={15} />
                  <span>Deterministic Audit Trail</span>
                </span>
              </div>
            </div>

            {/* Right Hero Column: Real-time Telemetry Console */}
            <div className="landing-hero__visual" data-reveal>
              <div className="telemetry-console">
                <div className="telemetry-console__bar">
                  <div className="telemetry-console__endpoint">
                    <span className="method">GET</span>
                    <span>/api/price?id=trailpack-28l</span>
                  </div>
                  <div className="telemetry-console__live-badge">
                    <Radio size={11} className="animate-pulse" />
                    <span>200 OK · 0.42ms</span>
                  </div>
                </div>

                <div className="telemetry-console__body">
                  <div className="console-product-header">
                    <div className="console-product-info">
                      <div className="console-product-avatar">
                        <Zap size={22} />
                      </div>
                      <div className="console-product-meta">
                        <h4>{WORKED_PRICE.product}</h4>
                        <span>SKU: TP-28L-IND · In Stock: {WORKED_PRICE.inventory} Units</span>
                      </div>
                    </div>
                    <div className="console-price-badge">
                      <span className="label">RECOMMENDED PRICE</span>
                      <span className="value">{formatIndianCurrency(WORKED_PRICE.recommendedPrice)}</span>
                    </div>
                  </div>

                  {/* Dynamic SVG Price Trajectory Curve */}
                  <TelemetryPriceCurve isSurge={true} />

                  {/* Multi-Signal Metrics Grid */}
                  <div className="console-signals-grid">
                    <div className="console-signal-card highlight">
                      <span className="title">INVENTORY</span>
                      <span className="stat">{WORKED_PRICE.inventory} Left (Critical)</span>
                    </div>
                    <div className="console-signal-card highlight">
                      <span className="title">15M VIEWS</span>
                      <span className="stat">{WORKED_PRICE.recentViews} Interactions</span>
                    </div>
                    <div className="console-signal-card">
                      <span className="title">COMPETITOR REF</span>
                      <span className="stat">{formatIndianCurrency(WORKED_PRICE.modeledReference)}</span>
                    </div>
                  </div>

                  {/* Decision Explanation */}
                  <div className="console-reasoning">
                    <strong>Rule 5 Selected: </strong>
                    Inventory scarcity (≤3 units) combined with elevated interaction velocity triggers +22% dynamic surge,
                    bounded strictly within 100% of MRP.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── 2. Scroll-Driven Sticky Card Reveal ── */}
        <ScrollRevealCards cards={[
          {
            icon: <Activity size={30} />,
            tag: "Indexed Catalog",
            val: signals.catalogItems.toLocaleString("en-IN"),
            desc: "Real catalog products continuously synced with MongoDB — every pricing decision backed by live inventory data.",
            accent: "#6366F1",
          },
          {
            icon: <Sparkles size={30} />,
            tag: "Model Vocabulary",
            val: signals.modelVocabulary.toLocaleString("en-IN"),
            desc: "High-dimensional feature embedding space powering our GRU4Rec session-depth recommendation engine.",
            accent: "#22D3EE",
          },
          {
            icon: <Zap size={30} />,
            tag: "GRU4Rec Latency",
            val: formatMilliseconds(gru.averageLatencyMs),
            desc: "P95 benchmarked response time for our ML recommendation pipeline — sub-millisecond, measured under real load.",
            accent: "#A78BFA",
          },
          {
            icon: <ShieldCheck size={30} />,
            tag: "Price Governance",
            val: "100%",
            desc: "Every single price output is guaranteed to be bounded strictly between 70% and 100% of MRP. Zero exceptions.",
            accent: "#34D399",
          },
        ]} />

        {/* ── 3. The Pricing Problem (Asymmetric Bento) ── */}
        <section className="landing-section" aria-labelledby="problem-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>The Margin Challenge</span>
              </div>
              <h2 id="problem-heading">
                Static pricing bleeds revenue in <em>high-velocity commerce.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Traditional catalogs update prices once a week or through disconnected manual spreadsheets. When traffic
                spikes or inventory dwindles, static pricing fails both retailers and shoppers.
              </p>
            </div>

            <div className="problem-bento" data-reveal>
              {/* Card 1: Legacy Static Approach */}
              <div
                className="problem-card problem-card--static"
                onClick={() => setActiveProblemModal("static")}
                style={{ cursor: "pointer" }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                    <div className="icon-chip">
                      <Activity size={22} />
                    </div>
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 10,
                      fontWeight: 700,
                      padding: "4px 10px",
                      borderRadius: 9999,
                      background: "rgba(251, 113, 133, 0.15)",
                      color: "#FB7185",
                      border: "1px solid rgba(251, 113, 133, 0.3)",
                      letterSpacing: "0.05em",
                      display: "flex",
                      alignItems: "center",
                      gap: 4
                    }}>
                      CLICK TO OPEN FULL VIEW
                      <ArrowUpRight size={12} />
                    </span>
                  </div>

                  <h3>The Legacy Approach: Disconnected & Blind</h3>
                  <p>
                    Pricing logic is siloed from real-time customer intent. High demand sells out remaining inventory at
                    deep discounts, leaving money on the table while stock-outs surprise shoppers.
                  </p>

                  <ul>
                    <li>
                      <span className="text-[#FB7185] font-bold">✕</span> Inability to react to sudden category interaction surges
                    </li>
                    <li>
                      <span className="text-[#FB7185] font-bold">✕</span> Blind discounting when remaining inventory is under 3 units
                    </li>
                    <li>
                      <span className="text-[#FB7185] font-bold">✕</span> Black-box pricing changes with zero auditability or rationale
                    </li>
                  </ul>
                </div>

                <div style={{
                  marginTop: 24,
                  paddingTop: 16,
                  borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: 11,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "#FB7185"
                }}>
                  <span>● VIEW REVENUE LOSS AUDIT</span>
                  <span style={{ color: "#94A3B8" }}>Opens in 30s auto-close window →</span>
                </div>
              </div>

              {/* Card 2: PriceIQ Synchronized Telemetry */}
              <div
                className="problem-card problem-card--smart"
                onClick={() => setActiveProblemModal("smart")}
                style={{ cursor: "pointer" }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                    <div className="icon-chip">
                      <Sparkles size={22} />
                    </div>
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 10,
                      fontWeight: 700,
                      padding: "4px 10px",
                      borderRadius: 9999,
                      background: "rgba(34, 211, 238, 0.15)",
                      color: "#22D3EE",
                      border: "1px solid rgba(34, 211, 238, 0.3)",
                      letterSpacing: "0.05em",
                      display: "flex",
                      alignItems: "center",
                      gap: 4
                    }}>
                      CLICK TO OPEN FULL VIEW
                      <ArrowUpRight size={12} />
                    </span>
                  </div>

                  <h3>The PriceIQ Engine: Synchronized Telemetry</h3>
                  <p>
                    Every click, cart addition, and stock depletion feeds directly into a deterministic pricing pipeline with
                    mathematical floor and ceiling safeguards.
                  </p>

                  <ul>
                    <li>
                      <span className="text-[#34D399] font-bold">✓</span> Instant micro-adjustments grounded in inventory scarcity
                    </li>
                    <li>
                      <span className="text-[#34D399] font-bold">✓</span> Transparent rule reasoning exposed for every calculated price
                    </li>
                    <li>
                      <span className="text-[#34D399] font-bold">✓</span> Guaranteed compliance: Prices never exceed MRP or breach 70% floor
                    </li>
                  </ul>
                </div>

                <div style={{
                  marginTop: 24,
                  paddingTop: 16,
                  borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: 11,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "#22D3EE"
                }}>
                  <span>● VIEW LIVE MARGIN TELEMETRY</span>
                  <span style={{ color: "#94A3B8" }}>Opens in 30s auto-close window →</span>
                </div>
              </div>
            </div>

            {/* ── Floating Full-Size Modal (Opens Over Cards With 30s Auto-Close Timer) ── */}
            {activeProblemModal && (
              <div
                style={{
                  position: "fixed",
                  inset: 0,
                  zIndex: 99999,
                  background: "rgba(3, 6, 18, 0.85)",
                  backdropFilter: "blur(16px)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "24px",
                  animation: "fadeIn 200ms ease-out",
                }}
                onClick={() => setActiveProblemModal(null)}
              >
                <div
                  style={{
                    width: "min(860px, 95vw)",
                    maxHeight: "90vh",
                    overflowY: "auto",
                    borderRadius: 24,
                    background: isDark
                      ? (activeProblemModal === "static"
                          ? "linear-gradient(155deg, rgba(34, 16, 26, 0.98) 0%, rgba(14, 8, 16, 0.99) 100%)"
                          : "linear-gradient(155deg, rgba(14, 26, 48, 0.98) 0%, rgba(8, 14, 28, 0.99) 100%)")
                      : "#FFFFFF",



                      border: `1.5px solid ${activeProblemModal === "static" ? "#FB7185" : "#22D3EE"}`,
                    boxShadow: `0 30px 90px rgba(0,0,0,0.9), 0 0 60px ${activeProblemModal === "static" ? "rgba(251,113,133,0.3)" : "rgba(34,211,238,0.3)"}`,
                    position: "relative",
                    overflow: "hidden",
                    padding: "36px 40px",
                  }}
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Top 30-Second Countdown Progress Bar */}
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: "rgba(255,255,255,0.1)" }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${(modalTimer / 30) * 100}%`,
                        background: activeProblemModal === "static"
                          ? "linear-gradient(90deg, #FB7185, #F43F5E)"
                          : "linear-gradient(90deg, #22D3EE, #34D399)",
                        transition: "width 1s linear",
                        boxShadow: `0 0 10px ${activeProblemModal === "static" ? "#FB7185" : "#22D3EE"}`,
                      }}
                    />
                  </div>

                  {/* Header: Title + 30s Countdown Timer Badge + Close Button */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "5px 12px",
                        borderRadius: 9999,
                        background: activeProblemModal === "static" ? "rgba(251,113,133,0.15)" : "rgba(34,211,238,0.15)",
                        border: `1px solid ${activeProblemModal === "static" ? "#FB7185" : "#22D3EE"}`,
                        color: activeProblemModal === "static" ? "#FB7185" : "#22D3EE",
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: 11,
                        fontWeight: 700,
                        textTransform: "uppercase",
                      }}>
                        <span style={{
                          width: 6,
                          height: 6,
                          borderRadius: "50%",
                          background: activeProblemModal === "static" ? "#FB7185" : "#22D3EE",
                          boxShadow: `0 0 8px ${activeProblemModal === "static" ? "#FB7185" : "#22D3EE"}`
                        }} />
                        {activeProblemModal === "static" ? "CRITICAL VALUE LOSS AUDIT" : "AUTONOMOUS MARGIN PRESERVATION"}
                      </span>
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      {/* 30-Second Auto-Close Countdown Display */}
                      <span style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        fontFamily: "'JetBrains Mono', monospace",
                        fontSize: 11,
                        color: "#94A3B8",
                        background: "rgba(255,255,255,0.04)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        padding: "4px 10px",
                        borderRadius: 6
                      }}>
                        <Clock size={12} style={{ color: activeProblemModal === "static" ? "#FB7185" : "#22D3EE" }} />
                        <span>Auto-closes in <strong style={{ color: "#FFFFFF" }}>{modalTimer}s</strong></span>
                      </span>

                      {/* Explicit Close Button */}
                      <button
                        type="button"
                        onClick={() => setActiveProblemModal(null)}
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: 8,
                          background: "rgba(255,255,255,0.08)",
                          border: "1px solid rgba(255,255,255,0.12)",
                          color: "#FFFFFF",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          cursor: "pointer",
                          transition: "all 150ms ease",
                        }}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  </div>

                  {/* Body Content based on Modal Type */}
                  {activeProblemModal === "static" ? (
                    <div>
                      <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: 13, color: "#FB7185", fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", marginBottom: 6 }}>
                          FAILURE SCENARIO · RUNAWAY MARGIN BLEED
                        </div>
                        <h3 style={{ fontSize: 26, fontWeight: 800, color: "#FFFFFF", margin: 0 }}>
                          The Cost of Disconnected &amp; Blind Pricing
                        </h3>
                        <p style={{ color: "#A8B1C2", fontSize: 15, lineHeight: 1.6, marginTop: 8 }}>
                          When shopper traffic accelerates and catalog inventory reaches low thresholds (≤ 3 units),
                          traditional batch catalogs leave prices discounted at wholesale minimums. Retailers forfeit millions in uncaptured margin while frustrating buyers with sudden stockouts.
                        </p>
                      </div>

                      {/* 4-Grid Diagnostic Matrix */}
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14, margin: "24px 0" }}>
                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(251,113,133,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Uncaptured Unit Margin</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#FB7185", marginTop: 4 }}>-₹700 / unit</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Sold 32% below equilibrium</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(251,113,133,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Stock-Out Abandonment</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#FB7185", marginTop: 4 }}>64% Drop-Off</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>High-intent buyers leave cart</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(251,113,133,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Catalog Ingestion Lag</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#FFFFFF", marginTop: 4 }}>7 Days Latency</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Manual CSV exports &amp; sheets</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(251,113,133,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Regulatory Safety</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#FBBF24", marginTop: 4 }}>Zero Rails</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Unchecked human data errors</span>
                        </div>
                      </div>

                      <div style={{ background: "rgba(251,113,133,0.08)", border: "1px solid rgba(251,113,133,0.25)", borderRadius: 12, padding: "14px 18px", fontSize: 13, color: "#CBD5E1" }}>
                        ⚠️ <strong>The Bottom Line:</strong> A catalog with 5,000 monthly orders loses approximately ₹35,00,000 in gross margin annually through uncoordinated manual discounting during traffic spikes.
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: 13, color: "#22D3EE", fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase", marginBottom: 6 }}>
                          OPTIMAL HARVESTING · DETERMINISTIC RECONCILIATION
                        </div>
                        <h3 style={{ fontSize: 26, fontWeight: 800, color: "#FFFFFF", margin: 0 }}>
                          The PriceIQ Engine: Synchronized Telemetry
                        </h3>
                        <p style={{ color: "#A8B1C2", fontSize: 15, lineHeight: 1.6, marginTop: 8 }}>
                          Every shopper click, category interaction surge, and inventory depletion feeds directly into our real-time
                          reinforcement learning pipeline. Prices are re-evaluated within milliseconds and clamped strictly within MRP boundaries.
                        </p>
                      </div>

                      {/* 4-Grid Diagnostic Matrix */}
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14, margin: "24px 0" }}>
                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Preserved Revenue Lift</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#34D399", marginTop: 4 }}>+21.6% Net</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>+₹606 per surge checkout</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Event Reaction Speed</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#22D3EE", marginTop: 4 }}>0.42 ms Live</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Calculated before DOM paint</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Ethical Guardrails</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#34D399", marginTop: 4 }}>100% Bounded</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Strict 70% floor &amp; 100% MRP</span>
                        </div>

                        <div style={{ background: "rgba(0,0,0,0.4)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 12, padding: "16px" }}>
                          <span style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>Thompson Multi-Arm</span>
                          <div style={{ fontSize: 24, fontWeight: 900, color: "#A78BFA", marginTop: 4 }}>Bandit A/B</div>
                          <span style={{ fontSize: 11, color: "#94A3B8" }}>Autonomous conversion optimization</span>
                        </div>
                      </div>

                      <div style={{ background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 12, padding: "14px 18px", fontSize: 13, color: "#CBD5E1" }}>
                        🛡️ <strong>Certified Governance:</strong> Fully auditable decision trails explain every price change to prevent predatory surges while ensuring merchant profitability.
                      </div>
                    </div>
                  )}

                  {/* Bottom Modal Actions */}
                  <div style={{ marginTop: 24, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.08)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, color: "#64748B", fontFamily: "'JetBrains Mono', monospace" }}>
                      Press ESC or click anywhere outside to dismiss
                    </span>
                    <button
                      type="button"
                      onClick={() => setActiveProblemModal(null)}
                      style={{
                        padding: "8px 20px",
                        borderRadius: 8,
                        background: activeProblemModal === "static" ? "#FB7185" : "#22D3EE",
                        color: "#080C1A",
                        fontSize: 12,
                        fontWeight: 800,
                        border: "none",
                        cursor: "pointer",
                        letterSpacing: "0.02em",
                      }}
                    >
                      Dismiss Window ({modalTimer}s)
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ── 4. PriceIQ Decision Pipeline (How It Works) ── */}
        <section className="landing-section" id="how-it-works" aria-labelledby="architecture-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Decision Pipeline</span>
              </div>
              <h2 id="architecture-heading">
                How PriceIQ computes <em>prices in milliseconds.</em>
              </h2>
              <p className="landing-section-heading__copy">
                A clean, explainable pipeline separated into distinct ingestion, evaluation, and delivery stages.
              </p>
            </div>

            {/* ── Continuous Infinite Loop Pipeline Track ── */}
            <div className="pipeline-marquee-container" data-reveal>
              <div className="pipeline-marquee-track">
                {[
                  {
                    step: "STAGE 01",
                    eyebrow: "INGESTION PHASE",
                    title: "Interaction Capture",
                    desc: "Shopper pageviews, cart adds, and wishlist events stream to the public tracking endpoint with sub-ms recording.",
                    accent: "#22D3EE",
                    accentGlow: "rgba(34, 211, 238, 0.25)",
                    icon: <Radio size={18} />,
                    stat: "0.42 ms",
                    statLabel: "Event Ingest Latency",
                    chips: ["Public SSE Stream", "Zero-Wait Dispatch"],
                  },
                  {
                    step: "STAGE 02",
                    eyebrow: "EVALUATION PHASE",
                    title: "Telemetry Aggregation",
                    desc: "Calculates 15-minute interaction velocity, remaining stock counts, and modeled marketplace references.",
                    accent: "#6366F1",
                    accentGlow: "rgba(99, 102, 241, 0.25)",
                    icon: <Layers size={18} />,
                    stat: "15-Min",
                    statLabel: "Rolling Window Velocity",
                    chips: ["Amazon & Flipkart Sync", "Stock Velocity Index"],
                  },
                  {
                    step: "STAGE 03",
                    eyebrow: "DECISION MATRIX",
                    title: "Deterministic Rule Engine",
                    desc: "Applies auditable conditions (scarcity thresholds, velocity spikes, behavioral segments) to determine multiplier.",
                    accent: "#A78BFA",
                    accentGlow: "rgba(167, 139, 250, 0.25)",
                    icon: <Cpu size={18} />,
                    stat: "+22%",
                    statLabel: "Scarcity Surge Peak",
                    chips: ["Bandit Multi-Arm", "Transparent Reasoning"],
                  },
                  {
                    step: "STAGE 04",
                    eyebrow: "DELIVERY & SAFETY",
                    title: "Guardrail Clamping",
                    desc: "Enforces strict regulatory floor (70% MRP) and ceiling (100% MRP) before delivering live to the storefront.",
                    accent: "#34D399",
                    accentGlow: "rgba(52, 211, 153, 0.25)",
                    icon: <ShieldCheck size={18} />,
                    stat: "100%",
                    statLabel: "Guaranteed MRP Cap",
                    chips: ["Anti-Gouging Lock", "Audit Traceability"],
                  },
                  // Duplicate set for seamless continuous 100% infinite loop
                  {
                    step: "STAGE 01",
                    eyebrow: "INGESTION PHASE",
                    title: "Interaction Capture",
                    desc: "Shopper pageviews, cart adds, and wishlist events stream to the public tracking endpoint with sub-ms recording.",
                    accent: "#22D3EE",
                    accentGlow: "rgba(34, 211, 238, 0.25)",
                    icon: <Radio size={18} />,
                    stat: "0.42 ms",
                    statLabel: "Event Ingest Latency",
                    chips: ["Public SSE Stream", "Zero-Wait Dispatch"],
                  },
                  {
                    step: "STAGE 02",
                    eyebrow: "EVALUATION PHASE",
                    title: "Telemetry Aggregation",
                    desc: "Calculates 15-minute interaction velocity, remaining stock counts, and modeled marketplace references.",
                    accent: "#6366F1",
                    accentGlow: "rgba(99, 102, 241, 0.25)",
                    icon: <Layers size={18} />,
                    stat: "15-Min",
                    statLabel: "Rolling Window Velocity",
                    chips: ["Amazon & Flipkart Sync", "Stock Velocity Index"],
                  },
                  {
                    step: "STAGE 03",
                    eyebrow: "DECISION MATRIX",
                    title: "Deterministic Rule Engine",
                    desc: "Applies auditable conditions (scarcity thresholds, velocity spikes, behavioral segments) to determine multiplier.",
                    accent: "#A78BFA",
                    accentGlow: "rgba(167, 139, 250, 0.25)",
                    icon: <Cpu size={18} />,
                    stat: "+22%",
                    statLabel: "Scarcity Surge Peak",
                    chips: ["Bandit Multi-Arm", "Transparent Reasoning"],
                  },
                  {
                    step: "STAGE 04",
                    eyebrow: "DELIVERY & SAFETY",
                    title: "Guardrail Clamping",
                    desc: "Enforces strict regulatory floor (70% MRP) and ceiling (100% MRP) before delivering live to the storefront.",
                    accent: "#34D399",
                    accentGlow: "rgba(52, 211, 153, 0.25)",
                    icon: <ShieldCheck size={18} />,
                    stat: "100%",
                    statLabel: "Guaranteed MRP Cap",
                    chips: ["Anti-Gouging Lock", "Audit Traceability"],
                  },
                ].map((stage, idx) => (
                  <div
                    key={idx}
                    className="pipeline-card-premium"
                    style={{
                      "--card-accent": stage.accent,
                      "--card-accent-glow": stage.accentGlow,
                    } as React.CSSProperties}
                  >
                    {/* Top Accent Neon Line */}
                    <div
                      className="pipeline-card-glow-line"
                      style={{
                        background: `linear-gradient(90deg, transparent, ${stage.accent}, transparent)`,
                        boxShadow: `0 0 10px ${stage.accent}`,
                      }}
                    />

                    {/* Header: Stage Badge + Icon Box */}
                    <div className="pipeline-card-header">
                      <div
                        className="pipeline-card-badge"
                        style={{
                          color: stage.accent,
                          background: `${stage.accent}16`,
                          border: `1px solid ${stage.accent}45`,
                        }}
                      >
                        <span
                          className="pipeline-badge-dot"
                          style={{
                            background: stage.accent,
                            boxShadow: `0 0 8px ${stage.accent}`,
                          }}
                        />
                        {stage.step}
                      </div>

                      <div
                        className="pipeline-card-icon-box"
                        style={{
                          color: stage.accent,
                          background: `${stage.accent}14`,
                          border: `1px solid ${stage.accent}30`,
                        }}
                      >
                        {stage.icon}
                      </div>
                    </div>

                    {/* Title & Phase */}
                    <div className="pipeline-card-title-wrap">
                      <span className="pipeline-card-eyebrow">{stage.eyebrow}</span>
                      <h4 className="pipeline-card-title">{stage.title}</h4>
                    </div>

                    {/* Description */}
                    <p className="pipeline-card-desc">{stage.desc}</p>

                    {/* Footer: Live Telemetry Stat & Chips */}
                    <div className="pipeline-card-footer">
                      <div className="pipeline-card-stat">
                        <span className="pipeline-stat-val" style={{ color: stage.accent }}>
                          {stage.stat}
                        </span>
                        <span className="pipeline-stat-lbl">{stage.statLabel}</span>
                      </div>
                      <div className="pipeline-card-chips">
                        {stage.chips.map((ch, chIdx) => (
                          <span key={chIdx} className="pipeline-chip">
                            {ch}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── 5. Interactive Pricing Engine Simulator ── */}
        <section className="landing-section" id="interactive-simulator" aria-labelledby="simulator-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Interactive Algorithm Simulator</span>
              </div>
              <h2 id="simulator-heading">
                Test the engine with <em>live scenario inputs.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Toggle stock levels and shopper interaction velocity to observe how the pricing engine recalculates prices
                in real time.
              </p>
            </div>

            <div className="simulator-card" data-reveal>
              <div className="simulator-controls">
                <div className="simulator-group">
                  <label>Remaining Stock Level</label>
                  <div className="simulator-btn-row">
                    <button
                      type="button"
                      className={`simulator-btn ${inventoryMode === "low" ? "is-active" : ""}`}
                      onClick={() => setInventoryMode("low")}
                    >
                      <span>Low Stock (≤ 3 units)</span>
                    </button>
                    <button
                      type="button"
                      className={`simulator-btn ${inventoryMode === "steady" ? "is-active" : ""}`}
                      onClick={() => setInventoryMode("steady")}
                    >
                      <span>Steady Stock (&gt; 15 units)</span>
                    </button>
                  </div>
                </div>

                <div className="simulator-group">
                  <label>15-Minute Shopper Demand</label>
                  <div className="simulator-btn-row">
                    <button
                      type="button"
                      className={`simulator-btn ${demandMode === "elevated" ? "is-active" : ""}`}
                      onClick={() => setDemandMode("elevated")}
                    >
                      <span>Elevated (≥ 20 views)</span>
                    </button>
                    <button
                      type="button"
                      className={`simulator-btn ${demandMode === "normal" ? "is-active" : ""}`}
                      onClick={() => setDemandMode("normal")}
                    >
                      <span>Normal Traffic (&lt; 20 views)</span>
                    </button>
                  </div>
                </div>

                <div className={`p-3.5 rounded-lg border text-xs leading-relaxed ${
                  isDark 
                    ? "bg-[#0D1220] border-[#252D40] text-[#A8B1C2]" 
                    : "bg-[#F0F2F8] border-[#E2E8F0] text-[#334155]"
                }`}>
                  <strong className={`block mb-1 font-semibold ${isDark ? "text-[#F8FAFC]" : "text-[#111827]"}`}>
                    Mathematical Formula:
                  </strong>
                  <code className={isDark ? "text-[#CBD5E1]" : "text-[#1E293B]"}>
                    P_final = clamp(P_base × (1 + R_surge), 0.70 × MRP, 1.00 × MRP)
                  </code>
                </div>
              </div>

              <div className="simulator-output">
                <div className="simulator-output-top">
                  <span className={`text-[11px] font-mono font-semibold ${isDark ? "text-[#A8B1C2]" : "text-[#475569]"}`}>
                    COMPUTED ENGINE OUTPUT
                  </span>
                  <span className={`simulator-rule-pill ${simulatorCalculation.active ? "surge" : "standard"}`}>
                    <Zap size={11} />
                    {simulatorCalculation.ruleTag}
                  </span>
                </div>

                <div className="simulator-price-display">
                  <div className="price-amount">{formatIndianCurrency(simulatorCalculation.price)}</div>
                  <div className="price-diff">
                    {simulatorCalculation.active ? (
                      <>
                        <TrendingUp size={14} />
                        <span>+{formatIndianCurrency(simulatorCalculation.diff)} Scarcity Surge Adjusted</span>
                      </>
                    ) : (
                      <span>Standard Base Price Equilibrium</span>
                    )}
                  </div>
                </div>

                <div className="simulator-breakdown">
                  <div className="row">
                    <span>Applied Rule:</span>
                    <span className={`font-semibold ${isDark ? "text-[#22D3EE]" : "text-[#0891B2]"}`}>
                      {simulatorCalculation.ruleName}
                    </span>
                  </div>
                  <div className="row">
                    <span>Base Reference:</span>
                    <span>{formatIndianCurrency(PRICING_RULES.basePrice)}</span>
                  </div>
                  <div className="row">
                    <span>MRP Maximum Bound:</span>
                    <span>₹2,805 (Enforced 100% Ceiling)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── 6. Core Platform Capabilities (Bento Matrix) ── */}
        <section className="landing-section" id="product" aria-labelledby="capabilities-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Architecture & Features</span>
              </div>
              <h2 id="capabilities-heading">
                Built for high precision, <em>zero guesswork.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Four foundational subsystems working synchronously across database, memory, and ML layers.
              </p>
            </div>

            <div className="capabilities-bento" data-reveal>
              {/* Card 1: Multi-Signal Engine */}
              <div className="bento-cell bento-cell--large">
                <div className="bento-icon-wrap">
                  <Zap size={22} />
                </div>
                <h3>Multi-Signal Real-Time Pricing</h3>
                <p>
                  Combines inventory levels, restock horizons, interaction velocity, and competitor price baselines into a
                  continuous recalculation loop without human bottleneck.
                </p>
                <div className="bento-widget">
                  <div className="p-3 rounded-lg bg-[#0D1220] border border-[#252D40] flex items-center justify-between text-xs font-mono">
                    <span className="text-[#A8B1C2]">Rule Trigger: stock ≤ 3 & views ≥ 20</span>
                    <span className="text-[#22D3EE] font-bold">+22% Adjusted</span>
                  </div>
                </div>
              </div>

              {/* Card 2: 3-Tier Recommendation Ladder */}
              <div className="bento-cell bento-cell--small">
                <div className="bento-icon-wrap">
                  <Sparkles size={22} />
                </div>
                <h3>Session-Depth Ladder</h3>
                <p>
                  Automatically selects the best recommendation model depending on shopper history depth.
                </p>
                <div className="bento-widget">
                  <div className="ladder-visual">
                    <div className="ladder-item">
                      <span className="depth">0 Events</span>
                      <span className="model">Popularity</span>
                    </div>
                    <div className="ladder-item">
                      <span className="depth">1-2 Events</span>
                      <span className="model">TF-IDF</span>
                    </div>
                    <div className="ladder-item" style={{ borderColor: "rgba(99, 102, 241, 0.4)" }}>
                      <span className="depth">3+ Events</span>
                      <span className="model text-[#A78BFA]">GRU4Rec</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3: Deterministic Experiment Routing */}
              <div className="bento-cell bento-cell--equal">
                <div className="bento-icon-wrap">
                  <GitBranch size={22} />
                </div>
                <h3>Deterministic A/B Routing</h3>
                <p>
                  Shoppers are assigned consistently to control or treatment cohorts using cryptographic hash seeds,
                  guaranteeing zero variant drift across sessions.
                </p>
                <div className="bento-widget">
                  <div className="split-pill-visual">
                    <span className="text-[#A8B1C2]">Cohort: Control (50%)</span>
                    <span className="text-[#6B7280]">⇄</span>
                    <span className="text-[#34D399]">Cohort: Dynamic Treatment (50%)</span>
                  </div>
                </div>
              </div>

              {/* Card 4: Redis Stream Ingestion */}
              <div className="bento-cell bento-cell--equal">
                <div className="bento-icon-wrap">
                  <Network size={22} />
                </div>
                <h3>Asynchronous Stream Buffering</h3>
                <p>
                  Interactions are ingested via high-throughput memory streams with fast database fallbacks, isolating UI
                  latency from ML training loops.
                </p>
                <div className="bento-widget">
                  <div className="stream-flow-visual">
                    <span className="text-[#22D3EE]">Client Click</span>
                    <span>→</span>
                    <span className="text-[#A78BFA]">Stream Ingest</span>
                    <span>→</span>
                    <span className="text-[#34D399]">Feature Store</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── 7. ML Benchmarks & Performance ── */}
        <section className="landing-section" id="benchmarks" aria-labelledby="benchmarks-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Empirical Evaluation</span>
              </div>
              <h2 id="benchmarks-heading">
                Measured recommender strategies, <em>sourced transparently.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Side-by-side held-out test evaluation on the {SNAPSHOT.testSamples} sample test set.
              </p>
            </div>

            <div className="benchmark-container" data-reveal>
              <div className="benchmark-header-row">
                <span className="text-xs font-mono text-[#A8B1C2]">MODEL EVALUATION REPORT</span>
                <span className="text-xs font-mono text-[#22D3EE] flex items-center gap-2">
                  <span className="status-ping" />
                  {statusCopy}
                </span>
              </div>

              <div className="benchmark-table-wrapper">
                <table className="piq-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Hit@10</th>
                      <th>NDCG@10</th>
                      <th>Catalog Coverage</th>
                      <th>Average Latency</th>
                      <th>P95 Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {signals.benchmark.map((row: BenchmarkRow) => (
                      <tr key={row.key} className={row.key === "gru4rec" ? "is-highlight" : ""}>
                        <td>
                          <span className={`strategy-badge ${row.key === "gru4rec" ? "gru" : row.key === "contentTfidf" ? "tfidf" : ""}`}>
                            <span className="dot" />
                            {row.label}
                          </span>
                        </td>
                        <td className="primary-stat">{formatPercent(row.hitAt10)}</td>
                        <td>{formatPercent(row.ndcgAt10)}</td>
                        <td>{formatPercent(row.coverage)}</td>
                        <td>{formatMilliseconds(row.averageLatencyMs)}</td>
                        <td>{formatMilliseconds(row.p95LatencyMs)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* ── 8. Interactive Preview Console ── */}
        <section className="landing-section" aria-labelledby="preview-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Deep Inspection</span>
              </div>
              <h2 id="preview-heading">
                Interfaces that reveal the <em>logic behind every number.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Explore real endpoint structures and returned payloads for pricing, recommendations, and system health.
              </p>
            </div>

            <div className="telemetry-console" data-reveal>
              <div className="flex border-b border-[#252D40] bg-[#0D1220]">
                {(Object.keys(PREVIEWS) as PreviewTab[]).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    className={`px-6 py-3.5 text-xs font-mono font-semibold transition-colors border-r border-[#252D40] ${
                      previewTab === tab ? "bg-[#121827] text-[#22D3EE] border-b-2 border-b-[#6366F1]" : "text-[#A8B1C2] hover:text-[#F8FAFC]"
                    }`}
                    onClick={() => setPreviewTab(tab)}
                  >
                    {tab.toUpperCase()} TELEMETRY
                  </button>
                ))}
              </div>

              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <span className="text-[10px] font-mono text-[#6B7280] block uppercase">{PREVIEWS[previewTab].eyebrow}</span>
                    <h3 className="font-display text-lg font-bold text-[#F8FAFC]">{PREVIEWS[previewTab].title}</h3>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-[#171E30] border border-[#252D40] text-[10px] font-mono text-[#22D3EE]">
                    LIVE STRUCTURE
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                  {PREVIEWS[previewTab].rows.map(([label, value]) => (
                    <div key={label} className="p-3 rounded-lg bg-[#0D1220] border border-[#252D40] flex justify-between items-center text-xs font-mono">
                      <span className="text-[#A8B1C2]">{label}</span>
                      <span className="text-[#F8FAFC] font-semibold">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── 9. Live Product Showcase ── */}
        <section className="landing-section" aria-labelledby="showcase-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Catalog Intelligence</span>
              </div>
              <h2 id="showcase-heading">
                Dynamic pricing active on <em>real products.</em>
              </h2>
              <p className="landing-section-heading__copy">
                Explore real catalog items with their live calculated prices and algorithm reasoning.
              </p>
            </div>

            <div className="products-showcase-grid" data-reveal>
              {/* Product 1 — 3D Flip Card */}
              <div className="flip-card">
                <div className="flip-card-inner">
                  {/* Front */}
                  <div className="flip-card-front showcase-product-card">
                    <div className="showcase-top-meta">
                      <span className="showcase-stock-badge">2 Left in Stock</span>
                      <span className="text-[11px] font-mono text-[#A78BFA]">Rule 5 Surge</span>
                    </div>
                    <h4>TrailPack 28L Pro Outdoor Backpack</h4>
                    <div className="showcase-pricing-row">
                      <span className="live-price">₹2,805</span>
                      <span className="mrp-price">MRP ₹2,805</span>
                      <span className="text-xs font-mono text-[#22D3EE]">+22% Scarcity</span>
                    </div>
                    <div className="showcase-rule-tag">
                      <Zap size={14} />
                      <span>High demand (24 views) + Low stock trigger</span>
                    </div>
                    <p className="flip-hint">Hover to see algorithm breakdown →</p>
                  </div>
                  {/* Back */}
                  <div className="flip-card-back">
                    <div className="flip-back-label">ALGORITHM BREAKDOWN</div>
                    <div className="flip-back-title">Rule 5: Behavioral Scarcity Surge</div>
                    <ul className="flip-back-list">
                      <li><span className="k">Inventory:</span> <span className="v crit">2 units (Critical)</span></li>
                      <li><span className="k">15m Views:</span> <span className="v">24 interactions</span></li>
                      <li><span className="k">Surge Applied:</span> <span className="v hi">+22%</span></li>
                      <li><span className="k">Base Price:</span> <span className="v">₹2,299</span></li>
                      <li><span className="k">MRP Ceiling:</span> <span className="v">₹2,805 (Enforced)</span></li>
                    </ul>
                    <Link to="/shop" className="landing-button landing-button--primary mt-auto text-xs py-2">
                      View in Storefront
                    </Link>
                  </div>
                </div>
              </div>

              {/* Product 2 — 3D Flip Card */}
              <div className="flip-card">
                <div className="flip-card-inner">
                  {/* Front */}
                  <div className="flip-card-front showcase-product-card">
                    <div className="showcase-top-meta">
                      <span className="showcase-stock-badge safe">18 In Stock</span>
                      <span className="text-[11px] font-mono text-[#34D399]">Equilibrium</span>
                    </div>
                    <h4>Acoustic Pro Studio Headphones</h4>
                    <div className="showcase-pricing-row">
                      <span className="live-price">₹4,499</span>
                      <span className="mrp-price">MRP ₹5,999</span>
                      <span className="text-xs font-mono text-[#34D399]">25% Off</span>
                    </div>
                    <div className="showcase-rule-tag">
                      <Check size={14} />
                      <span>Competitor match + Steady stock buffer</span>
                    </div>
                    <p className="flip-hint">Hover to see algorithm breakdown →</p>
                  </div>
                  {/* Back */}
                  <div className="flip-card-back">
                    <div className="flip-back-label">ALGORITHM BREAKDOWN</div>
                    <div className="flip-back-title">Rule 1: Standard Equilibrium Base</div>
                    <ul className="flip-back-list">
                      <li><span className="k">Inventory:</span> <span className="v">18 units (Healthy)</span></li>
                      <li><span className="k">Competitor Ref:</span> <span className="v">₹4,599</span></li>
                      <li><span className="k">Adjustment:</span> <span className="v safe">-25% (Match)</span></li>
                      <li><span className="k">Base MRP:</span> <span className="v">₹5,999</span></li>
                      <li><span className="k">Floor Guard:</span> <span className="v">₹4,199 (Enforced)</span></li>
                    </ul>
                    <Link to="/shop" className="landing-button landing-button--primary mt-auto text-xs py-2">
                      View in Storefront
                    </Link>
                  </div>
                </div>
              </div>

              {/* Product 3 — 3D Flip Card */}
              <div className="flip-card">
                <div className="flip-card-inner">
                  {/* Front */}
                  <div className="flip-card-front showcase-product-card">
                    <div className="showcase-top-meta">
                      <span className="showcase-stock-badge safe">45 In Stock</span>
                      <span className="text-[11px] font-mono text-[#22D3EE]">Volume Curve</span>
                    </div>
                    <h4>Ergonomic Mechanical Keyboard</h4>
                    <div className="showcase-pricing-row">
                      <span className="live-price">₹3,199</span>
                      <span className="mrp-price">MRP ₹3,999</span>
                      <span className="text-xs font-mono text-[#22D3EE]">20% Off</span>
                    </div>
                    <div className="showcase-rule-tag">
                      <Sparkles size={14} />
                      <span>Segment-optimized for value-seeker session</span>
                    </div>
                    <p className="flip-hint">Hover to see algorithm breakdown →</p>
                  </div>
                  {/* Back */}
                  <div className="flip-card-back">
                    <div className="flip-back-label">ALGORITHM BREAKDOWN</div>
                    <div className="flip-back-title">Rule 3: Segment Volume Curve</div>
                    <ul className="flip-back-list">
                      <li><span className="k">Inventory:</span> <span className="v">45 units (Surplus)</span></li>
                      <li><span className="k">Segment:</span> <span className="v">Value-seeker Intent</span></li>
                      <li><span className="k">Discount:</span> <span className="v safe">-20% Optimized</span></li>
                      <li><span className="k">Base MRP:</span> <span className="v">₹3,999</span></li>
                      <li><span className="k">Floor Guard:</span> <span className="v">₹2,799 (Min)</span></li>
                    </ul>
                    <Link to="/shop" className="landing-button landing-button--primary mt-auto text-xs py-2">
                      View in Storefront
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── 10. Transparency & Compliance ── */}
        <section className="landing-section" aria-labelledby="transparency-heading">
          <div className="landing-shell">
            <div className="landing-section-heading" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Safety & Governance</span>
              </div>
              <h2 id="transparency-heading">
                Pricing guardrails built for <em>regulatory compliance.</em>
              </h2>
              <p className="landing-section-heading__copy">
                PriceIQ avoids predatory pricing through hard mathematical invariants.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5" data-reveal>
              <div className="p-6 rounded-xl bg-[#121827] border border-[#252D40]">
                <div className="w-9 h-9 rounded-lg bg-[#171E30] border border-[#252D40] flex items-center justify-center text-[#22D3EE] mb-4">
                  <ShieldCheck size={20} />
                </div>
                <h3 className="font-display text-base font-bold text-[#F8FAFC] mb-2">MRP Ceiling Invariant</h3>
                <p className="text-xs text-[#A8B1C2] leading-relaxed">
                  No algorithm output can exceed the printed Maximum Retail Price under any demand condition, ensuring
                  strict adherence to consumer protection laws.
                </p>
              </div>

              <div className="p-6 rounded-xl bg-[#121827] border border-[#252D40]">
                <div className="w-9 h-9 rounded-lg bg-[#171E30] border border-[#252D40] flex items-center justify-center text-[#A78BFA] mb-4">
                  <Lock size={20} />
                </div>
                <h3 className="font-display text-base font-bold text-[#F8FAFC] mb-2">70% Margin Floor Guard</h3>
                <p className="text-xs text-[#A8B1C2] leading-relaxed">
                  Deep discounts are bounded at 70% of MRP to prevent predatory race-to-the-bottom pricing and preserve
                  healthy merchant operating margins.
                </p>
              </div>

              <div className="p-6 rounded-xl bg-[#121827] border border-[#252D40]">
                <div className="w-9 h-9 rounded-lg bg-[#171E30] border border-[#252D40] flex items-center justify-center text-[#34D399] mb-4">
                  <Check size={20} />
                </div>
                <h3 className="font-display text-base font-bold text-[#F8FAFC] mb-2">Deterministic Audit Log</h3>
                <p className="text-xs text-[#A8B1C2] leading-relaxed">
                  Every price recommendation is logged alongside the exact inputs (stock count, recent views, competitor
                  reference) for complete internal inspectability.
                </p>
              </div>
            </div>

            <div className="mt-8 p-4 rounded-xl bg-[#0D1220] border border-[#252D40]" data-reveal>
              <span className="text-xs font-mono text-[#A8B1C2] block mb-2 font-bold uppercase">Governance Notes:</span>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-[#6B7280]">
                {METHOD_NOTES.map((note) => (
                  <div key={note} className="flex items-start gap-2">
                    <Check size={14} className="text-[#34D399] mt-0.5 shrink-0" />
                    <span>{note}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── 11. Final Conversion CTA ── */}
        <section className="landing-final-cta" aria-labelledby="cta-heading">
          <div className="landing-shell">
            <div className="cta-box" data-reveal>
              <div className="landing-eyebrow">
                <span className="dot" />
                <span>Ready for Deployment</span>
              </div>
              <h2 id="cta-heading">Experience the pricing intelligence platform today.</h2>
              <p>
                Browse the complete catalog, observe real-time recalculations as you view and cart products, and inspect the
                auditable decision trails.
              </p>
              <div className="cta-actions">
                <Link className="landing-button landing-button--primary" to="/shop">
                  <span>Launch Live Platform</span>
                  <ArrowRight size={17} />
                </Link>
                <a
                  className="landing-button landing-button--outline"
                  href={REPOSITORY_URL}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span>View GitHub Repository</span>
                  <ArrowUpRight size={15} />
                </a>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ── 12. Technical Footer ── */}
      <footer className="landing-footer">
        <div className="landing-shell landing-footer__top">
          <a className="landing-brand" href="#top" aria-label="PriceIQ home">
            <span className="landing-brand__mark">
              <Zap size={18} />
            </span>
            <span>PriceIQ</span>
          </a>
          <p className="text-xs text-[#A8B1C2] m-0">
            Intelligent dynamic commerce pricing with deterministic auditability and real-time session modeling.
          </p>
          <a
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            className="text-xs font-mono text-[#22D3EE] hover:underline flex items-center gap-1.5"
          >
            <span>GitHub Repository</span>
            <ArrowUpRight size={13} />
          </a>
        </div>

        <div className="landing-shell landing-footer__bottom">
          <span>PriceIQ Production Intelligence · {SNAPSHOT.label}</span>
          <div className="flex items-center gap-4">
            {TECH_STACK.slice(0, 4).map((tech) => (
              <span key={tech} className="text-[#6B7280]">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}
