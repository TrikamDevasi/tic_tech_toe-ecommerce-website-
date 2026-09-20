/**
 * Landing-page facts and worked examples.
 *
 * Sources: backend/src/services/pricingEngine.js, ml-service /health and
 * /evaluate responses, and the local catalog endpoint. Keep customer-facing
 * numerical claims here so the audit trail stays small and explicit.
 */

export type BenchmarkKey = "gru4rec" | "popularity" | "recentlyViewed" | "contentTfidf";

export type BenchmarkRow = {
  key: BenchmarkKey;
  label: string;
  hitAt10: number;
  ndcgAt10: number;
  coverage: number;
  averageLatencyMs: number;
  p95LatencyMs: number;
};

export const SNAPSHOT = {
  label: "16 Sep 2026 snapshot",
  catalogItems: 215,
  modelVocabulary: 217,
  testSamples: 300,
  dataQuality: {
    totalEvents: 98_576,
    usableSessions: 2_000,
  },
} as const;

// Committed fallback for GET http://localhost:8000/evaluate on 16 Sep 2026.
export const BENCHMARK_SNAPSHOT: BenchmarkRow[] = [
  {
    key: "gru4rec",
    label: "GRU4Rec",
    hitAt10: 0.0433,
    ndcgAt10: 0.0179,
    coverage: 0.5302,
    averageLatencyMs: 1.09,
    p95LatencyMs: 1.74,
  },
  {
    key: "popularity",
    label: "Popularity",
    hitAt10: 0.06,
    ndcgAt10: 0.0313,
    coverage: 0.1674,
    averageLatencyMs: 0.01,
    p95LatencyMs: 0.01,
  },
  {
    key: "recentlyViewed",
    label: "Recently viewed",
    hitAt10: 0.0433,
    ndcgAt10: 0.0158,
    coverage: 1,
    averageLatencyMs: 0.01,
    p95LatencyMs: 0.01,
  },
  {
    key: "contentTfidf",
    label: "Content TF-IDF",
    hitAt10: 0.0433,
    ndcgAt10: 0.0199,
    coverage: 1,
    averageLatencyMs: 0.43,
    p95LatencyMs: 0.62,
  },
];

export const PRICING_RULES = {
  basePrice: 2_299,
  lowStockDemandAdjustment: 0.22,
  lowStockThreshold: 3,
  demandThreshold: 20,
  minimumOfMrp: 0.7,
  maximumOfMrp: 1,
  modeledCompetitorDiscount: 0.05,
} as const;

export const WORKED_PRICE = {
  product: "TrailPack 28L",
  inventory: 2,
  recentViews: 24,
  basePrice: PRICING_RULES.basePrice,
  modeledReference: 2_360,
  recommendedPrice: 2_805,
  rationale: "Low stock + elevated recent views selects the +22% rule.",
} as const;

export const NAV_ITEMS = [
  ["Product", "#product"],
  ["How it works", "#how-it-works"],
  ["Benchmarks", "#benchmarks"],
  ["Architecture", "#architecture"],
] as const;

export const REPOSITORY_URL = "https://github.com/TrikamDevasi/tic_tech_toe-ecommerce-website-";

export const METHOD_NOTES = [
  "Pricing decisions are rule-based. The implementation bounds a price between 70% and 100% of MRP.",
  "The competitor reference used by the pricing engine is modeled from a randomized ±5% range and cached; it is not a live market feed.",
  "Marketplace catalog responses may be cached. Derived stock, demand, and price-reason labels from that feed are not treated as facts here.",
  "The demo's ML endpoint reports a non-synthetic dataset flag, but this repository does not certify the underlying training-data provenance.",
] as const;

export const TECH_STACK = [
  "React",
  "TypeScript",
  "Vite",
  "Express",
  "MongoDB",
  "Redis",
  "FastAPI",
  "PyTorch",
  "GRU4Rec",
] as const;

export const PREVIEWS = {
  pricing: {
    title: "Price decision",
    eyebrow: "WORKED EXAMPLE — NOT LIVE",
    rows: [
      ["Base price", "₹2,299"],
      ["Inventory", "2 units"],
      ["Recent views", "24"],
      ["Selected rule", "+22%"],
    ],
  },
  recommendations: {
    title: "Session recommendation",
    eyebrow: "STRATEGY LADDER",
    rows: [
      ["0 interactions", "Popularity"],
      ["1–2 interactions", "Content TF-IDF"],
      ["3+ interactions", "GRU4Rec"],
      ["Endpoint", "/recommend/session"],
    ],
  },
  observability: {
    title: "Event path",
    eyebrow: "WHEN REDIS + WORKER ARE CONFIGURED",
    rows: [
      ["Capture", "POST /api/track"],
      ["Queue", "Redis Stream"],
      ["Process", "streamWorker"],
      ["Persist", "MongoDB Event"],
    ],
  },
} as const;
