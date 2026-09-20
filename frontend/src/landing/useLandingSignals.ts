import { useEffect, useState } from "react";
import { BENCHMARK_SNAPSHOT, type BenchmarkKey, type BenchmarkRow, SNAPSHOT } from "./claims";

type SignalSource = "live" | "snapshot";

export type LandingSignals = {
  benchmark: BenchmarkRow[];
  catalogItems: number;
  modelVocabulary: number;
  testSamples: number;
  source: SignalSource;
  updatedAt: string;
  isLoading: boolean;
  error: boolean;
};

const initialSignals: LandingSignals = {
  benchmark: BENCHMARK_SNAPSHOT,
  catalogItems: SNAPSHOT.catalogItems,
  modelVocabulary: SNAPSHOT.modelVocabulary,
  testSamples: SNAPSHOT.testSamples,
  source: "snapshot",
  updatedAt: SNAPSHOT.label,
  isLoading: true,
  error: false,
};

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | null {
  return typeof value === "object" && value !== null ? value as UnknownRecord : null;
}

function getNumber(record: UnknownRecord | null, keys: string[], fallback: number) {
  if (!record) return fallback;
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return fallback;
}

const strategyAliases: Record<BenchmarkKey, string[]> = {
  gru4rec: ["gru4rec", "GRU4Rec", "gru"],
  popularity: ["popularity", "Popularity"],
  recentlyViewed: ["recently_viewed", "RecentlyViewed", "recentlyViewed"],
  contentTfidf: ["content_tfidf", "Content_TFIDF", "contentTfidf"],
};

function toBenchmarkRows(payload: unknown): BenchmarkRow[] | null {
  const root = asRecord(payload);
  const candidates = [root?.results, root?.strategies, root?.metrics, root?.evaluation];
  const container = candidates.map(asRecord).find(Boolean) ?? root;
  if (!container) return null;

  const rows = BENCHMARK_SNAPSHOT.map((fallback) => {
    const source = strategyAliases[fallback.key]
      .map((alias) => asRecord(container[alias]))
      .find(Boolean);
    if (!source) return fallback;

    return {
      ...fallback,
      hitAt10: getNumber(source, ["hit_at_10", "hit@10", "hitAt10"], fallback.hitAt10),
      ndcgAt10: getNumber(source, ["ndcg_at_10", "ndcg@10", "ndcgAt10"], fallback.ndcgAt10),
      coverage: getNumber(source, ["catalog_coverage", "coverage"], fallback.coverage),
      averageLatencyMs: getNumber(source, ["average_latency_ms", "avg_latency_ms", "averageLatencyMs"], fallback.averageLatencyMs),
      p95LatencyMs: getNumber(source, ["p95_latency_ms", "p95LatencyMs"], fallback.p95LatencyMs),
    };
  });

  return rows.some((row, index) => row !== BENCHMARK_SNAPSHOT[index]) ? rows : null;
}

function productCount(payload: unknown) {
  if (Array.isArray(payload)) return payload.length;
  const record = asRecord(payload);
  return Array.isArray(record?.products) ? record.products.length : SNAPSHOT.catalogItems;
}

function localEndpoint(path: string, fallback: string) {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return configured ? `${configured.replace(/\/$/, "")}${path}` : fallback;
}

export function useLandingSignals() {
  const [signals, setSignals] = useState<LandingSignals>(initialSignals);

  useEffect(() => {
    const controller = new AbortController();
    const mlBaseUrl = (import.meta.env.VITE_ML_SERVICE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://localhost:8000";

    Promise.allSettled([
      fetch(`${mlBaseUrl}/evaluate`, { signal: controller.signal }).then((response) => {
        if (!response.ok) throw new Error("Evaluation service unavailable");
        return response.json();
      }),
      fetch(`${mlBaseUrl}/health`, { signal: controller.signal }).then((response) => {
        if (!response.ok) throw new Error("Model health unavailable");
        return response.json();
      }),
      fetch(localEndpoint("/api/products", "http://localhost:5000/api/products"), { signal: controller.signal }).then((response) => {
        if (!response.ok) throw new Error("Catalog unavailable");
        return response.json();
      }),
    ]).then(([evaluationResult, healthResult, productsResult]) => {
      if (controller.signal.aborted) return;
      const evaluation = evaluationResult.status === "fulfilled" ? asRecord(evaluationResult.value) : null;
      const health = healthResult.status === "fulfilled" ? asRecord(healthResult.value) : null;
      const benchmark = toBenchmarkRows(evaluation);
      const hasLiveBenchmark = Boolean(benchmark);
      const timestamp = typeof evaluation?.evaluation_timestamp === "string"
        ? evaluation.evaluation_timestamp
        : SNAPSHOT.label;

      setSignals({
        benchmark: benchmark ?? BENCHMARK_SNAPSHOT,
        catalogItems: productsResult.status === "fulfilled" ? productCount(productsResult.value) : SNAPSHOT.catalogItems,
        modelVocabulary: getNumber(health, ["vocab_size", "vocabulary_size"], SNAPSHOT.modelVocabulary),
        testSamples: getNumber(evaluation, ["test_samples_count", "test_samples"], SNAPSHOT.testSamples),
        source: hasLiveBenchmark ? "live" : "snapshot",
        updatedAt: hasLiveBenchmark ? `live evaluation · ${timestamp}` : SNAPSHOT.label,
        isLoading: false,
        error: !hasLiveBenchmark || evaluationResult.status === "rejected" || healthResult.status === "rejected" || productsResult.status === "rejected",
      });
    }).catch(() => {
      if (!controller.signal.aborted) setSignals({ ...initialSignals, isLoading: false, error: true });
    });

    return () => controller.abort();
  }, []);

  return signals;
}
