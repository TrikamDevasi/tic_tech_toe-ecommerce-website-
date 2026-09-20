import { ChevronLeft, ChevronRight, Sparkles, AlertCircle, Cpu } from "lucide-react";
import { useRef } from "react";
import type { Product } from "@/contexts/ProductContext";
import ProductCard from "./ProductCard";

export default function RecommendationRow({ title, products }: { title: string; products: Product[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const scroll = (dir: number) => ref.current?.scrollBy({ left: dir * 300, behavior: "smooth" });

  if (!products || products.length === 0) return null;

  // Check if any product contains ML metadata
  const mlProduct = products.find((p) => p._recReason?.startsWith("ml-") || p._recConfidence !== undefined);
  const isLowConfidence = products.some((p) => p._isLowConfidence);
  const avgConfidence = mlProduct
    ? Math.round(
        products.reduce((acc, p) => acc + (p._recConfidence || 0), 0) /
          Math.max(1, products.filter((p) => p._recConfidence !== undefined).length)
      )
    : null;

  return (
    <section className="space-y-3 rounded-2xl border border-border/40 bg-card/30 p-4 backdrop-blur-sm">
      <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-bold text-foreground">{title}</h2>
          {mlProduct && (
            <div className="flex items-center gap-1.5">
              <span className="inline-flex items-center gap-1 rounded-full bg-orange-500/10 border border-orange-500/30 px-2 py-0.5 text-[11px] font-semibold text-orange-400">
                <Cpu size={12} />
                PyTorch GRU4Rec ML
              </span>
              {avgConfidence !== null && avgConfidence > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-blue-500/10 border border-blue-500/30 px-2 py-0.5 text-[11px] font-medium text-blue-400">
                  <Sparkles size={11} />
                  {avgConfidence}% Match Confidence
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <button
              onClick={() => scroll(-1)}
              className="rounded-full border border-border p-1.5 text-muted-foreground hover:text-accent hover:border-accent transition-colors"
              aria-label="Scroll left"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => scroll(1)}
              className="rounded-full border border-border p-1.5 text-muted-foreground hover:text-accent hover:border-accent transition-colors"
              aria-label="Scroll right"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {isLowConfidence && (
        <div className="flex items-center gap-2 rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-1.5 text-xs text-amber-300">
          <AlertCircle size={14} className="shrink-0 text-amber-400" />
          <span>Low model confidence for current session depth. Showing diversified trending items.</span>
        </div>
      )}

      <div ref={ref} className="flex gap-4 overflow-x-auto scrollbar-hide pb-2">
        {products.map((p) => (
          <div key={p.id} className="min-w-[200px] max-w-[200px]">
            <ProductCard product={p} />
          </div>
        ))}
      </div>

      <div className="text-[11px] text-muted-foreground/70 flex items-center justify-between border-t border-border/20 pt-2">
        <span>* Recommendations are generated via trained sequential neural networks and collaborative session signals.</span>
      </div>
    </section>
  );
}
