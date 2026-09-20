import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { fetchMarketplaceHome, fetchMarketplaceCategory, getOrCreateSessionId, getOrCreateUserId } from "@/api";
import { products as staticProducts } from "@/data/products";
import { toast } from "sonner";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// ── Types ─────────────────────────────────────────────────────────────────────
export interface Product {
  id: string | number;
  source?: 'amazon' | 'flipkart' | 'local';
  sourceId?: string;
  name: string;
  brand: string;
  category: string;
  mrp: number;
  livePrice: number;
  basePrice?: number;
  stock: number;
  rating: number;
  reviewCount: number;
  discount: number;
  priceReason: "High Demand" | "Limited Stock" | "Competitor Match" | "Standard Price";
  demandBadge: string | null;
  images: string[];
  description: string;
  specs: Record<string, string>;
  productUrl?: string;
  viewCount?: number;
  cartAddCount?: number;
  purchaseCount?: number;
  amazonUrl?: string;
  asin?: string;
  _recReason?: string;
  _recStrategy?: string;
  _recConfidence?: number;
  _isLowConfidence?: boolean;
  _confidenceMessage?: string;
}

export interface MarketplaceFeed {
  featured: Product[];
  flashDeals: Product[];
  electronics: Product[];
  fashion: Product[];
  homeKitchen: Product[];
  beauty: Product[];
  books: Product[];
  sports: Product[];
  toys?: Product[];
  toysAvailable?: boolean;
  cached?: boolean;
}

export type FeedSource = "live" | "cached" | "local";

interface ProductContextType {
  products: Product[];
  marketplaceFeed: MarketplaceFeed | null;
  feedSource: FeedSource;
  loading: boolean;
  apiError: boolean;
  categoryProducts: Product[];
  categoryLoading: boolean;
  updateProductPrice: (id: string | number, price: number, reason: string, discount: number) => void;
  getProductById: (id: string | number) => Product | undefined;
  loadCategory: (slug: string) => Promise<void>;
  sessionId: string;
  userId: string;
}

const sessionId = getOrCreateSessionId();
const userId = getOrCreateUserId();

const EMPTY_FEED: MarketplaceFeed = {
  featured: [], flashDeals: [], electronics: [], fashion: [],
  homeKitchen: [], beauty: [], books: [], sports: [],
};

const ProductContext = createContext<ProductContextType>({
  products: [], marketplaceFeed: null, feedSource: "local",
  loading: true, apiError: false, categoryProducts: [], categoryLoading: false,
  updateProductPrice: () => {}, getProductById: () => undefined,
  loadCategory: async () => {}, sessionId, userId,
});

export const useProducts = () => useContext(ProductContext);

// label → slug mapping
export const CATEGORY_SLUGS: Record<string, string> = {
  "Electronics":   "electronics",
  "Fashion":       "fashion",
  "Home & Kitchen":"home-kitchen",
  "Beauty":        "beauty",
  "Books":         "books",
  "Sports":        "sports",
  "Toys":          "toys",
};

export const ProductProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [marketplaceFeed, setMarketplaceFeed] = useState<MarketplaceFeed | null>(null);
  const [feedSource, setFeedSource] = useState<FeedSource>("local");
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState(false);
  const [categoryProducts, setCategoryProducts] = useState<Product[]>([]);
  const [categoryLoading, setCategoryLoading] = useState(false);
  const productsRef = useRef<Product[]>([]);
  productsRef.current = products;

  // ── Load marketplace home feed ─────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const feed = await fetchMarketplaceHome() as MarketplaceFeed & { cached?: boolean };

        // Flatten all sections into deduplicated products list
        const all: Product[] = [
          ...(feed.featured || []),
          ...(feed.flashDeals || []),
          ...(feed.electronics || []),
          ...(feed.fashion || []),
          ...(feed.homeKitchen || []),
          ...(feed.beauty || []),
          ...(feed.books || []),
          ...(feed.sports || []),
          ...(feed.toys || []),
        ];
        const unique = Array.from(new Map(all.map(p => [p.id, p])).values());

        if (unique.length > 0) {
          setProducts(unique);
          setMarketplaceFeed(feed);
          setFeedSource(feed.cached ? "cached" : "live");
          setApiError(false);
        } else {
          throw new Error("Empty marketplace feed");
        }
      } catch (err) {
        console.warn("Marketplace feed failed, falling back:", err);
        setApiError(true);
        setMarketplaceFeed(EMPTY_FEED);
        setFeedSource("local");
        try {
          const local = await fetch(`${BASE_URL}/api/products`).then(r => r.json());
          setProducts(Array.isArray(local) && local.length > 0 ? local : staticProducts as unknown as Product[]);
        } catch {
          setProducts(staticProducts as unknown as Product[]);
        }
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  // ── Load products for a specific category ─────────────────────────────────
  const loadCategory = useCallback(async (slug: string) => {
    if (!slug || slug === "all") {
      setCategoryProducts([]);
      return;
    }
    setCategoryLoading(true);
    try {
      const data = await fetchMarketplaceCategory(slug) as { products: Product[]; cached?: boolean };
      setCategoryProducts(Array.isArray(data) ? data : (data?.products || []));
    } catch {
      setCategoryProducts([]);
    } finally {
      setCategoryLoading(false);
    }
  }, []);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const updateProductPrice = useCallback(
    (id: string | number, price: number, reason: string, discount: number) => {
      setProducts(prev =>
        prev.map(p =>
          String(p.id) === String(id)
            ? { ...p, livePrice: price, priceReason: reason as Product["priceReason"], discount }
            : p
        )
      );
    },
    []
  );

  const getProductById = useCallback(
    (id: string | number) => productsRef.current.find(p => String(p.id) === String(id)),
    []
  );

  return (
    <ProductContext.Provider value={{
      products, marketplaceFeed, feedSource, loading, apiError,
      categoryProducts, categoryLoading,
      updateProductPrice, getProductById, loadCategory,
      sessionId, userId,
    }}>
      {children}
    </ProductContext.Provider>
  );
};
