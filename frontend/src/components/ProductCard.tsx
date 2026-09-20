import { Link } from "react-router-dom";
import { Heart, ShoppingCart, Star, Zap, Package, Store } from "lucide-react";
import type { Product } from "@/contexts/ProductContext";
import { useCart } from "@/contexts/CartContext";
import { useWishlist } from "@/contexts/WishlistContext";
import { useProducts } from "@/contexts/ProductContext";
import { trackEvent } from "@/api";

interface ProductCardProps {
  product: Product;
}

export default function ProductCard({ product }: ProductCardProps) {
  const { addToCart } = useCart();
  const { isWishlisted, toggleWishlist } = useWishlist();
  const { sessionId, userId } = useProducts();
  const wishlisted = isWishlisted(product.id);

  const mainImage = product.images?.[0] || "";
  const isDynamic = product.priceReason !== "Standard Price";

  const track = (event: string, extra?: Record<string, unknown>) =>
    trackEvent(event, product.id, sessionId, userId, {
      productName: product.name,
      category:    product.category,
      price:       product.livePrice,
      source:      product.source || "local",
      sourceId:    product.sourceId,
      ...extra,
    });

  const handleAddToCart = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    addToCart(product);
    track("add_to_cart");
  };

  const handleWishlist = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    toggleWishlist(product);
    if (!wishlisted) track("wishlist_add");
  };

  return (
    <div className="group relative flex flex-col rounded-xl border border-gray-700/50 bg-gray-800 overflow-hidden transition-all duration-200 hover:-translate-y-1 hover:shadow-xl hover:shadow-black/30 hover:border-orange-500/30 min-h-[340px]">

      {/* Wishlist */}
      <button
        onClick={handleWishlist}
        className="absolute top-2 right-2 z-10 p-1.5 rounded-full bg-gray-900/90 backdrop-blur-sm transition-all hover:scale-110 shadow-sm"
        aria-label={wishlisted ? "Remove from wishlist" : "Add to wishlist"}
      >
        <Heart size={15} className={wishlisted ? "fill-rose-500 text-rose-500" : "text-gray-400 hover:text-rose-400"} />
      </button>

      {/* Discount badge */}
      {product.discount > 0 && (
        <span className="absolute top-2 left-2 z-10 rounded-md bg-red-500/90 px-1.5 py-0.5 text-[10px] font-bold text-white leading-tight">
          -{product.discount}%
        </span>
      )}

      {/* AI price badge */}
      {isDynamic && (
        <div className="absolute top-9 left-2 z-10">
          <span className="flex items-center gap-0.5 rounded-md bg-orange-500/90 px-1.5 py-0.5 text-[9px] font-bold text-white">
            <Zap size={8} /> AI
          </span>
        </div>
      )}

      {/* ML recommendation match confidence badge */}
      {product._recConfidence !== undefined && product._recConfidence > 0 && (
        <div className="absolute top-2 left-12 z-10">
          <span className="flex items-center gap-1 rounded-md bg-blue-600/90 backdrop-blur-sm px-1.5 py-0.5 text-[9px] font-bold text-white shadow-sm">
            🎯 {product._recConfidence}%
          </span>
        </div>
      )}

      {/* Image — white bg, object-contain */}
      <Link to={`/product/${product.id}`} className="block" onClick={() => track("product_click")}>
        <div className="relative overflow-hidden bg-white h-52 flex items-center justify-center">
          {mainImage ? (
            <img
              src={mainImage}
              alt={product.name}
              className="absolute inset-0 h-full w-full object-contain p-3 transition-transform duration-300 group-hover:scale-105"
              loading="lazy"
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          ) : (
            <ShoppingCart size={40} className="text-gray-300" />
          )}
        </div>
      </Link>

      {/* Info */}
      <Link to={`/product/${product.id}`} onClick={() => track("product_click")} className="block flex-1 p-3 space-y-1 border-t border-gray-700/40">

        {/* Brand + source badge row */}
        <div className="flex items-center justify-between gap-1 min-h-[1.25rem]">
          <p className="text-[10px] font-medium text-gray-500 uppercase tracking-wide truncate">{product.brand}</p>
          {product.source === "amazon" && (
            <span className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[8px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 flex-shrink-0">
              <Package size={7} /> Amazon
            </span>
          )}
          {product.source === "flipkart" && (
            <span className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[8px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30 flex-shrink-0">
              <Store size={7} /> Flipkart
            </span>
          )}
        </div>

        {/* Name — line-clamp-2, consistent height */}
        <h3 className="text-sm font-semibold text-white line-clamp-2 leading-snug min-h-[2.5rem] hover:text-orange-400 transition-colors">
          {product.name}
        </h3>

        {/* Stars */}
        <div className="flex items-center gap-1">
          <div className="flex">
            {[1, 2, 3, 4, 5].map(i => (
              <Star key={i} size={10} className={i <= Math.round(product.rating) ? "fill-amber-400 text-amber-400" : "text-gray-600"} />
            ))}
          </div>
          <span className="text-[10px] text-gray-500">({product.reviewCount.toLocaleString("en-IN")})</span>
        </div>

        {/* Price — tabular-nums */}
        <div className="flex items-center gap-2 flex-wrap pt-1">
          <span className="text-base font-bold text-green-400 tabular-nums">
            ₹{product.livePrice.toLocaleString("en-IN")}
          </span>
          {product.mrp > product.livePrice && (
            <span className="text-xs text-gray-500 line-through tabular-nums">
              ₹{product.mrp.toLocaleString("en-IN")}
            </span>
          )}
          {product.discount > 0 && (
            <span className="rounded-full bg-red-500/15 px-1.5 py-0 text-[10px] text-red-400 font-semibold">
              {product.discount}% off
            </span>
          )}
        </div>
      </Link>

      {/* Demand badge + Add to Cart */}
      {product.demandBadge && (
        <div className="mx-3 mb-1 bg-orange-500/15 text-orange-400 text-[10px] font-semibold text-center py-0.5 rounded">
          🔥 {product.demandBadge}
        </div>
      )}
      <button
        onClick={handleAddToCart}
        className="mx-3 mb-3 flex items-center justify-center gap-1.5 rounded-lg bg-orange-500 py-2 text-xs font-bold text-white transition-colors hover:bg-orange-600 active:scale-95"
      >
        <ShoppingCart size={13} /> Add to Cart
      </button>
    </div>
  );
}
