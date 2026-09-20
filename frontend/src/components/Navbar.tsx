import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Zap, Search, Heart, ShoppingCart, BarChart2, Menu, X,
  Layers, Smartphone, Shirt, UtensilsCrossed, Dumbbell, Sparkles, BookOpen, Gamepad2,
  LogOut, LogIn, ChevronDown, Sun, Moon,
} from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";
import { useCart } from "@/contexts/CartContext";
import { useWishlist } from "@/contexts/WishlistContext";
import { useAuth } from "@/contexts/AuthContext";
import { useState, useRef, useEffect } from "react";
import { toast } from "sonner";

/* ─── Design tokens (inline CSS vars for the navbar) ──────────────────── */
const NAV_STYLES = `
  .navbar-root {
    --nav-bg: rgba(15, 15, 15, 0.85);
    --nav-border: oklch(1 0 0 / 0.06);
    --nav-accent: #f97316;
    --nav-accent-glow: oklch(0.72 0.18 45 / 0.3);
    --nav-accent-glow-lg: oklch(0.72 0.18 45 / 0.35);
    --nav-border-subtle: oklch(1 0 0 / 0.10);
    --nav-icon-hover-bg: oklch(1 0 0 / 0.06);
    --nav-radius-btn: 0.5rem;
    --nav-radius-pill: 9999px;
    --nav-transition-snappy: all 180ms cubic-bezier(0.16, 1, 0.3, 1);
    --nav-font: 'Plus Jakarta Sans', 'DM Sans', 'Inter', system-ui, sans-serif;
  }

  html.light .navbar-root,
  .light .navbar-root {
    --nav-bg: rgba(255, 255, 255, 0.92);
    --nav-border: #E2E8F0;
    --nav-accent: #4F46E5;
    --nav-border-subtle: #E2E8F0;
    --nav-icon-hover-bg: #F1F5F9;
  }
  html.light .navbar-root .navbar-logo-text .logo-white,
  .light .navbar-root .navbar-logo-text .logo-white {
    color: #111827;
  }
  html.light .navbar-root .navbar-search-form,
  .light .navbar-root .navbar-search-form {
    background: #F0F2F8;
    border-color: #E2E8F0;
  }
  html.light .navbar-root .navbar-search-input,
  .light .navbar-root .navbar-search-input {
    color: #111827;
  }
  html.light .navbar-root .navbar-search-input::placeholder,
  .light .navbar-root .navbar-search-input::placeholder {
    color: #64748B;
  }
  html.light .navbar-root .category-tab-btn,
  .light .navbar-root .category-tab-btn {
    color: #475569;
  }
  html.light .navbar-root .category-tab-btn:hover,
  .light .navbar-root .category-tab-btn:hover {
    color: #111827;
    background: #F1F5F9;
  }
  html.light .navbar-root .navbar-icon-btn,
  .light .navbar-root .navbar-icon-btn {
    color: #475569;
  }
  html.light .navbar-root .navbar-icon-btn:hover,
  .light .navbar-root .navbar-icon-btn:hover {
    color: #111827;
    background: #F1F5F9;
  }

  /* ── Scrolled state: blur appears after 60px ── */
  .navbar-root {
    position: sticky;
    top: 0;
    z-index: 50;
    background: var(--nav-bg);
    backdrop-filter: blur(0px);
    border-bottom: 1px solid var(--nav-border);
    font-family: var(--nav-font);
    transition: backdrop-filter 240ms ease, background 240ms ease;
  }
  .navbar-root.scrolled {
    backdrop-filter: blur(16px) saturate(180%);
    background: var(--nav-bg);
  }

  /* ── Inner container ── */
  .navbar-inner {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 clamp(1rem, 4vw, 2.5rem);
    height: 64px;
    display: flex;
    align-items: center;
    gap: 1.25rem;
  }

  /* ── Logo ── */
  .navbar-logo {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    text-decoration: none;
    flex-shrink: 0;
    user-select: none;
  }
  .navbar-logo-icon {
    color: var(--nav-accent);
    transition: var(--nav-transition-snappy);
  }
  .navbar-logo:hover .navbar-logo-icon {
    filter: drop-shadow(0 0 6px oklch(0.72 0.18 45 / 0.5));
  }
  .navbar-logo-text {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .navbar-logo-text .logo-white { color: #fff; }
  .navbar-logo-text .logo-orange { color: var(--nav-accent); }

  /* ── Search form ── */
  .navbar-search-form {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    height: 44px;
    border-radius: var(--nav-radius-pill);
    border: 1px solid oklch(1 0 0 / 0.12);
    background: oklch(1 0 0 / 0.07);
    overflow: hidden;
    transition: var(--nav-transition-snappy);
    position: relative;
  }
  .navbar-search-form:focus-within {
    border-color: var(--nav-accent);
    box-shadow: 0 0 0 3px oklch(0.72 0.18 45 / 0.2);
    background: oklch(1 0 0 / 0.1);
  }

  /* Platform switcher */
  .platform-switcher {
    display: flex;
    align-items: center;
    height: 100%;
    flex-shrink: 0;
  }
  .platform-select-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0 12px 0 14px;
    height: 100%;
    background: none;
    border: none;
    color: rgba(255,255,255,0.55);
    font-size: 0.78rem;
    font-weight: 600;
    font-family: var(--nav-font);
    cursor: pointer;
    white-space: nowrap;
    transition: var(--nav-transition-snappy);
    position: relative;
  }
  .platform-select-btn:hover { color: rgba(255,255,255,0.9); }
  .platform-divider {
    width: 1px;
    height: 22px;
    background: oklch(1 0 0 / 0.14);
    flex-shrink: 0;
  }

  /* Platform dropdown */
  .platform-dropdown {
    position: absolute;
    top: calc(100% + 8px);
    left: 0;
    background: #1a1a1a;
    border: 1px solid oklch(1 0 0 / 0.12);
    border-radius: 10px;
    overflow: hidden;
    z-index: 100;
    min-width: 130px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    animation: dropdown-in 140ms cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes dropdown-in {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .platform-option {
    display: block;
    width: 100%;
    padding: 9px 14px;
    background: none;
    border: none;
    color: rgba(255,255,255,0.7);
    font-size: 0.82rem;
    font-family: var(--nav-font);
    font-weight: 500;
    text-align: left;
    cursor: pointer;
    transition: var(--nav-transition-snappy);
  }
  .platform-option:hover, .platform-option.active {
    background: oklch(1 0 0 / 0.06);
    color: #fff;
  }
  .platform-option.active { color: var(--nav-accent); font-weight: 600; }

  /* Search icon inside pill */
  .search-icon-left {
    display: flex;
    align-items: center;
    padding: 0 10px;
    color: rgba(255,255,255,0.35);
    flex-shrink: 0;
    pointer-events: none;
  }

  /* Text input */
  .navbar-search-input {
    flex: 1;
    height: 100%;
    background: none;
    border: none;
    outline: none;
    color: #fff;
    font-size: 0.875rem;
    font-family: var(--nav-font);
    padding: 0 8px 0 0;
  }
  .navbar-search-input::placeholder { color: rgba(255,255,255,0.38); }

  /* Clear button */
  .search-clear-btn {
    background: none;
    border: none;
    color: rgba(255,255,255,0.35);
    cursor: pointer;
    padding: 0 6px;
    display: flex;
    align-items: center;
    transition: var(--nav-transition-snappy);
    flex-shrink: 0;
  }
  .search-clear-btn:hover { color: rgba(255,255,255,0.8); }

  /* Search submit button — flush right */
  .search-submit-btn {
    height: 100%;
    padding: 0 20px;
    background: var(--nav-accent);
    color: #fff;
    border: none;
    font-size: 0.875rem;
    font-weight: 600;
    font-family: var(--nav-font);
    cursor: pointer;
    border-radius: 0 var(--nav-radius-pill) var(--nav-radius-pill) 0;
    flex-shrink: 0;
    transition: var(--nav-transition-snappy);
    white-space: nowrap;
  }
  .search-submit-btn:hover {
    background: #fb923c;
    box-shadow: 0 4px 12px oklch(0.72 0.18 45 / 0.35);
  }

  /* ── Right actions ── */
  .navbar-right {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
  }

  /* Icon buttons */
  .nav-icon-btn {
    position: relative;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: 10px;
    color: rgba(255,255,255,0.55);
    cursor: pointer;
    text-decoration: none;
    transition: var(--nav-transition-snappy);
  }
  .nav-icon-btn:hover {
    background: var(--nav-icon-hover-bg);
    color: rgba(255,255,255,0.95);
  }

  /* Tooltip */
  .nav-icon-btn::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: -28px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(20,20,20,0.95);
    color: rgba(255,255,255,0.9);
    font-size: 0.68rem;
    font-family: var(--nav-font);
    font-weight: 500;
    padding: 3px 7px;
    border-radius: 5px;
    white-space: nowrap;
    pointer-events: none;
    opacity: 0;
    transition: opacity 120ms ease;
    z-index: 60;
  }
  .nav-icon-btn:hover::after { opacity: 1; }

  /* Badge */
  .nav-badge {
    position: absolute;
    top: 3px;
    right: 3px;
    min-width: 16px;
    height: 16px;
    background: var(--nav-accent);
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    font-family: var(--nav-font);
    border-radius: var(--nav-radius-pill);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 3px;
    line-height: 1;
    pointer-events: none;
  }

  /* Divider between icons and auth */
  .nav-divider {
    width: 1px;
    height: 20px;
    background: oklch(1 0 0 / 0.1);
    margin: 0 6px;
    flex-shrink: 0;
  }

  /* Auth buttons */
  .btn-ghost {
    padding: 8px 16px;
    border-radius: var(--nav-radius-btn);
    border: 1px solid oklch(1 0 0 / 0.2);
    background: transparent;
    color: #fff;
    font-size: 0.85rem;
    font-weight: 500;
    font-family: var(--nav-font);
    cursor: pointer;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    transition: var(--nav-transition-snappy);
  }
  .btn-ghost:hover {
    border-color: var(--nav-accent);
    color: var(--nav-accent);
  }

  .btn-solid {
    padding: 8px 18px;
    border-radius: var(--nav-radius-btn);
    border: none;
    background: linear-gradient(135deg, #fb923c, #f97316);
    color: #fff;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: var(--nav-font);
    cursor: pointer;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    transition: var(--nav-transition-snappy);
  }
  .btn-solid:hover {
    filter: brightness(1.08);
    box-shadow: 0 4px 12px var(--nav-accent-glow-lg);
  }

  /* ── Avatar dropdown ── */
  .avatar-trigger {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px 4px 4px;
    border-radius: 999px;
    border: 1px solid oklch(1 0 0 / 0.15);
    background: oklch(1 0 0 / 0.06);
    cursor: pointer;
    transition: var(--nav-transition-snappy);
  }
  .avatar-trigger:hover { border-color: oklch(0.72 0.18 45 / 0.5); }
  .avatar-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--nav-accent);
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .avatar-name {
    font-size: 0.8rem;
    font-weight: 500;
    color: rgba(255,255,255,0.85);
    max-width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .avatar-chevron {
    color: rgba(255,255,255,0.4);
    transition: transform 180ms ease;
  }
  .avatar-chevron.open { transform: rotate(180deg); }

  .avatar-menu {
    position: absolute;
    right: 0;
    top: calc(100% + 8px);
    min-width: 190px;
    background: #111;
    border: 1px solid oklch(1 0 0 / 0.12);
    border-radius: 12px;
    overflow: hidden;
    z-index: 100;
    box-shadow: 0 12px 32px rgba(0,0,0,0.6);
    animation: dropdown-in 140ms cubic-bezier(0.16, 1, 0.3, 1);
  }
  .avatar-menu-header {
    padding: 10px 14px;
    border-bottom: 1px solid oklch(1 0 0 / 0.1);
  }
  .avatar-menu-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #fff;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .avatar-menu-email {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.45);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-top: 1px;
  }
  .avatar-menu-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 14px;
    color: rgba(255,255,255,0.6);
    font-size: 0.82rem;
    font-family: var(--nav-font);
    font-weight: 500;
    text-decoration: none;
    background: none;
    border: none;
    width: 100%;
    cursor: pointer;
    transition: var(--nav-transition-snappy);
  }
  .avatar-menu-item:hover { color: #fff; background: oklch(1 0 0 / 0.06); }
  .avatar-menu-item.danger { color: #f87171; }
  .avatar-menu-item.danger:hover { background: rgba(248,113,113,0.08); }

  /* ── Category bar ── */
  .category-bar {
    border-top: 1px solid oklch(1 0 0 / 0.07);
  }
  .category-bar-inner {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0.5rem clamp(1rem, 4vw, 2.5rem);
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .category-bar-inner::-webkit-scrollbar { display: none; }
  .cat-pill {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: var(--nav-font);
    cursor: pointer;
    border: 1px solid oklch(1 0 0 / 0.12);
    background: none;
    color: rgba(255,255,255,0.55);
    transition: var(--nav-transition-snappy);
  }
  .cat-pill:hover { border-color: oklch(0.72 0.18 45 / 0.5); color: rgba(255,255,255,0.9); }
  .cat-pill.active {
    background: var(--nav-accent);
    border-color: var(--nav-accent);
    color: #fff;
    box-shadow: 0 2px 8px oklch(0.72 0.18 45 / 0.3);
  }

  /* ── Mobile ── */
  .navbar-mobile-toggle {
    display: none;
    width: 40px;
    height: 40px;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    border-radius: 10px;
    color: rgba(255,255,255,0.6);
    cursor: pointer;
    flex-shrink: 0;
    transition: var(--nav-transition-snappy);
  }
  .navbar-mobile-toggle-flex {
    display: none;
  }
  .navbar-mobile-toggle:hover { background: var(--nav-icon-hover-bg); color: #fff; }

  @media (max-width: 768px) {
    .navbar-inner { height: 56px; gap: 0.75rem; }
    .navbar-search-desktop { display: none !important; }
    .navbar-right-desktop { display: none !important; }
    .navbar-mobile-toggle { display: flex; }
    .navbar-mobile-toggle-flex { display: flex; }

    /* Mobile: search row below header */
    .navbar-mobile-search {
      padding: 0.6rem 1rem 0.75rem;
      border-bottom: 1px solid oklch(1 0 0 / 0.07);
    }
  }

  @media (min-width: 769px) {
    .navbar-mobile-toggle { display: none; }
    .navbar-mobile-toggle-flex { display: none; }
    .navbar-mobile-search { display: none; }
    .mobile-menu-panel { display: none !important; }
  }

  /* Mobile menu panel */
  .mobile-menu-panel {
    border-top: 1px solid oklch(1 0 0 / 0.08);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    animation: dropdown-in 160ms cubic-bezier(0.16, 1, 0.3, 1);
  }
  .mobile-menu-link {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    font-family: var(--nav-font);
    font-weight: 500;
    text-decoration: none;
    background: none;
    border: none;
    width: 100%;
    cursor: pointer;
    transition: var(--nav-transition-snappy);
  }
  .mobile-menu-link:hover { color: #fff; background: oklch(1 0 0 / 0.06); }
  .mobile-menu-link.danger { color: #f87171; }
  .mobile-menu-link.danger:hover { background: rgba(248,113,113,0.08); }

  .mobile-user-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    background: oklch(1 0 0 / 0.04);
    border: 1px solid oklch(1 0 0 / 0.08);
  }
  .mobile-divider {
    height: 1px;
    background: oklch(1 0 0 / 0.08);
  }
`;

/* ─── Categories ────────────────────────────────────────────────────────── */
const CATEGORIES = [
  { label: "All",            icon: <Layers size={13} /> },
  { label: "Electronics",    icon: <Smartphone size={13} /> },
  { label: "Fashion",        icon: <Shirt size={13} /> },
  { label: "Home & Kitchen", icon: <UtensilsCrossed size={13} /> },
  { label: "Sports",         icon: <Dumbbell size={13} /> },
  { label: "Beauty",         icon: <Sparkles size={13} /> },
  { label: "Books",          icon: <BookOpen size={13} /> },
  { label: "Toys",           icon: <Gamepad2 size={13} /> },
];

const PLATFORMS = ["All", "Amazon", "Flipkart"];

interface NavbarProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  activeCategory: string;
  onCategoryChange: (c: string) => void;
}

export default function Navbar({ searchQuery, onSearchChange, activeCategory, onCategoryChange }: NavbarProps) {
  const { isDark, toggleTheme } = useTheme();
  const { cartCount }           = useCart();
  const { items: wishlistItems }= useWishlist();
  const { user, isAuthenticated, signOut } = useAuth();

  const [mobileOpen, setMobileOpen]     = useState(false);
  const [avatarOpen, setAvatarOpen]     = useState(false);
  
  // Platform dropdown state (desktop)
  const [platformOpenDesktop, setPlatformOpenDesktop] = useState(false);
  // Platform dropdown state (mobile)
  const [platformOpenMobile, setPlatformOpenMobile] = useState(false);
  
  const [platform, setPlatform]         = useState("All");
  const [scrolled, setScrolled]         = useState(false);
  const [searchInput, setSearchInput]   = useState(searchQuery || "");

  const location  = useLocation();
  const navigate  = useNavigate();
  const avatarRef   = useRef<HTMLDivElement>(null);
  const platformRefDesktop = useRef<HTMLDivElement>(null);
  const platformRefMobile = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* Sync search prop */
  useEffect(() => { setSearchInput(searchQuery || ""); }, [searchQuery]);
  useEffect(() => { return () => { if (debounceRef.current) clearTimeout(debounceRef.current); }; }, []);

  /* Scrolled state */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  /* Close dropdowns on outside click */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (avatarRef.current   && !avatarRef.current.contains(e.target as Node))   setAvatarOpen(false);
      if (platformRefDesktop.current && !platformRefDesktop.current.contains(e.target as Node)) setPlatformOpenDesktop(false);
      if (platformRefMobile.current && !platformRefMobile.current.contains(e.target as Node)) setPlatformOpenMobile(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  /* Handlers */
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchInput(value);
    onSearchChange(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (value.trim().length >= 2) {
      debounceRef.current = setTimeout(() => {
        navigate(`/search?q=${encodeURIComponent(value.trim())}`);
      }, 300);
    }
  };

  const handleSearchSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (searchInput.trim()) navigate(`/search?q=${encodeURIComponent(searchInput.trim())}`);
  };

  const handleClear = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setSearchInput("");
    onSearchChange("");
  };

  const handleSignOut = () => {
    signOut();
    setAvatarOpen(false);
    setMobileOpen(false);
    toast.success("Signed out successfully");
    navigate("/");
  };

  const initials = user?.name
    ? user.name.split(" ").map((w: string) => w[0]).join("").toUpperCase().slice(0, 2)
    : "?";

  const showCategoryBar = location.pathname === "/" || location.pathname === "/wishlist";
  const isAuthPage = location.pathname === "/login" || location.pathname === "/signup";
  if (isAuthPage) return null;

  return (
    <>
      {/* Inject scoped styles */}
      <style>{NAV_STYLES}</style>

      <header className={`navbar-root ${scrolled ? "scrolled" : ""}`}>
        {/* ── Main row ── */}
        <div className="navbar-inner">

          {/* Logo */}
          <Link to="/" className="navbar-logo">
            <Zap size={26} className="navbar-logo-icon" strokeWidth={2.5} />
            <span className="navbar-logo-text">
              <span className="logo-white">Price</span><span className="logo-orange">IQ</span>
            </span>
          </Link>

          {/* Search — Desktop */}
          <form onSubmit={handleSearchSubmit} className="navbar-search-form navbar-search-desktop" style={{ maxWidth: "none" }}>
            {/* Platform switcher */}
            <div className="platform-switcher" ref={platformRefDesktop} style={{ position: "relative" }}>
              <button
                type="button"
                className="platform-select-btn"
                onClick={() => setPlatformOpenDesktop(v => !v)}
                aria-label="Select platform"
              >
                {platform}
                <ChevronDown size={12} style={{ transition: "transform 180ms ease", transform: platformOpenDesktop ? "rotate(180deg)" : "none" }} />
              </button>
              {platformOpenDesktop && (
                <div className="platform-dropdown">
                  {PLATFORMS.map(p => (
                    <button
                      key={p}
                      type="button"
                      className={`platform-option ${platform === p ? "active" : ""}`}
                      onClick={() => { setPlatform(p); setPlatformOpenDesktop(false); }}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="platform-divider" />

            {/* Search icon */}
            <div className="search-icon-left">
              <Search size={16} />
            </div>

            {/* Input */}
            <input
              id="navbar-search-desktop"
              type="text"
              value={searchInput}
              onChange={handleInputChange}
              onKeyDown={(e) => e.key === "Enter" && handleSearchSubmit()}
              placeholder="Search Amazon + Flipkart products..."
              autoComplete="off"
              className="navbar-search-input"
            />

            {/* Clear */}
            {searchInput && (
              <button type="button" onClick={handleClear} className="search-clear-btn" aria-label="Clear search">
                <X size={14} />
              </button>
            )}

            {/* Submit */}
            <button type="submit" className="search-submit-btn">Search</button>
          </form>

          {/* Right — Desktop */}
          <div className="navbar-right navbar-right-desktop">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="nav-icon-btn"
              data-tooltip={isDark ? "Light mode" : "Dark mode"}
              aria-label="Toggle theme"
            >
              {isDark ? <Sun size={18} /> : <Moon size={18} />}
            </button>

            {/* Wishlist */}
            <Link to="/wishlist" className="nav-icon-btn" data-tooltip="Wishlist" aria-label="Wishlist">
              <Heart size={18} />
              {wishlistItems.length > 0 && (
                <span className="nav-badge">{wishlistItems.length}</span>
              )}
            </Link>

            {/* Cart */}
            <Link to="/cart" className="nav-icon-btn" data-tooltip="Cart" aria-label="Cart">
              <ShoppingCart size={18} />
              {cartCount > 0 && (
                <span className="nav-badge">{cartCount}</span>
              )}
            </Link>

            {/* Divider */}
            <div className="nav-divider" />

            {/* Auth */}
            {isAuthenticated ? (
              <div style={{ position: "relative" }} ref={avatarRef}>
                <button
                  onClick={() => setAvatarOpen(v => !v)}
                  className="avatar-trigger"
                  aria-label="Account menu"
                >
                  <div className="avatar-circle">{initials}</div>
                  <span className="avatar-name">{user?.name?.split(" ")[0]}</span>
                  <ChevronDown size={12} className={`avatar-chevron ${avatarOpen ? "open" : ""}`} />
                </button>

                {avatarOpen && (
                  <div className="avatar-menu">
                    <div className="avatar-menu-header">
                      <p className="avatar-menu-name">{user?.name}</p>
                      <p className="avatar-menu-email">{user?.email}</p>
                    </div>
                    <div style={{ padding: "4px 0" }}>
                      {[
                        { to: "/wishlist",  icon: <Heart size={14} />,    label: "Wishlist" },
                        { to: "/dashboard", icon: <BarChart2 size={14} />, label: "Dashboard" },
                      ].map(item => (
                        <Link key={item.to} to={item.to}
                          onClick={() => setAvatarOpen(false)}
                          className="avatar-menu-item">
                          {item.icon} {item.label}
                        </Link>
                      ))}
                      <button onClick={handleSignOut} className="avatar-menu-item danger">
                        <LogOut size={14} /> Sign out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <Link to="/login" className="btn-ghost">
                  <LogIn size={15} /> Sign In
                </Link>
                <Link to="/signup" className="btn-solid">
                  <Zap size={14} strokeWidth={2.5} /> Join free
                </Link>
              </div>
            )}
          </div>

          {/* Mobile: icon pills + hamburger */}
          <div className="navbar-right" style={{ gap: "2px", marginLeft: "auto" }}>
            <div className="navbar-mobile-toggle-flex">
              <Link to="/cart" className="nav-icon-btn" data-tooltip="Cart" aria-label="Cart" style={{ display: "flex" }}>
                <ShoppingCart size={18} />
                {cartCount > 0 && <span className="nav-badge">{cartCount}</span>}
              </Link>
            </div>
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="navbar-mobile-toggle navbar-mobile-toggle-flex"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* ── Mobile: search row ── */}
        <div className="navbar-mobile-search">
          <form onSubmit={handleSearchSubmit} className="navbar-search-form" style={{ maxWidth: "none", display: "flex" }}>
            {/* Platform switcher */}
            <div className="platform-switcher" ref={platformRefMobile} style={{ position: "relative" }}>
              <button
                type="button"
                className="platform-select-btn"
                onClick={() => setPlatformOpenMobile(v => !v)}
                aria-label="Select platform"
              >
                {platform}
                <ChevronDown size={12} style={{ transition: "transform 180ms ease", transform: platformOpenMobile ? "rotate(180deg)" : "none" }} />
              </button>
              {platformOpenMobile && (
                <div className="platform-dropdown">
                  {PLATFORMS.map(p => (
                    <button
                      key={p}
                      type="button"
                      className={`platform-option ${platform === p ? "active" : ""}`}
                      onClick={() => { setPlatform(p); setPlatformOpenMobile(false); }}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="platform-divider" />

            {/* Search icon */}
            <div className="search-icon-left">
              <Search size={16} />
            </div>

            {/* Input */}
            <input
              id="navbar-search-mobile"
              type="text"
              value={searchInput}
              onChange={handleInputChange}
              onKeyDown={(e) => e.key === "Enter" && handleSearchSubmit()}
              placeholder="Search products..."
              autoComplete="off"
              className="navbar-search-input"
            />

            {/* Clear */}
            {searchInput && (
              <button type="button" onClick={handleClear} className="search-clear-btn" aria-label="Clear search">
                <X size={14} />
              </button>
            )}

            {/* Submit */}
            <button type="submit" className="search-submit-btn">Search</button>
          </form>
        </div>

        {/* ── Mobile menu ── */}
        {mobileOpen && (
          <div className="mobile-menu-panel">
            {isAuthenticated && (
              <>
                <div className="mobile-user-card">
                  <div className="avatar-circle">{initials}</div>
                  <div style={{ minWidth: 0 }}>
                    <p style={{ fontSize: "0.85rem", fontWeight: 600, color: "#fff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{user?.name}</p>
                    <p style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.45)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{user?.email}</p>
                  </div>
                </div>
                <div className="mobile-divider" />
              </>
            )}
            <button onClick={toggleTheme} className="mobile-menu-link">
              {isDark ? <Sun size={18} /> : <Moon size={18} />}
              {isDark ? "Switch to Light" : "Switch to Dark"}
            </button>
            <Link to="/wishlist" onClick={() => setMobileOpen(false)} className="mobile-menu-link">
              <Heart size={18} /> Wishlist {wishlistItems.length > 0 && `(${wishlistItems.length})`}
            </Link>
            <Link to="/cart" onClick={() => setMobileOpen(false)} className="mobile-menu-link">
              <ShoppingCart size={18} /> Cart {cartCount > 0 && `(${cartCount})`}
            </Link>
            {isAuthenticated ? (
              <>
                <Link to="/dashboard" onClick={() => setMobileOpen(false)} className="mobile-menu-link">
                  <BarChart2 size={18} /> Dashboard
                </Link>
                <div className="mobile-divider" />
                <button onClick={handleSignOut} className="mobile-menu-link danger">
                  <LogOut size={18} /> Sign out
                </button>
              </>
            ) : (
              <>
                <div className="mobile-divider" />
                <Link to="/login" onClick={() => setMobileOpen(false)} className="mobile-menu-link">
                  <LogIn size={18} /> Sign In
                </Link>
                <Link to="/signup" onClick={() => setMobileOpen(false)} className="btn-solid" style={{ justifyContent: "center", borderRadius: "10px", padding: "10px 18px" }}>
                  <Zap size={16} strokeWidth={2.5} /> Join free
                </Link>
              </>
            )}
          </div>
        )}

        {/* ── Category bar ── */}
        {showCategoryBar && (
          <div className="category-bar">
            <div className="category-bar-inner">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat.label}
                  onClick={() => onCategoryChange(cat.label)}
                  className={`cat-pill ${activeCategory === cat.label ? "active" : ""}`}
                >
                  {cat.icon}
                  {cat.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>
    </>
  );
}
