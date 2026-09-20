# PriceIQ Frontend 🛍️

> **Next-Gen E-Commerce Storefront & Real-Time Operational Intelligence Dashboard**  
> Built with **React 18**, **TypeScript**, **Vite**, **Tailwind CSS**, and **Radix UI**.

---

## 🌟 Overview

The **PriceIQ Frontend** is a dual-interface web application serving two distinct user experiences:
1. **Consumer Storefront**: An e-commerce shopping experience featuring real-time dynamic pricing tickers, live scarcity indicators, flash deal alerts, session-aware recommendations, unified multi-vendor search (Amazon & Flipkart), and an embedded AI shopping assistant.
2. **Operations & Analytics Dashboard (`/dashboard`)**: An analytics console displaying live Server-Sent Events (SSE), real-time revenue KPIs, dynamic pricing A/B test results with automated statistical significance (Z-score), API latency tracking (P50/P99), recommendation model quality (NDCG@10/Hit Rate), predictive inventory stock-out timelines, and an audited fairness panel.

---

## 🚀 Key Features

### 🛒 Consumer Storefront
- **Live Dynamic Price Ticker**: Polls `/api/price` every 30 seconds with animated flash updates ("⚡ Price just updated!") whenever supply-demand shifts alter pricing.
- **Transparency Badges ("Why this price?")**: Displays clear reasons for dynamic adjustments (e.g. *Limited Stock*, *High Demand*, *Competitor Match*).
- **Session-Aware Recommendations**: Integrates `RecommendationRow` powered by deep learning (`GRU4Rec`), item-item co-views, and category-affinity scoring.
- **Unified Multi-Vendor Search**: Search across live Amazon and Flipkart products with real-time feedback, category chips, and price filters.
- **Interactive Cart & Checkout**: Dynamic cross-sell modules and non-blocking telemetry tracking for abandoned cart vs. completed orders.
- **AI Shopping Assistant**: Floating conversational concierge powered by Groq LLMs providing product recommendations and answering customer inquiries.

### 📊 Real-Time Operations Dashboard (`/dashboard`)
- **Live Event Firehose**: Direct connection to `/api/events/live` via SSE streaming live page views, cart adds, price changes, and checkouts.
- **A/B Testing Analysis**: Compares Variant A (Control - Rule-based) vs. Variant B (Treatment - Dynamic Pricing & GRU4Rec) with conversion rates, AOV, and two-proportion Z-score significance indicators.
- **24-Hour Price History**: Multi-line Recharts time-series graph tracking pricing across Amazon, Flipkart, and local catalogue products.
- **Predictive Inventory Health**: Table forecasting daily item purchase velocity and estimating exact days until stock depletion.
- **Algorithmic Fairness Audit**: Displays approved behavioral factors vs. prohibited demographic factors, verifying unbiased dynamic pricing.

---

## 🗂️ Project Directory Structure

```
frontend/src/
├── api/
│   └── index.ts               # Strongly typed API client & SSE EventSource listener
├── components/
│   ├── ChatAssistant.tsx      # Floating AI shopping concierge
│   ├── CountdownBadge.tsx     # Urgency and flash-deal timers
│   ├── Footer.tsx             # Responsive global footer
│   ├── Navbar.tsx             # Navigation bar with live search & cart counter
│   ├── PriceBadge.tsx         # Color-coded pricing rationale tags
│   ├── ProductCard.tsx        # Standardized product card with dynamic price display
│   ├── ProtectedRoute.tsx     # Auth wrapper for secured routes
│   ├── RecommendationRow.tsx  # Horizontal scroll recommendation carousel
│   ├── ScrollToTop.tsx        # Route transition scroll reset
│   └── ui/                    # Reusable shadcn/Radix UI primitive components
├── contexts/
│   ├── AuthContext.tsx        # Authentication state (JWT & Google OAuth)
│   └── ProductContext.tsx     # Marketplace & catalogue data provider
├── hooks/
│   ├── use-mobile.tsx         # Responsive breakpoint listener
│   └── use-toast.ts           # Toast notifications system
├── pages/
│   ├── Dashboard.tsx          # Real-time analytics, A/B testing & fairness dashboard
│   ├── Home.tsx               # Homepage with categories, hero banner & trending feeds
│   ├── ProductDetail.tsx      # Deep dive view with dynamic price poller & recommendations
│   ├── SearchResults.tsx      # Multi-source search listing with sorting & filters
│   ├── Cart.tsx               # Cart review, cross-sell recommendations & checkout
│   ├── Wishlist.tsx           # Saved items collection
│   ├── Login.tsx / Signup.tsx # User authentication forms
│   └── NotFound.tsx           # 404 handler
├── App.tsx                    # Route definitions & React Query client setup
└── main.tsx                   # Application entrypoint
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the `frontend` root directory:

```env
# URL pointing to the PriceIQ backend API
VITE_API_URL=http://localhost:5000
```

*For production deployments (e.g. Vercel), set `VITE_API_URL` to your production backend URL (e.g. `https://your-backend.onrender.com`).*

---

## 🛠️ Scripts & Commands

| Command | Description |
|---|---|
| `npm run dev` | Starts the Vite development server on `http://localhost:5173` |
| `npm run build` | Compiles and type-checks the application for production |
| `npm run preview` | Locally serves the optimized production build |
| `npm run lint` | Runs ESLint to check for code quality and type safety |
| `npm test` | Runs unit and component test suites with Vitest |
| `npx playwright test` | Executes end-to-end browser tests |

---

## 🧪 Testing

- **Unit & Integration Tests**: Executed via [Vitest](https://vitest.dev/) and `@testing-library/react`.
  ```bash
  npm test
  ```
- **End-to-End Tests**: Automated browser testing via [Playwright](https://playwright.dev/).
  ```bash
  npx playwright test
  ```

---

## 🎨 Design System & Theme
- **Color Palette**: Dark-mode inspired high-contrast palette with vibrant accents (emerald greens for conversion/wins, amber for stock scarcity, and indigo/cyan for brand identity).
- **Typography**: Clean, readable sans-serif font stack optimized for rapid tabular scanning of live metrics and prices.
- **Animations**: Subtle micro-interactions, CSS pulse rings for live indicators, and glowing highlights for real-time price updates.
