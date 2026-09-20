import {
  ArrowDownRight,
  ArrowRight,
  BarChart3,
  Boxes,
  Check,
  ChevronRight,
  CircleDot,
  Code2,
  Database,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Network,
  Radar,
  ScanLine,
  ServerCog,
  ShieldCheck,
  Sparkles,
  Waypoints,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { useMemo, useState, type ReactNode } from "react";
import LandingNavbar from "@/components/landing/LandingNavbar";
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
const formatMilliseconds = (value: number) => `${value.toFixed(value < 1 ? 2 : 2).replace(/0+$/, "").replace(/\.$/, "")} ms`;
const formatIndianCurrency = (value: number) => new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
}).format(value);

function SectionHeading({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="landing-section-heading" data-reveal>
      <p className="landing-eyebrow"><span />{eyebrow}</p>
      <h2>{title}</h2>
      {children && <div className="landing-section-heading__copy">{children}</div>}
    </div>
  );
}

function ProductSignalVisual() {
  return (
    <div className="hero-console" aria-label="Worked pricing decision example">
      <div className="hero-console__bar">
        <span><i className="status-dot" /> decision / pricing</span>
        <span className="hero-console__status">RULE PATH</span>
      </div>
      <div className="hero-console__content">
        <div className="hero-console__product">
          <div className="product-shape" aria-hidden="true"><span /><i /></div>
          <div>
            <p>{WORKED_PRICE.product}</p>
            <small>worked example — not live</small>
          </div>
          <strong>{formatIndianCurrency(WORKED_PRICE.recommendedPrice)}</strong>
        </div>
        <div className="hero-console__signal-grid">
          <SignalCell label="Inventory" value={`${WORKED_PRICE.inventory} units`} state="low" />
          <SignalCell label="Recent views" value={`${WORKED_PRICE.recentViews}`} state="active" />
          <SignalCell label="Market reference" value={formatIndianCurrency(WORKED_PRICE.modeledReference)} state="modeled" />
          <SignalCell label="Rule selected" value="+22%" state="rule" />
        </div>
        <div className="hero-console__decision">
          <span>Price recommendation</span>
          <div className="price-line"><strong>{formatIndianCurrency(WORKED_PRICE.recommendedPrice)}</strong><i>bounded</i></div>
          <p>{WORKED_PRICE.rationale}</p>
        </div>
      </div>
      <div className="hero-console__footer"><span>audit-ready input trail</span><span>modelled reference · not live</span></div>
    </div>
  );
}

function SignalCell({ label, value, state }: { label: string; value: string; state: string }) {
  return (
    <div className={`signal-cell signal-cell--${state}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <i aria-hidden="true" />
    </div>
  );
}

function ArchitectureNode({ icon: Icon, label, detail }: { icon: LucideIcon; label: string; detail: string }) {
  return <div className="architecture-node"><Icon aria-hidden="true" size={18} /><strong>{label}</strong><span>{detail}</span></div>;
}

function BenchmarkTable({ rows }: { rows: BenchmarkRow[] }) {
  return (
    <div className="benchmark-table-wrap" tabIndex={0} aria-label="Recommendation benchmark results">
      <table className="benchmark-table">
        <thead>
          <tr><th>Strategy</th><th>Hit@10</th><th>NDCG@10</th><th>Coverage</th><th>Avg latency</th><th>P95 latency</th></tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <th scope="row"><span className={`strategy-dot strategy-dot--${row.key}`} />{row.label}</th>
              <td>{formatPercent(row.hitAt10)}</td>
              <td>{formatPercent(row.ndcgAt10)}</td>
              <td>{formatPercent(row.coverage)}</td>
              <td>{formatMilliseconds(row.averageLatencyMs)}</td>
              <td>{formatMilliseconds(row.p95LatencyMs)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Landing() {
  const [previewTab, setPreviewTab] = useState<PreviewTab>("pricing");
  const [inventoryMode, setInventoryMode] = useState<"low" | "steady">("low");
  const [demandMode, setDemandMode] = useState<"elevated" | "normal">("elevated");
  const signals = useLandingSignals();
  useReveal();

  const demo = useMemo(() => {
    const selected = inventoryMode === "low" && demandMode === "elevated";
    const multiplier = selected ? 1 + PRICING_RULES.lowStockDemandAdjustment : 1;
    return {
      selected,
      price: Math.round(PRICING_RULES.basePrice * multiplier),
      rule: selected ? "low stock + elevated demand" : "control price",
    };
  }, [demandMode, inventoryMode]);

  const gru = signals.benchmark.find((row) => row.key === "gru4rec") ?? signals.benchmark[0];
  const statusCopy = signals.isLoading
    ? "Checking local services…"
    : signals.source === "live"
      ? `Live benchmark response · ${signals.updatedAt}`
      : `Committed fallback · ${signals.updatedAt}`;

  return (
    <div className="landing-page" id="top">
      <a className="landing-skip-link" href="#main-content">Skip to content</a>
      <LandingNavbar />

      <main id="main-content">
        <section className="landing-hero" aria-labelledby="landing-title">
          <div className="landing-shell landing-hero__grid">
            <div className="landing-hero__copy" data-reveal>
              <p className="landing-eyebrow"><span />COMMERCE INTELLIGENCE</p>
              <h1 id="landing-title">Price Smarter.<br />Personalize Faster.<br /><em>Sell Better.</em></h1>
              <p className="landing-lede">PriceIQ records shopper interactions, applies auditable price rules, and serves session-aware recommendations alongside cached marketplace catalog context.</p>
              <div className="landing-hero__actions">
                <a className="landing-button landing-button--primary" href="#demo">Explore the system <ArrowRight size={17} aria-hidden="true" /></a>
                <a className="landing-text-link" href="#benchmarks">See benchmarks <ArrowDownRight size={16} aria-hidden="true" /></a>
              </div>
              <p className="landing-trust"><Check size={15} aria-hidden="true" /> Interaction tracking <span /> Rule-based pricing <span /> Session-aware recommendations</p>
            </div>
            <div className="landing-hero__visual" data-reveal><ProductSignalVisual /></div>
          </div>
          <div className="landing-shell landing-hero__datum" aria-label="System capabilities">
            <span>EVENT CAPTURE</span><i /><span>PRICE RULES</span><i /><span>SESSION LADDER</span><i /><span>BENCHMARK ENDPOINT</span>
          </div>
        </section>

        <section className="landing-section landing-problem" aria-labelledby="problem-title">
          <div className="landing-shell landing-problem__grid">
            <div><SectionHeading eyebrow="THE PROBLEM" title="E-commerce moves faster than static pricing." /></div>
            <div className="landing-problem__copy" data-reveal>
              <p>Static catalog settings cannot express what changed in a shopper session, a product’s inventory, or a price rule’s rationale. Teams need a smaller, inspectable decision loop.</p>
              <div className="landing-problem__cards">
                <article><ScanLine aria-hidden="true" /><h3>Signals fragment</h3><p>Views, cart actions, inventory, and catalog context arrive through different paths.</p></article>
                <article><Radar aria-hidden="true" /><h3>Rules become opaque</h3><p>Price changes should expose their inputs and limits instead of appearing as a black box.</p></article>
                <article><BarChart3 aria-hidden="true" /><h3>Models need context</h3><p>A recommendation strategy should degrade gracefully when a session is still sparse.</p></article>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-section landing-solution" id="product" aria-labelledby="solution-title">
          <div className="landing-shell">
            <SectionHeading eyebrow="THE SOLUTION" title="One intelligence layer. Clearer decisions." >
              <p>A deliberate boundary between observed inputs, deterministic pricing rules, and a session recommendation service.</p>
            </SectionHeading>
            <div className="solution-map" data-reveal>
              <div className="solution-map__column"><p>INPUTS</p><div>Shopper interaction events</div><div>Cached marketplace catalog</div><div>Product inventory</div></div>
              <div className="solution-map__connector" aria-hidden="true"><span /><i /><span /></div>
              <div className="solution-map__center"><div className="solution-map__logo"><span className="landing-brand__mark"><i /><i /><i /></span> PriceIQ</div><strong>Decision layer</strong><p>Rule evaluation<br />Session strategy selection</p><small>explainable paths</small></div>
              <div className="solution-map__connector" aria-hidden="true"><span /><i /><span /></div>
              <div className="solution-map__column solution-map__column--outputs"><p>OUTPUTS</p><div>Bounded price decision</div><div>Product recommendations</div><div>Dashboard-ready events</div></div>
            </div>
            <div className="landing-solution__note" data-reveal><ShieldCheck size={17} aria-hidden="true" /><p>“Competitive” is intentionally not a promise here: the pricing engine’s competitor reference is modeled, not a live competitor feed.</p></div>
          </div>
        </section>

        <section className="landing-section landing-process" id="how-it-works" aria-labelledby="how-title">
          <div className="landing-shell">
            <SectionHeading eyebrow="HOW IT WORKS" title="Small loops. Visible logic." />
            <ol className="process-grid">
              <li data-reveal><span>01</span><div><CircleDot aria-hidden="true" /><h3>Observe</h3><p>Capture view, cart, wishlist, and purchase events through the public tracking route.</p></div></li>
              <li data-reveal><span>02</span><div><Layers3 aria-hidden="true" /><h3>Understand</h3><p>Read session history, inventory fields, and cached product catalog context.</p></div></li>
              <li data-reveal><span>03</span><div><Waypoints aria-hidden="true" /><h3>Decide</h3><p>Apply bounded pricing rules and select a recommendation strategy by session depth.</p></div></li>
              <li data-reveal><span>04</span><div><Gauge aria-hidden="true" /><h3>Measure</h3><p>Inspect dashboard endpoints and compare recommender strategies on one held-out set.</p></div></li>
            </ol>
          </div>
        </section>

        <section className="landing-section landing-capabilities" aria-labelledby="capabilities-title">
          <div className="landing-shell">
            <SectionHeading eyebrow="CAPABILITIES" title="Purpose-built for inspectable commerce workflows." />
            <div className="capabilities-grid">
              <article className="capability-card capability-card--large" data-reveal><div className="capability-card__top"><span className="icon-box"><Zap /></span><span className="card-index">01</span></div><h3>Rule-based price decisions</h3><p>Inventory and engagement thresholds lead to a bounded output. The modeled market reference can lower a treatment price, but it is never labeled as live competitor pricing.</p><div className="mini-rule"><span>stock ≤ 3 + views ≥ 20</span><strong>+22%</strong></div></article>
              <article className="capability-card" data-reveal><div className="capability-card__top"><span className="icon-box"><Sparkles /></span><span className="card-index">02</span></div><h3>Session-aware recommendations</h3><p>A practical fallback ladder: popularity for empty sessions, content similarity for short ones, GRU4Rec from three interactions.</p><div className="ladder"><i /><i /><i /></div></article>
              <article className="capability-card" data-reveal><div className="capability-card__top"><span className="icon-box"><GitBranch /></span><span className="card-index">03</span></div><h3>Experiment routing</h3><p>A deterministic control/treatment assignment is available in the API. This page does not claim an experiment result.</p><div className="split-pill"><span>control</span><i /><span>treatment</span></div></article>
              <article className="capability-card" data-reveal><div className="capability-card__top"><span className="icon-box"><Network /></span><span className="card-index">04</span></div><h3>Event processing path</h3><p>When Redis and the stream worker are configured, interaction tracking may also enqueue a Redis Stream for asynchronous processing.</p><div className="stream-line"><i /><i /><i /><i /></div></article>
            </div>
          </div>
        </section>

        <section className="landing-section landing-benchmarks" id="benchmarks" aria-labelledby="benchmark-title">
          <div className="landing-shell">
            <div className="benchmark-heading"><SectionHeading eyebrow="ML BENCHMARK" title="Measured strategies. No magic score." ><p>Results below are the service’s reported held-out evaluation. They are shown side by side; no single strategy is presented as universally best.</p></SectionHeading><div className={`data-status ${signals.error ? "data-status--fallback" : ""}`} role="status" aria-live="polite"><i />{statusCopy}</div></div>
            <div data-reveal><BenchmarkTable rows={signals.benchmark} /></div>
            <div className="benchmark-notes" data-reveal><p><strong>{signals.testSamples.toLocaleString("en-IN")}</strong> held-out samples reported by the evaluation endpoint.</p><p>Coverage measures the catalog share surfaced by that strategy, not relevance.</p><p>{signals.source === "live" ? "The ML service responded during this page load." : "The ML service did not provide a usable response, so this table uses the committed fallback snapshot."}</p></div>
          </div>
        </section>

        <section className="landing-section landing-preview" aria-labelledby="preview-title">
          <div className="landing-shell landing-preview__grid">
            <SectionHeading eyebrow="PRODUCT PREVIEW" title="Interfaces that reveal the path behind the output." >
              <p>Illustrative panels below are sample data and endpoint vocabulary, not a live merchant account.</p>
            </SectionHeading>
            <div className="preview-console" data-reveal>
              <div className="preview-tabs" role="tablist" aria-label="Product preview mode">
                {(Object.keys(PREVIEWS) as PreviewTab[]).map((tab) => <button key={tab} role="tab" type="button" aria-selected={previewTab === tab} aria-controls={`preview-${tab}`} id={`tab-${tab}`} onClick={() => setPreviewTab(tab)}>{tab === "pricing" ? "Pricing" : tab === "recommendations" ? "Recommendations" : "Observability"}</button>)}
              </div>
              <div className="preview-body" id={`preview-${previewTab}`} role="tabpanel" aria-labelledby={`tab-${previewTab}`}>
                <div className="preview-body__head"><div><p>{PREVIEWS[previewTab].eyebrow}</p><h3>{PREVIEWS[previewTab].title}</h3></div><span className="sample-chip">SAMPLE DATA</span></div>
                <div className="preview-rows">{PREVIEWS[previewTab].rows.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong><i aria-hidden="true" /></div>)}</div>
                <div className="preview-spark" aria-hidden="true"><span /><span /><span /><span /><span /><span /><span /><span /></div>
              </div>
            </div>
          </div>
        </section>

        <section className="landing-section landing-metrics" aria-labelledby="metrics-title">
          <div className="landing-shell">
            <SectionHeading eyebrow="SERVICE SNAPSHOT" title="Concrete system signals, clearly sourced." />
            <div className="metric-grid" data-reveal>
              <article><span>CATALOG ITEMS</span><strong>{signals.catalogItems.toLocaleString("en-IN")}</strong><p>from <code>/api/products</code></p></article>
              <article><span>MODEL VOCABULARY</span><strong>{signals.modelVocabulary.toLocaleString("en-IN")}</strong><p>from ML <code>/health</code></p></article>
              <article><span>GRU AVG LATENCY</span><strong>{formatMilliseconds(gru.averageLatencyMs)}</strong><p>from <code>/evaluate</code></p></article>
              <article><span>FALLBACK SNAPSHOT</span><strong>{SNAPSHOT.label}</strong><p>committed when services are unavailable</p></article>
            </div>
          </div>
        </section>

        <section className="landing-section landing-architecture" id="architecture" aria-labelledby="architecture-title">
          <div className="landing-shell">
            <SectionHeading eyebrow="ARCHITECTURE" title="A local, legible service boundary." >
              <p>The frontend, API, stateful stores, and recommender service are separate so each path can be inspected without conflating it with a promise of uptime or freshness.</p>
            </SectionHeading>
            <div className="architecture-map" data-reveal>
              <ArchitectureNode icon={Code2} label="React client" detail="Vite + TypeScript" />
              <ChevronRight aria-hidden="true" className="architecture-arrow" />
              <ArchitectureNode icon={ServerCog} label="Express API" detail="pricing + events + catalog" />
              <ChevronRight aria-hidden="true" className="architecture-arrow" />
              <div className="architecture-storage"><ArchitectureNode icon={Database} label="MongoDB" detail="products + events" /><ArchitectureNode icon={Boxes} label="Redis" detail="cache + streams when configured" /></div>
              <ChevronRight aria-hidden="true" className="architecture-arrow" />
              <ArchitectureNode icon={LineChart} label="ML service" detail="FastAPI + GRU4Rec" />
            </div>
          </div>
        </section>

        <section className="landing-section landing-transparency" aria-labelledby="transparency-title">
          <div className="landing-shell landing-transparency__grid">
            <SectionHeading eyebrow="TRANSPARENCY" title="The useful details stay in view." />
            <div className="transparency-list" data-reveal>
              <article><h3>Why this price?</h3><p>Show the triggering rule, observed inventory and view thresholds, reference type, plus the MRP bounds.</p></article>
              <article><h3>Which model?</h3><p>The recommendation endpoint follows a visible session-depth strategy ladder. A model confidence score is not returned, so this page does not invent one.</p></article>
              <article><h3>What are the data limits?</h3><p>Marketplace responses can be cached, the competitor reference is modeled, and training-data provenance is not certified by this demo repository.</p></article>
            </div>
          </div>
          <div className="landing-shell method-notes" data-reveal><p>Method notes</p>{METHOD_NOTES.map((note) => <div key={note}><Check size={16} aria-hidden="true" />{note}</div>)}</div>
        </section>

        <section className="landing-section landing-demo" id="demo" aria-labelledby="demo-title">
          <div className="landing-shell landing-demo__grid">
            <div><SectionHeading eyebrow="INTERACTIVE RULE WALKTHROUGH" title="Change the inputs. See the selected rule." ><p>A worked calculation based on the committed pricing-engine thresholds. This is a demonstrator, not a request to the price API.</p></SectionHeading>
              <div className="demo-controls" data-reveal><fieldset><legend>Inventory</legend><button type="button" className={inventoryMode === "low" ? "is-selected" : ""} onClick={() => setInventoryMode("low")}>Low · {PRICING_RULES.lowStockThreshold} or fewer</button><button type="button" className={inventoryMode === "steady" ? "is-selected" : ""} onClick={() => setInventoryMode("steady")}>Steady</button></fieldset><fieldset><legend>Recent views</legend><button type="button" className={demandMode === "elevated" ? "is-selected" : ""} onClick={() => setDemandMode("elevated")}>Elevated · {PRICING_RULES.demandThreshold}+</button><button type="button" className={demandMode === "normal" ? "is-selected" : ""} onClick={() => setDemandMode("normal")}>Normal</button></fieldset></div>
            </div>
            <div className="demo-output" data-reveal><p>WORKED OUTPUT — NOT LIVE</p><span className={`demo-output__state ${demo.selected ? "is-active" : ""}`}><i /> {demo.rule}</span><strong>{formatIndianCurrency(demo.price)}</strong><div><span>base {formatIndianCurrency(PRICING_RULES.basePrice)}</span><i /><span>{demo.selected ? "+22% combined rule" : "control"}</span></div><small>Price bounds in implementation: {formatPercent(PRICING_RULES.minimumOfMrp)}–{formatPercent(PRICING_RULES.maximumOfMrp)} of MRP.</small></div>
          </div>
        </section>

        <section className="landing-section landing-credibility" aria-labelledby="credibility-title">
          <div className="landing-shell"><SectionHeading eyebrow="BUILT WITH" title="Open, familiar building blocks." /><div className="tech-list" data-reveal>{TECH_STACK.map((technology) => <span key={technology}>{technology}</span>)}</div></div>
        </section>

        <section className="landing-final-cta" aria-labelledby="cta-title">
          <div className="landing-shell landing-final-cta__box" data-reveal><p className="landing-eyebrow"><span />DEMO SYSTEM</p><h2 id="cta-title">Inspect the workflow,<br /><em>then make it yours.</em></h2><p>Explore the storefront, trace the service boundaries, and see the source behind each material statement on this page.</p><div><Link className="landing-button landing-button--primary" to="/shop">Open platform <ArrowRight size={17} aria-hidden="true" /></Link><a className="landing-text-link" href={REPOSITORY_URL} target="_blank" rel="noreferrer">View repository <ArrowDownRight size={16} aria-hidden="true" /></a></div></div>
        </section>
      </main>

      <footer className="landing-footer">
        <div className="landing-shell landing-footer__top"><a className="landing-brand" href="#top"><span className="landing-brand__mark" aria-hidden="true"><i /><i /><i /></span>PriceIQ</a><p>Commerce intelligence demo with documented boundaries.</p><a href={REPOSITORY_URL} target="_blank" rel="noreferrer">GitHub <ArrowDownRight size={15} aria-hidden="true" /></a></div>
        <div className="landing-shell landing-footer__bottom"><span>Demo system · {SNAPSHOT.label}</span><span>Built for a hackathon prototype</span></div>
      </footer>
    </div>
  );
}
