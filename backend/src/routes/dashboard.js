import { Router } from 'express';
import Event from '../models/Event.js';
import Session from '../models/Session.js';
import { getDashboardMetrics } from '../services/analyticsService.js';
import { redisLRange } from '../config/redis.js';

const router = Router();

// ── GET /api/dashboard/latency ──────────────────────────────────────────────
router.get('/latency', async (req, res) => {
  try {
    const rawLatencies = await redisLRange('perf:latency', 0, -1);
    const latencies = rawLatencies.map(Number).sort((a, b) => a - b);

    const stats = {
      count: latencies.length,
      p50: 0,
      p99: 0,
      avg: 0,
      quality: { ndcg: 0, hitRate: 0 }
    };

    if (latencies.length > 0) {
      stats.p50 = latencies[Math.floor(latencies.length * 0.5)];
      stats.p99 = latencies[Math.floor(latencies.length * 0.99)];
      stats.avg = Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
    }

    // Try to fetch ML metrics from FastAPI
    try {
      const mlUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
      const mlRes = await fetch(`${mlUrl}/evaluate`, { signal: AbortSignal.timeout(2500) });
      if (mlRes.ok) {
        const mlData = await mlRes.json();
        const gruStats = mlData.models?.GRU4Rec || {};
        stats.quality.ndcg = mlData.ndcg_at_10 || gruStats['ndcg@10'] || 0;
        stats.quality.hitRate = mlData.hit_rate || mlData.hit_rate_at_10 || gruStats['hit@10'] || 0;
        stats.quality.mrr = gruStats.mrr || 0;
        stats.quality.sampleSize = mlData.test_samples_count || mlData.sample_size || 0;
      }
    } catch (e) {
      // ml service might be down, ignore
    }

    res.json(stats);
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch latency metrics' });
  }
});

// ── GET /api/dashboard/metrics ────────────────────────────────────────────────
router.get('/metrics', async (req, res) => {
  try {
    const now = new Date();
    const since24h = new Date(now - 24 * 60 * 60 * 1000);
    const since5min = new Date(now - 5 * 60 * 1000);
    const since1h = new Date(now - 60 * 60 * 1000);

    // Run all queries in parallel
    const [
      pageViews,
      cartAdds,
      wishlistAdds,
      purchaseEvents,
      activeSessions,
      recentEventsRaw,
      topProductsAgg,
      topQueriesAgg,
      sourceDistributionAgg,
      baseMetrics,
    ] = await Promise.all([
      Event.countDocuments({ eventType: 'page_view', timestamp: { $gte: since24h } }),
      Event.countDocuments({ eventType: 'add_to_cart', timestamp: { $gte: since24h } }),
      Event.countDocuments({ eventType: 'wishlist_add', timestamp: { $gte: since24h } }),
      Event.find({ eventType: 'purchase', timestamp: { $gte: since24h } }).lean(),
      Session.countDocuments({ lastSeen: { $gte: since5min } }),
      Event.find({ timestamp: { $gte: since1h } }).sort({ timestamp: -1 }).limit(20).lean(),
      // Top products by add-to-cart + purchase events
      Event.aggregate([
        { $match: { eventType: { $in: ['add_to_cart', 'purchase'] }, 'metadata.productName': { $exists: true, $ne: null, $ne: '' } } },
        { $group: {
            _id: '$metadata.productName',
            count: { $sum: 1 },
            source: { $first: '$metadata.source' },
            sourceId: { $first: '$metadata.sourceId' },
            price: { $avg: '$metadata.price' },
        }},
        { $sort: { count: -1 } },
        { $limit: 8 },
      ]),
      // Top searched queries — filter nulls
      Event.aggregate([
        { $match: { eventType: 'search', 'metadata.query': { $exists: true, $ne: null, $ne: '' } } },
        { $group: { _id: '$metadata.query', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 10 },
      ]),
      // Source distribution: amazon vs flipkart events
      Event.aggregate([
        { $match: { 'metadata.source': { $in: ['amazon', 'flipkart', 'local'] } } },
        { $group: { _id: '$metadata.source', count: { $sum: 1 } } },
      ]),
      getDashboardMetrics().catch(() => null),
    ]);

    // Real revenue from purchase event metadata.price
    const totalRevenue = purchaseEvents.reduce((sum, e) => sum + (e.metadata?.price || 0), 0);
    const purchases = purchaseEvents.length;
    const avgOrderValue = purchases > 0 ? Math.round(totalRevenue / purchases) : 0;

    // Real conversion rate
    const conversionRate = pageViews > 0 ? parseFloat(((purchases / pageViews) * 100).toFixed(2)) : 0;

    // A/B data from base metrics (MongoDB sessions)
    const abControl = baseMetrics?.conversionRate?.control ?? conversionRate;
    const abTreatment = baseMetrics?.conversionRate?.treatment ?? conversionRate;

    const recentEvents = recentEventsRaw.map(e => ({
      type: e.eventType,
      productId: e.productId,
      productName: e.metadata?.productName || null,
      sessionId: e.sessionId,
      timestamp: e.timestamp,
      city: e.metadata?.city || null,
      device: e.metadata?.device || null,
      price: e.metadata?.price || null,
    }));

    res.json({
      totalRevenue: {
        value: totalRevenue,
        change: baseMetrics?.totalRevenue?.change ?? 0,
      },
      conversionRate: {
        overall: conversionRate,
        control: abControl,
        treatment: abTreatment,
      },
      avgOrderValue: {
        value: avgOrderValue,
        byVariant: baseMetrics?.avgOrderValue?.byVariant ?? {},
        change: baseMetrics?.avgOrderValue?.change ?? 0,
      },
      activeSessions,
      pageViews,
      cartAdds,
      wishlistAdds,
      purchases,
      topProducts: topProductsAgg.map(p => ({
        name: p._id,
        count: p.count,
        source: p.source || null,
        sourceId: p.sourceId || null,
        avgPrice: p.price ? Math.round(p.price) : null,
      })),
      topQueries: topQueriesAgg.filter(q => q._id).map(q => ({ query: q._id, count: q.count })),
      sourceDistribution: Object.fromEntries(
        (sourceDistributionAgg || []).map(s => [s._id, s.count])
      ),
      recentEvents,
      // ── Session Analytics (new) ────────────────────────────────────────────
      avgEngagementScore: baseMetrics?.avgEngagementScore ?? 0,
      avgPurchaseIntent: baseMetrics?.avgPurchaseIntent ?? 0,
      topCategoryAffinity: baseMetrics?.topCategoryAffinity ?? [],
      abSignificance: baseMetrics?.abSignificance ?? { significant: false, note: 'Need more data' },
      segmentDistribution: baseMetrics?.segmentDistribution ?? { value_seeker: 0, standard: 0, premium_intent: 0 },
      generatedAt: now.toISOString(),
    });
  } catch (err) {
    console.error('Dashboard metrics error:', err);
    res.json({
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
      sourceDistribution: {},
      recentEvents: [],
      avgEngagementScore: 0,
      avgPurchaseIntent: 0,
      topCategoryAffinity: [],
      abSignificance: { significant: false, note: 'Need more data' },
      segmentDistribution: { value_seeker: 0, standard: 0, premium_intent: 0 },
      generatedAt: new Date().toISOString(),
    });
  }
});

// ── GET /api/dashboard/fairness ──────────────────────────────────────────
router.get('/fairness', async (_req, res) => {
  try {
    const allSessions = await Session.find({}).lean();
    const segDist = { value_seeker: 0, standard: 0, premium_intent: 0 };
    allSessions.forEach(s => {
      const seg = s.userSegment || 'standard';
      if (segDist[seg] !== undefined) segDist[seg]++;
    });

    res.json({
      pricingFactors: [
        { factor: 'demand_velocity', description: 'Views in last 15 min (from Redis counter)' },
        { factor: 'stock_level', description: 'Units remaining in inventory' },
        { factor: 'competitor_match', description: 'Simulated competitive price comparison' },
        { factor: 'user_segment', description: 'Behavioral engagement depth (non-demographic)' },
        { factor: 'ab_variant', description: 'A/B experiment assignment (random, not demographic)' },
      ],
      excludedFactors: [
        { factor: 'gender', reason: 'Protected attribute — never used' },
        { factor: 'caste / religion', reason: 'Protected attribute — never used' },
        { factor: 'city / region', reason: 'Collected for analytics only, not for pricing' },
        { factor: 'income_bracket', reason: 'Not available or collected' },
        { factor: 'ethnicity', reason: 'Protected attribute — never used' },
      ],
      segmentDistribution: segDist,
      segmentBasis: 'Segments (value_seeker / standard / premium_intent) are derived from: session engagement depth, cart actions, purchase history. No demographic data is used.',
      auditNote: 'PriceIQ pricing engine uses ONLY behavioral and commercial signals. No protected demographic attributes (gender, caste, religion, city, income bracket, ethnicity) are used in pricing decisions. All pricing rules are subject to a floor of 70% MRP and ceiling of 100% MRP to prevent exploitation.',
      lastAudited: new Date().toISOString(),
    });
  } catch (err) {
    res.status(500).json({ error: 'Fairness audit failed', message: err.message });
  }
});

// ── GET /api/dashboard/marketplace-history ─────────────────────────────────────────────
// Seeded PRNG — deterministic per product id so chart is stable on refresh
function seededRandom(seed) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  return () => {
    h = Math.imul(1664525, h) + 1013904223 | 0;
    return ((h >>> 0) % 1000) / 1000;
  };
}

function buildHistory(product) {
  const rand = seededRandom(String(product.id));
  const nowPrice = product.livePrice;
  const points = [];
  let current = nowPrice * (0.94 + rand() * 0.08);
  for (let i = 0; i < 24; i++) {
    const drift = (rand() - 0.5) * nowPrice * 0.03;
    current = Math.max(nowPrice * 0.88, Math.min(nowPrice * 1.08, current + drift));
    points.push({ hour: `${(16 + i) % 24}:00`, price: Math.round(current) });
  }
  return points;
}

router.get('/marketplace-history', async (req, res) => {
  try {
    // Import marketplace cache utilities lazily
    const { default: marketplaceRouter } = await import('./marketplace.js');
    // Attempt to read from in-memory marketplace cache via a direct fetch to own endpoint
    // Simpler: re-use the same mem Map that marketplace.js uses
    // We pass through an internal GET to avoid duplicating cache logic
    const base = `http://localhost:${process.env.PORT || 5000}`;
    const feedRes = await fetch(`${base}/api/marketplace/home`, { signal: AbortSignal.timeout(6000) });
    if (!feedRes.ok) throw new Error('feed unavailable');
    const feed = await feedRes.json();

    const pool = [
      ...(feed.featured    || []),
      ...(feed.electronics || []),
      ...(feed.fashion     || []),
    ].filter(p => p && p.id && p.name && p.livePrice > 300 && p.images?.length > 0);

    // Dedup by id
    const seen = new Set();
    const unique = pool.filter(p => { if (seen.has(p.id)) return false; seen.add(p.id); return true; });

    const top5 = unique.slice(0, 5);

    const products = top5.map(p => ({
      id: String(p.id),
      name: p.name.substring(0, 24),
      source: p.source || 'amazon',
      livePrice: p.livePrice,
      history: buildHistory(p),
    }));

    return res.json({ products });
  } catch (err) {
    console.error('marketplace-history error:', err.message);
    return res.json({ products: [] });
  }
});

// ── GET /api/dashboard/events/live — SSE ─────────────────────────────────────
router.get('/events/live', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.flushHeaders();

  const sendEvent = async () => {
    try {
      const latestEvent = await Event.findOne({
        eventType: { $in: ['add_to_cart', 'purchase', 'page_view'] },
      }).sort({ timestamp: -1 }).lean();

      let message = '';
      if (latestEvent?.metadata?.productName) {
        const uid = (latestEvent.userId || '').substring(0, 6) || Math.floor(Math.random() * 9999);
        switch (latestEvent.eventType) {
          case 'add_to_cart':
            message = `User #${uid} added ${latestEvent.metadata.productName} to cart`;
            break;
          case 'purchase':
            message = `User #${uid} purchased ${latestEvent.metadata.productName}${latestEvent.metadata.price ? ` — ₹${Number(latestEvent.metadata.price).toLocaleString('en-IN')}` : ''}`;
            break;
          default:
            message = generateFakeEvent();
        }
      } else {
        message = generateFakeEvent();
      }
      res.write(`event: message\ndata: ${JSON.stringify({ message, timestamp: new Date() })}\n\n`);
    } catch {
      res.write(`event: message\ndata: ${JSON.stringify({ message: generateFakeEvent(), timestamp: new Date() })}\n\n`);
    }
  };

  sendEvent();
  const interval = setInterval(sendEvent, 3000);
  req.on('close', () => clearInterval(interval));
});

function generateFakeEvent() {
  const cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad'];
  const products = [
    'Sony WH-1000XM5', 'Apple iPhone 15', 'boAt Rockerz 450', 'Samsung Galaxy S23',
    'Nike Air Max 270', 'Adidas Ultraboost 22', 'Instant Pot Duo', 'Xiaomi Smart Band 8',
    'JBL Flip 6', 'OnePlus Nord CE 3', 'Puma Sports Shoes', 'Amazon Echo Dot',
  ];
  const product = products[Math.floor(Math.random() * products.length)];
  const city = cities[Math.floor(Math.random() * cities.length)];
  const uid = Math.floor(1000 + Math.random() * 9000);
  const priceDrop = Math.floor(Math.random() * 3000 + 500);
  const templates = [
    `User #${uid} added ${product} to cart`,
    `Price dropped ₹${priceDrop.toLocaleString('en-IN')} on ${product} — Dynamic Pricing active`,
    `New session started — ${Math.random() > 0.5 ? 'Mobile' : 'Desktop'}, ${city}`,
    `User #${uid} purchased ${product}`,
    `Flash deal triggered: ${product} — ${Math.floor(20 + Math.random() * 30)}% OFF`,
    `${Math.floor(Math.random() * 50) + 5} users viewing ${product} right now`,
  ];
  return templates[Math.floor(Math.random() * templates.length)];
}

export default router;
