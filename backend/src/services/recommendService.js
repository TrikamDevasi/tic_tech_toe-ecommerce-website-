import Product from '../models/Product.js';
import Event from '../models/Event.js';
import Session from '../models/Session.js';
import { redisLRange, redisGet, redisSet } from '../config/redis.js';

// ── Time-of-day category preferences for cold-start ─────────────────────────
// ── Behavioral & Contextual Categories for cold-start (PS3 Task 9) ───────────
function getColdStartCategories(context = {}) {
  const { device, referral } = context;
  const hour = new Date().getHours();

  let categories = [];

  // 1. Time-of-day baseline
  if (hour >= 6 && hour < 10) categories = ['Books', 'Health', 'Sports'];
  else if (hour >= 10 && hour < 14) categories = ['Electronics', 'Fashion', 'Beauty'];
  else if (hour >= 14 && hour < 20) categories = ['Electronics', 'Home & Kitchen', 'Gaming'];
  else categories = ['Books', 'Home & Kitchen', 'Fashion'];

  // 2. Device signals (Mobile users favor quick consumables/lifestyle)
  if (device === 'Mobile') {
    categories = [...new Set(['Gaming', 'Fashion', ...categories])];
  }

  // 3. Referral signals (Tech blogs favor electronics)
  if (referral && (referral.includes('tech') || referral.includes('gizmodo'))) {
    categories = ['Electronics', 'Gaming', ...categories];
  }

  return [...new Set(categories)];
}

// ── Item-item collaborative filtering via co-view signals ────────────────────
async function getCoviewedProducts(productId) {
  // Try cache first (5 min TTL)
  const cacheKey = `coview:${productId}`;
  try {
    const cached = await redisGet(cacheKey);
    if (cached) return JSON.parse(cached);
  } catch {}

  try {
    // Find sessions that viewed this product
    const viewerSessions = await Event.distinct('sessionId', {
      eventType: 'page_view',
      productId: String(productId),
    });

    if (!viewerSessions.length) return [];

    // Find what else those sessions viewed
    const coViewEvents = await Event.find({
      sessionId: { $in: viewerSessions },
      eventType: 'page_view',
      productId: { $ne: null, $ne: String(productId) },
    }).lean();

    // Count co-occurrence frequencies
    const freq = {};
    coViewEvents.forEach(e => {
      const pid = parseInt(e.productId, 10);
      if (pid && !isNaN(pid)) freq[pid] = (freq[pid] || 0) + 1;
    });

    // Sort by frequency, return top 5 IDs
    const topCoviewed = Object.entries(freq)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([id]) => parseInt(id, 10));

    // Cache result for 5 minutes
    try { await redisSet(cacheKey, JSON.stringify(topCoviewed), 300); } catch {}

    return topCoviewed;
  } catch {
    return [];
  }
}

/**
 * Get 10 recommendations for a product/session.
 * Priority order:
 *  1) Co-viewed (collaborative filtering)
 *  2) Same category (content-based)
 *  3) Session recently-viewed history
 *  4) Trending by viewCount
 *  5) Contextual cold-start (time-of-day categories)
 *  6) Random fill
 */
export async function getRecommendations(sessionId, productId, context = {}, abVariant = 'control') {
  // A/B: treatment variant uses ML service (GRU4Rec)
  if (abVariant === 'treatment') {
    try {
      const sessionHistory = await redisLRange(`viewed:${sessionId}`, 0, 19);
      if (sessionHistory.length >= 2) {
        const mlRes = await fetch(
          `${process.env.ML_SERVICE_URL || 'http://localhost:8000'}/recommend/session`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_history: sessionHistory, top_k: 8 }),
            signal: AbortSignal.timeout(800)
          }
        );
        if (mlRes.ok) {
          const data = await mlRes.json();
          if (data.recommendations?.length > 0) {
            const recIds = data.recommendations;
            const numericIds = recIds.map(Number).filter(n => !isNaN(n));
            const products = await Product.find({
              $or: [
                { id: { $in: numericIds } },
                { id: { $in: recIds.map(String) } }
              ]
            }).limit(8);
            if (products.length > 0) {
              return products.map(p => ({ ...p.toObject(), _recReason: `ml-${data.model || 'gru4rec'}` }));
            }
          }
        }
      }
    } catch (mlErr) {
      console.warn('ML rec failed, falling back:', mlErr.message);
    }
  }

  const all = await Product.find({}).lean();
  const current = all.find((p) => p.id === productId);
  const exclude = new Set([productId]);
  const result = [];

  // ── 0. ML Service: Collaborative / Session recs ──────────────────────────
  try {
    const mlUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    const viewedRaw = (sessionId && sessionId !== 'anonymous') 
      ? await redisLRange(`viewed:${sessionId}`, 0, 9) 
      : [];
    
    // Canonical product IDs (numeric or string) from Redis session history
    const sessionHistory = viewedRaw.filter(Boolean);

    if (sessionHistory.length > 0) {
      const mlRes = await fetch(`${mlUrl}/recommend/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_history: sessionHistory, top_k: 10 }),
        signal: AbortSignal.timeout(1500)
      });
      if (mlRes.ok) {
        const { recommendations: recIds } = await mlRes.json();
        const recStrSet = new Set((recIds || []).map(String));
        const mlProducts = all.filter(p => recStrSet.has(String(p.id)) && !exclude.has(p.id));
        mlProducts.forEach(p => {
          result.push({ ...p, _recReason: 'ml-personalized' });
          exclude.add(p.id);
        });
      }
    } else if (productId) {
      // Fallback: recommend by product ID if no session history
      const mlRes = await fetch(`${mlUrl}/recommend/${productId}?top_k=10`, { signal: AbortSignal.timeout(1000) });
      if (mlRes.ok) {
        const { recommendations: recIds } = await mlRes.json();
        const recStrSet = new Set((recIds || []).map(String));
        const mlProducts = all.filter(p => recStrSet.has(String(p.id)) && !exclude.has(p.id));
        mlProducts.forEach(p => {
          result.push({ ...p, _recReason: 'ml-content' });
          exclude.add(p.id);
        });
      }
    }
  } catch (err) {
    console.warn('ML Service unreachable, falling back to rule-based recs');
  }

  // ── 1. Collaborative filtering: co-viewed products ──────────────────────
  if (productId) {
    const coviewedIds = await getCoviewedProducts(productId);
    const coviewedProducts = all.filter(p => coviewedIds.includes(p.id) && !exclude.has(p.id));
    coviewedProducts.forEach(p => {
      result.push({ ...p, _recReason: 'co-viewed' });
      exclude.add(p.id);
    });
  }

  // ── 2. Same category (content-based) ────────────────────────────────────
  if (current) {
    const sameCategory = all
      .filter((p) => p.category === current.category && !exclude.has(p.id))
      .sort((a, b) => b.viewCount - a.viewCount)
      .slice(0, 4);
    sameCategory.forEach((p) => {
      result.push({ ...p, _recReason: 'same-category' });
      exclude.add(p.id);
    });
  }

  // ── 3. Session recently-viewed history from Redis ────────────────────────
  if (sessionId && sessionId !== 'anonymous') {
    const viewedRaw = await redisLRange(`viewed:${sessionId}`, 0, 9);
    const viewedIds = viewedRaw.map(Number).filter((id) => !exclude.has(id));
    const viewedProducts = all.filter((p) => viewedIds.includes(p.id));
    viewedProducts.forEach((p) => {
      if (result.length < 10 && !exclude.has(p.id)) {
        result.push({ ...p, _recReason: 'session-history' });
        exclude.add(p.id);
      }
    });
  }

  // ── 4. Trending (top by viewCount) ──────────────────────────────────────
  const trending = all
    .filter((p) => !exclude.has(p.id))
    .sort((a, b) => b.viewCount - a.viewCount)
    .slice(0, 3);
  trending.forEach((p) => {
    if (result.length < 10) {
      result.push({ ...p, _recReason: 'trending' });
      exclude.add(p.id);
    }
  });

  // ── 5. Contextual cold-start (behavioral signals - PS3 Task 9) ─────────
  if (result.length < 6) {
    let coldStartCtx = context;
    if (sessionId && sessionId !== 'anonymous' && !coldStartCtx.device) {
      const session = await Session.findOne({ sessionId }).lean();
      if (session) {
        coldStartCtx = { 
          device: session.deviceType, 
          referral: session.referralSource,
          ...coldStartCtx
        };
      }
    }

    const timeCategories = coldStartCtx.timeCategories || getColdStartCategories(coldStartCtx);
    const contextualProducts = all
      .filter(p => !exclude.has(p.id) && timeCategories.includes(p.category))
      .sort((a, b) => b.viewCount - a.viewCount)
      .slice(0, 4);
    contextualProducts.forEach(p => {
      if (result.length < 10) {
        result.push({ ...p, _recReason: 'contextual' });
        exclude.add(p.id);
      }
    });
  }

  // ── 6. Random fill ──────────────────────────────────────────────────────
  const remaining = all.filter((p) => !exclude.has(p.id)).sort(() => Math.random() - 0.5);
  for (const p of remaining) {
    if (result.length >= 10) break;
    result.push({ ...p, _recReason: 'random' });
  }

  return result.slice(0, 10);
}
