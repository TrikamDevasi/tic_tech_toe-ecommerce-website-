import { Router } from 'express';
import Event from '../models/Event.js';
import Product from '../models/Product.js';
import { redisLPush, redisSet, redisGet, redisXAdd } from '../config/redis.js';

const router = Router();

// ── Engagement score weights ─────────────────────────────────────────────────
const ENGAGEMENT_WEIGHTS = {
  page_view: 1,
  search: 2,
  wishlist_add: 2,
  add_to_cart: 3,
  purchase: 5,
  remove_from_cart: 0,
};

// ── User segment from session data ──────────────────────────────────────────
function computeSegment(session) {
  const eng = session.engagementScore || 0;
  const intent = session.purchaseIntentScore || 0;
  const affinity = session.categoryAffinity || {};
  const electronicsAffinity = affinity instanceof Map
    ? (affinity.get('Electronics') || 0)
    : (affinity['Electronics'] || 0);

  if (eng > 15 || electronicsAffinity > 5) return 'premium_intent';
  if (intent < 0.2 && eng < 5) return 'value_seeker';
  return 'standard';
}

// POST /api/track
router.post('/', async (req, res, next) => {
  try {
    const {
      eventType,
      productId,
      sessionId = 'anonymous',
      userId,
      metadata = {},
      device,
      city,
    } = req.body;

    if (!eventType) {
      return res.status(400).json({ error: 'BadRequest', message: 'eventType is required' });
    }

    // ── Non-blocking Event Processing via Redis Streams ──────────────────
    // Publish to 'clickstream' for background worker to handle:
    // 1. Session analytics (engagement, affinity, intent)
    // 2. Product counter updates
    // 3. User segmentation
    await redisXAdd(
      'clickstream',
      '*',
      'data',
      JSON.stringify({
        eventType,
        productId,
        sessionId,
        userId,
        metadata,
        device: device || metadata.device || 'Unknown',
        city: city || metadata.city || 'Unknown',
        timestamp: new Date().toISOString()
      })
    );

    // Still create the raw event record for immediate persistent visibility
    await Event.create({
      sessionId,
      userId,
      eventType,
      productId: productId || null,
      metadata,
      device: device || metadata.device || 'Unknown',
      city: city || metadata.city || 'Unknown',
    });

    res.json({ success: true });
  } catch (err) {
    next(err);
  }
});

export default router;
