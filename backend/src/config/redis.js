import Redis from 'ioredis';

let redis = null;
let redisAvailable = false;
const memoryStore = new Map();

// Safe in-memory fallback object matching ioredis methods used across the app
const fallbackRedis = {
  get: async (key) => memoryStore.get(key) || null,
  set: async (key, val) => { memoryStore.set(key, String(val)); return 'OK'; },
  del: async (key) => { memoryStore.delete(key); return 1; },
  incr: async (key) => {
    const cur = parseInt(memoryStore.get(key) || '0', 10) + 1;
    memoryStore.set(key, String(cur));
    return cur;
  },
  expire: async () => 1,
  lpush: async () => 1,
  lrange: async () => [],
  ltrim: async () => 'OK',
  xadd: async () => '0-0',
  xread: async () => null,
};

export function getRedis() {
  if (redis && redisAvailable) return redis;
  return fallbackRedis;
}

export function isRedisAvailable() {
  return redisAvailable;
}

export async function connectRedis() {
  const url = process.env.REDIS_URL?.trim();

  // If REDIS_URL is not set or empty, smoothly use in-memory store without error spam
  if (!url) {
    console.log('ℹ️  No REDIS_URL configured — running with fast in-memory cache.');
    redisAvailable = false;
    redis = null;
    return;
  }

  const isTLS = url.startsWith('rediss://');

  try {
    redis = new Redis(url, {
      maxRetriesPerRequest: 1,
      connectTimeout: 5000,
      lazyConnect: true,
      retryStrategy(times) {
        if (times > 2) return null; // Stop retrying after 2 attempts to avoid log spam
        return 1000;
      },
      ...(isTLS && {
        tls: {
          rejectUnauthorized: false,
        },
      }),
    });

    redis.on('connect', () => {
      redisAvailable = true;
      console.log('✅ Redis connected' + (isTLS ? ' (TLS/Upstash)' : ''));
    });

    let loggedError = false;
    redis.on('error', (err) => {
      redisAvailable = false;
      if (!loggedError) {
        console.warn('⚠️  Redis connection failed (falling back to in-memory/DB):', err.message);
        loggedError = true;
      }
    });

    redis.on('close', () => {
      redisAvailable = false;
    });

    await redis.connect();
  } catch (err) {
    redisAvailable = false;
    console.warn('⚠️  Redis unavailable, continuing with in-memory cache/DB fallback');
  }
}

// ── Safe wrappers — fallback to memory store if Redis is unavailable ────────

export async function redisGet(key) {
  if (redisAvailable && redis) {
    try { return await redis.get(key); } catch { /* fallback to memory */ }
  }
  return memoryStore.get(key) || null;
}

export async function redisSet(key, value, exSeconds) {
  if (redisAvailable && redis) {
    try {
      if (exSeconds) await redis.set(key, value, 'EX', exSeconds);
      else await redis.set(key, value);
      return;
    } catch { /* fallback to memory */ }
  }
  memoryStore.set(key, String(value));
}

export async function redisDel(key) {
  if (redisAvailable && redis) {
    try { await redis.del(key); return; } catch { /* silent */ }
  }
  memoryStore.delete(key);
}

export async function redisLPush(key, value) {
  if (redisAvailable && redis) {
    try {
      await redis.lpush(key, value);
      await redis.ltrim(key, 0, 9);
      return;
    } catch { /* silent */ }
  }
}

export async function redisLRange(key, start, stop) {
  if (redisAvailable && redis) {
    try { return await redis.lrange(key, start, stop); } catch { return []; }
  }
  return [];
}

export async function redisIncr(key) {
  if (redisAvailable && redis) {
    try { return await redis.incr(key); } catch { /* silent */ }
  }
  const cur = parseInt(memoryStore.get(key) || '0', 10) + 1;
  memoryStore.set(key, String(cur));
  return cur;
}

export async function redisExpire(key, seconds) {
  if (redisAvailable && redis) {
    try { await redis.expire(key, seconds); } catch { /* silent */ }
  }
}

export async function redisGetInt(key) {
  const val = await redisGet(key);
  return val ? parseInt(val, 10) : 0;
}

export async function redisXAdd(stream, ...args) {
  if (redisAvailable && redis) {
    try { await redis.xadd(stream, ...args); } catch { /* silent */ }
  }
}
