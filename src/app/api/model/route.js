/**
 * What the served model is and how often it is wrong, for pages that show it
 * without a reading — the home page's "Measured, not claimed" strip.
 *
 * Read-only, and the same rule as /api/predict: the browser never addresses
 * Flask. The numbers come from the checkpoint actually being served (app.py
 * drops any report measured at another threshold), so the home page cannot
 * advertise an error rate for a model or operating point that isn't live.
 *
 * Hosted, the model server scales to zero (deploy/azure.sh), so a call after
 * a quiet spell waits ~30 s for it to start. These numbers change only when a
 * checkpoint is deployed, so on Cloudflare the answer is kept in the edge
 * cache and served at once; past FRESH_MS the next visit still gets the cached
 * copy and refreshes it in the background. Only a cold cache waits for the
 * server. (`next: { revalidate }` does nothing here: open-next.config.ts uses
 * the read-only static-assets cache.) Under `next dev` there is no edge cache
 * and every call goes to the server.
 */

import { getCloudflareContext } from '@opennextjs/cloudflare';

const MODEL_API = process.env.MODEL_API_URL ?? 'http://127.0.0.1:5000';

/** Matches the cold-start allowance in ../predict/route.js. */
const TIMEOUT_MS = 180_000;

/** How old a cached answer may be before a visit refreshes it. */
const FRESH_MS = 60 * 60 * 1000;
/** How long the edge keeps it at all — a week without a single visit. */
const KEEP_S = 7 * 24 * 60 * 60;
/** Cache API keys are URLs; this one is never fetched. */
const CACHE_KEY = 'https://soundsentinal.internal/api/model';

async function fromServer() {
  const response = await fetch(`${MODEL_API}/health`, {
    signal: AbortSignal.timeout(TIMEOUT_MS),
    cache: 'no-store',
  });
  if (!response.ok) throw new Error(`status ${response.status}`);
  const payload = await response.json();
  return {
    threshold_score: payload.threshold_score ?? null,
    band_low: payload.uncertain_band?.low ?? null,
    measured: Array.isArray(payload.measured) ? payload.measured : [],
    model: payload.model ?? null,
  };
}

async function store(cache, body) {
  await cache.put(
    CACHE_KEY,
    new Response(JSON.stringify(body), {
      headers: {
        'content-type': 'application/json',
        'cache-control': `max-age=${KEEP_S}`,
        'x-fetched-at': String(Date.now()),
      },
    })
  );
}

function edgeCache() {
  /* `caches.default` exists only in the Workers runtime. */
  return typeof caches !== 'undefined' && caches.default ? caches.default : null;
}

export async function GET() {
  const cache = edgeCache();

  if (cache) {
    const hit = await cache.match(CACHE_KEY);
    if (hit) {
      const age = Date.now() - Number(hit.headers.get('x-fetched-at') ?? 0);
      if (age > FRESH_MS) {
        getCloudflareContext().ctx.waitUntil(
          fromServer().then((body) => store(cache, body)).catch(() => {})
        );
      }
      return new Response(hit.body, { headers: { 'content-type': 'application/json' } });
    }
  }

  try {
    const body = await fromServer();
    if (cache) await store(cache, body);
    return Response.json(body);
  } catch {
    return Response.json({ error: 'The model server is not running.' }, { status: 503 });
  }
}
