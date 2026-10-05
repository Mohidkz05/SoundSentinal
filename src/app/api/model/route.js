/**
 * What the served model is and how often it is wrong, for pages that show it
 * without a reading — the home page's "Measured, not claimed" strip.
 *
 * Read-only, and the same rule as /api/predict: the browser never addresses
 * Flask. The numbers come from the checkpoint actually being served (app.py
 * drops any report measured at another threshold), so the home page cannot
 * advertise an error rate for a model or operating point that isn't live.
 * Hosted, the model server scales to zero (deploy/azure.sh), so the first call
 * after a quiet spell waits for it to start — up to a couple of minutes. The
 * timeout allows for that rather than reporting a sleeping server as down, and
 * the hour-long revalidate means most visits never wait: these numbers change
 * only when a new checkpoint is deployed.
 */

const MODEL_API = process.env.MODEL_API_URL ?? 'http://127.0.0.1:5000';

/** Matches the cold-start allowance in ../predict/route.js. */
const TIMEOUT_MS = 180_000;

export async function GET() {
  try {
    const response = await fetch(`${MODEL_API}/health`, {
      signal: AbortSignal.timeout(TIMEOUT_MS),
      next: { revalidate: 3600 },
    });
    if (!response.ok) throw new Error(`status ${response.status}`);
    const payload = await response.json();
    return Response.json({
      threshold_score: payload.threshold_score ?? null,
      band_low: payload.uncertain_band?.low ?? null,
      measured: Array.isArray(payload.measured) ? payload.measured : [],
      model: payload.model ?? null,
    });
  } catch {
    return Response.json({ error: 'The model server is not running.' }, { status: 503 });
  }
}
