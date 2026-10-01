/**
 * What the served model is and how often it is wrong, for pages that show it
 * without a reading — the home page's "Measured, not claimed" strip.
 *
 * Read-only, and the same rule as /api/predict: the browser never addresses
 * Flask. The numbers come from the checkpoint actually being served (app.py
 * drops any report measured at another threshold), so the home page cannot
 * advertise an error rate for a model or operating point that isn't live.
 * A short revalidate keeps the home page from calling Flask on every visit.
 */

const MODEL_API = process.env.MODEL_API_URL ?? 'http://127.0.0.1:5000';

export async function GET() {
  try {
    const response = await fetch(`${MODEL_API}/health`, {
      signal: AbortSignal.timeout(5_000),
      next: { revalidate: 60 },
    });
    if (!response.ok) throw new Error(`status ${response.status}`);
    const payload = await response.json();
    return Response.json({
      threshold_score: payload.threshold_score ?? null,
      measured: Array.isArray(payload.measured) ? payload.measured : [],
      model: payload.model ?? null,
    });
  } catch {
    return Response.json({ error: 'The model server is not running.' }, { status: 503 });
  }
}
