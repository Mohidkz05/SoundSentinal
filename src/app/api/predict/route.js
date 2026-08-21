/**
 * The one place the browser and the model server meet.
 *
 * The upload goes to this route, and this route forwards it to Flask. Calling
 * `http://127.0.0.1:5000/predict` from the page instead would mean a CORS
 * preflight on every analysis (so `flask-cors`, so another dependency and
 * another thing to get wrong), and it would put the model server's address in
 * the client bundle — which makes it a public surface the moment this is
 * hosted anywhere. Proxying keeps Flask on loopback, keeps CORS out of the
 * picture entirely, and means one environment variable moves at deploy time.
 *
 * It also normalises failure. Everything below returns `{ error }` with a
 * sensible status, so the page has exactly one shape to handle and never has
 * to parse an HTML error page or distinguish "the model said no" from "the
 * model server isn't running".
 */

const MODEL_API = process.env.MODEL_API_URL ?? 'http://127.0.0.1:5000';

/** Matches MAX_BYTES in upload/page.js and MAX_UPLOAD_BYTES in ai_model/app.py.
 *  Checked here as well because a client-side check is a courtesy, not a
 *  control — this route is reachable directly. */
const MAX_BYTES = 5 * 1024 * 1024;

/** Analysis is a few hundred milliseconds of CPU; anything past this is the
 *  server being wedged rather than slow, and the page should say so instead of
 *  spinning until the browser gives up. */
const TIMEOUT_MS = 30_000;

function fail(message, status) {
  return Response.json({ error: message }, { status });
}

export async function POST(request) {
  let form;
  try {
    form = await request.formData();
  } catch {
    return fail('That request was not a file upload.', 400);
  }

  const file = form.get('file');
  if (!file || typeof file === 'string') {
    return fail('No audio file was included in the request.', 400);
  }
  if (file.size === 0) {
    return fail('That file is empty.', 400);
  }
  if (file.size > MAX_BYTES) {
    return fail(`Audio file exceeds the ${MAX_BYTES / (1024 * 1024)} MB limit.`, 413);
  }

  /* Rebuilt rather than forwarded as-is: the incoming FormData carries the
     browser's boundary and headers, and reusing them across the hop produces a
     body Werkzeug rejects. */
  const upstreamForm = new FormData();
  upstreamForm.append('file', file, file.name);

  let upstream;
  try {
    upstream = await fetch(`${MODEL_API}/predict`, {
      method: 'POST',
      body: upstreamForm,
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (e) {
    /* The common case by far during development: Flask isn't running. Say that,
       with the command that fixes it — a generic "analysis failed" sends people
       looking at their audio file. */
    const timedOut = e?.name === 'TimeoutError';
    console.error(`[api/predict] ${timedOut ? 'timed out' : 'could not reach'} ${MODEL_API}:`, e?.message);
    return fail(
      timedOut
        ? 'The model server took too long to answer.'
        : 'The model server is not reachable. Start it with `python ai_model/app.py`.',
      503
    );
  }

  let payload;
  try {
    payload = await upstream.json();
  } catch {
    return fail('The model server returned something that was not JSON.', 502);
  }

  if (!upstream.ok) {
    return fail(payload?.error ?? 'The model server could not analyse that clip.', upstream.status);
  }

  if (typeof payload?.spoof_probability !== 'number') {
    /* A 200 without the field the whole UI is built on means the two sides have
       drifted — worth failing loudly rather than rendering a reading of 0. */
    console.error('[api/predict] upstream 200 without spoof_probability:', payload);
    return fail('The model server returned an unexpected response.', 502);
  }

  return Response.json(payload);
}
