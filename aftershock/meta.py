"""Thin clients for Meta Model API: Muse Spark/Wonder (chat), SAM 3.1 (segmentation), Muse Voice Transcribe."""
import os, io, json, base64, asyncio, time, threading, requests
import numpy as np
from PIL import Image

API = "https://api.meta.ai/v1"
def _h(): return {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}

# ---- Call ledger: every model call is counted and priced, and can be streamed to a live listener ----
# List prices (pay-as-you-go, standard mode) from https://dev.meta.ai/docs/pricing-rate-limits, checked 2026-11-01.
PRICE = {"muse-spark-1.3": (1.25e-6, 4.25e-6), "muse-wonder-1.0": (4.00e-6, 20.0e-6)}   # (input, output) per token
SAM_PER_IMAGE = 2.50 / 1000
ASR_PER_HOUR = 0.18
_LOCK = threading.Lock()
LEDGER = []            # one dict per call: model, tag, ms, tokens_in, tokens_out, usd, ok
LISTENERS = []         # callables receiving {"event": "start"|"end", ...}; used by the live-run SSE stream
BUDGET_USD = float(os.environ.get("AFTERSHOCK_BUDGET_USD", "5"))
# Optional persistent spend record (e.g. a Cloud Storage bucket mounted as a volume) so the cap survives
# restarts and scale-to-zero. Without it the cap applies to this process only.
SPEND_FILE = os.environ.get("AFTERSHOCK_SPEND_FILE", "")

class BudgetExceeded(RuntimeError): pass

def load_persisted():
    try: return json.load(open(SPEND_FILE)) if SPEND_FILE and os.path.exists(SPEND_FILE) else {}
    except Exception: return {}
_BASE = load_persisted()            # spend and live runs recorded by earlier processes

def save_persisted(**extra):
    if not SPEND_FILE: return
    with _LOCK:
        d = {"usd": round(_BASE.get("usd", 0.0) + sum(c["usd"] for c in LEDGER), 6), "calls": _BASE.get("calls", 0) + len(LEDGER),
             "live_runs": _BASE.get("live_runs", 0), "t": time.time()}
        d.update(extra)
    try:
        os.makedirs(os.path.dirname(SPEND_FILE) or ".", exist_ok=True); tmp = SPEND_FILE + ".tmp"
        json.dump(d, open(tmp, "w")); os.replace(tmp, SPEND_FILE)
    except Exception: pass

def spent():
    """Spend counted against the cap: this process plus anything persisted by earlier processes."""
    with _LOCK: return round(_BASE.get("usd", 0.0) + sum(c["usd"] for c in LEDGER), 4)

def summary(calls=None):
    calls = LEDGER if calls is None else calls; out = {}
    for c in calls:
        k = c["tag"].split(":")[0]; o = out.setdefault(k, {"calls": 0, "usd": 0.0, "tokens_in": 0, "tokens_out": 0})
        o["calls"] += 1; o["usd"] = round(o["usd"] + c["usd"], 4); o["tokens_in"] += c.get("tokens_in", 0); o["tokens_out"] += c.get("tokens_out", 0)
    return out

def _emit(ev):
    for f in list(LISTENERS):
        try: f(ev)
        except Exception: pass

def _start(model, tag):
    if spent() >= BUDGET_USD: raise BudgetExceeded(f"API budget cap ${BUDGET_USD} reached")
    cid = f"{time.time():.6f}-{threading.get_ident()}"
    _emit({"event": "start", "id": cid, "model": model, "tag": tag, "t": time.time()}); return cid, time.time()

def _end(cid, t0, model, tag, ok, tokens_in=0, tokens_out=0, usd=0.0, note=""):
    rec = {"id": cid, "model": model, "tag": tag, "ms": int((time.time() - t0) * 1000), "tokens_in": tokens_in,
           "tokens_out": tokens_out, "usd": round(usd, 6), "ok": ok, "note": note}
    with _LOCK: LEDGER.append(rec)
    save_persisted(live_runs=_BASE.get("live_runs", 0))
    _emit({"event": "end", **rec}); return rec

def data_url(im, max_side=1280, q=85):
    im = im.convert("RGB").copy(); im.thumbnail((max_side, max_side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def chat_json(prompt, images=(), model="muse-spark-1.3", max_side=1024, retries=3, tag=None, timeout=90):
    """Ask a Muse model for a JSON object. Returns (dict, usage). Every call is recorded in LEDGER under `tag`."""
    tag = tag or model
    cid, t0 = _start(model, tag)
    res, usage = _chat_json(prompt, images, model, max_side, retries, timeout)
    u = usage or {}; ti, to = u.get("prompt_tokens", 0), u.get("completion_tokens", 0); pi, po = PRICE.get(model, (0, 0))
    _end(cid, t0, model, tag, "error" not in res, ti, to, ti * pi + to * po, str(res.get("error", ""))[:120] if "error" in res else "")
    return res, usage

def _chat_json(prompt, images, model, max_side, retries, timeout=90):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": data_url(i, max_side)}} for i in images]
    last = None
    for a in range(retries):
        try:
            r = requests.post(f"{API}/chat/completions", headers=_h(), timeout=timeout, json={
                "model": model, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}})
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                return json.loads(t[t.find("{"):t.rfind("}") + 1]), r.json().get("usage")
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503): time.sleep(2 * (a + 1)); continue
            break
        except Exception as e:
            last = repr(e); time.sleep(1)
    return {"error": last}, None

def segment(im, noun, max_side=1600, tag="sam"):
    cid, t0 = _start("sam-3.1", tag)
    try:
        out = _segment(im, noun, max_side)
        _end(cid, t0, "sam-3.1", tag, True, usd=SAM_PER_IMAGE, note=f"{len(out)} objects"); return out
    except Exception as e:
        _end(cid, t0, "sam-3.1", tag, False, usd=SAM_PER_IMAGE, note=repr(e)[:120]); raise

def _segment(im, noun, max_side=1600):
    """SAM 3.1 text-prompted segmentation. Returns list of {box, conf, mask(bool ndarray in box)} in the
    coordinate space of `im` (rescaled from the submitted size)."""
    import meta_sam_parser as msp
    sent = im.convert("RGB").copy(); sent.thumbnail((max_side, max_side)); scale = im.size[0] / sent.size[0]
    r = requests.post(f"{API}/responses", headers={**_h(), "Accept": "text/event-stream"}, stream=True, timeout=120, json={
        "model": "sam-3.1", "stream": True, "metadata": {"include_confidence": "true"},
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": noun}, {"type": "input_image", "image_url": data_url(sent, max_side, 88)}]}]})
    events = [json.loads(l[6:]) for l in (x.decode() for x in r.iter_lines()) if l.startswith("data: ") and l[6:] != "[DONE]"]
    async def src():
        for e in events: yield e
    async def run():
        p = msp.parse_responses_stream(src(), msp.image_segmentation_format())
        async for _ in p: pass
        fr = p.final_result
        return (await fr()) if callable(fr) else (await fr if asyncio.iscoroutine(fr) else fr)
    res = asyncio.run(run()); out = []
    for rec in (res.records or []):
        if getattr(rec, "mask", None) is None or getattr(rec, "bounds", None) is None: continue
        mk = rec.mask; raw = np.frombuffer(msp.decode_mask_to_raster(mk), dtype=np.uint8)
        if raw.size != mk.width * mk.height: raw = np.unpackbits(raw)[: mk.width * mk.height]
        b = rec.bounds
        out.append({"box": [b.left * scale, b.top * scale, b.right * scale, b.bottom * scale],
                    "conf": float(rec.confidence or 0), "mask": raw.reshape(mk.height, mk.width) > 0})
    return out

def transcribe(wav_path, languages=None, mode="PUSH_TO_TALK"):
    cid, t0 = _start("muse-voice-transcribe-1.0", "voice")
    try:
        r = _transcribe(wav_path, languages, mode)
        _end(cid, t0, "muse-voice-transcribe-1.0", "voice", True, usd=(r.get("audioDurationMs") or 0) / 3.6e6 * ASR_PER_HOUR); return r
    except Exception as e:
        _end(cid, t0, "muse-voice-transcribe-1.0", "voice", False, note=repr(e)[:120]); raise

def _transcribe(wav_path, languages=None, mode="PUSH_TO_TALK"):
    req = {"mode": mode, "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    if languages: req["languageBias"] = list(languages)
    for a in range(3):
        with open(wav_path, "rb") as f:
            r = requests.post(f"{API}/asr/transcribe", headers=_h(), timeout=120, files={
                "request": (None, json.dumps(req), "application/json"), "audio": ("audio.wav", f, "audio/wav")})
        if r.status_code == 200: return r.json()
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"transcribe failed {r.status_code}: {r.text[:200]}")
