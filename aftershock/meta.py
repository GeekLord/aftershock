"""Thin clients for Meta Model API: Muse Spark/Wonder (chat), SAM 3.1 (segmentation), Muse Voice Transcribe."""
import os, io, json, base64, asyncio, time, requests
import numpy as np
from PIL import Image

API = "https://api.meta.ai/v1"
def _h(): return {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}

def data_url(im, max_side=1280, q=85):
    im = im.convert("RGB").copy(); im.thumbnail((max_side, max_side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def chat_json(prompt, images=(), model="muse-spark-1.3", max_side=1024, retries=3):
    """Ask a Muse model for a JSON object. Returns (dict, usage)."""
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": data_url(i, max_side)}} for i in images]
    last = None
    for a in range(retries):
        try:
            r = requests.post(f"{API}/chat/completions", headers=_h(), timeout=90, json={
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

def segment(im, noun, max_side=1600):
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
    req = {"mode": mode, "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    if languages: req["languageBias"] = list(languages)
    for a in range(3):
        with open(wav_path, "rb") as f:
            r = requests.post(f"{API}/asr/transcribe", headers=_h(), timeout=120, files={
                "request": (None, json.dumps(req), "application/json"), "audio": ("audio.wav", f, "audio/wav")})
        if r.status_code == 200: return r.json()
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"transcribe failed {r.status_code}: {r.text[:200]}")
