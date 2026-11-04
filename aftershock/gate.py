"""Scene gate: one cheap look at the whole image before any SAM, tiling or per-building Spark work.
A free local pre-check (size, aspect ratio) runs first; then ONE Muse Spark call on a small copy of the image decides whether it is
aerial / overhead imagery of buildings or terrain. Anything else (a portrait, an indoor scene, a document, a screenshot) is rejected
in seconds, and the visitor can still choose "Analyse anyway" (force=1), which skips this gate.
The gate fails OPEN: a model error, timeout or unparseable answer lets the image through, so an API hiccup never blocks real data."""
import concurrent.futures as cf, logging, time
from . import meta

log = logging.getLogger("aftershock.gate")
MODEL = "muse-spark-1.3"
TAG = "spark_gate:scene"
MAX_SIDE = 640          # long side of the copy sent to the model
MIN_SIDE = 240          # local pre-check: smaller than this on either side (below 240p video) is not a usable survey frame
MAX_ASPECT = 4.0        # local pre-check: panoramas, banners, receipts
TIMEOUT_S = 20
MAX_TOKENS = 1200       # Spark reasons before answering; this caps the bill without truncating the short JSON

PROMPT = ("You are the intake filter of a disaster damage-survey tool that analyses drone and aerial photos. Look at the WHOLE image and say what it is. "
 "is_aerial = taken from above (drone, aircraft, satellite, or a high oblique overhead view), not from ground or eye level. "
 "shows_buildings_or_terrain = shows buildings, roads, land, vegetation, water or debris from above. "
 "verdict = proceed ONLY when it is aerial/overhead imagery of buildings or terrain (damage may be absent or unclear: undamaged areas are valid survey frames). "
 "verdict = reject for ground-level photos of people, selfies, portraits, indoor scenes, documents, screenshots, maps, charts, drawings, logos or close-ups of objects. "
 'Reply ONLY JSON: {"is_aerial":true|false,"shows_buildings_or_terrain":true|false,"disaster_damage_visible":"yes|no|unclear",'
 '"scene":"<=12 words: what the image shows","verdict":"proceed|reject","reason":"<=20 words"}')

def _words(s, n):
    w = str(s or "").split(); return " ".join(w[:n]) + ("…" if len(w) > n else "")

def local_check(img):
    """Free checks, no model call. Returns a reject reason, or None when the image may go to the model."""
    W, H = img.size
    if min(W, H) < MIN_SIDE: return f"image is {W}×{H} px; survey frames need at least {MIN_SIDE} px on each side"
    if max(W, H) / max(1, min(W, H)) > MAX_ASPECT: return f"aspect ratio {max(W, H) / min(W, H):.1f}:1 looks like a banner, panorama or screenshot, not a drone frame"
    return None

def _result(verdict, stage, t0, scene="", reason="", usd=0.0, ms=None, **kw):
    return dict(verdict=verdict, stage=stage, scene=scene, reason=reason, model=MODEL if stage in ("model", "fail-open") else None,
                seconds=round((ms / 1000) if ms is not None else time.time() - t0, 1), usd=round(usd, 6), **kw)

def forced():
    """What the gate reports when the visitor asked to analyse anyway."""
    return dict(verdict="proceed", stage="forced", scene="", reason="analysed anyway at your request (scene check skipped)", model=None, seconds=0.0, usd=0.0)

def scene_check(img, name=None):
    """Return a verdict dict: verdict proceed|reject, stage local|model|fail-open, scene, reason, seconds, usd (+ the model's booleans)."""
    t0 = time.time(); img = img.convert("RGB")
    why = local_check(img)
    if why: return _result("reject", "local", t0, scene=f"{img.size[0]}×{img.size[1]} px image", reason=why, name=name)
    small = img.copy(); small.thumbnail((MAX_SIDE, MAX_SIDE))
    n0 = len(meta.LEDGER)
    ex = cf.ThreadPoolExecutor(1)
    try:
        fut = ex.submit(meta.chat_json, PROMPT, [small], MODEL, MAX_SIDE, 1, TAG, TIMEOUT_S, MAX_TOKENS)
        res, _ = fut.result(timeout=TIMEOUT_S + 2)
    except Exception as e:   # timeout, budget cap, anything else: fail open
        log.warning("scene gate failed open: %r", e)
        return _result("proceed", "fail-open", t0, reason=f"scene check unavailable ({type(e).__name__}); proceeding", name=name)
    finally:
        ex.shutdown(wait=False)
    rec = next((c for c in reversed(meta.LEDGER[n0:]) if c["tag"] == TAG), None)
    usd, ms = (rec["usd"], rec["ms"]) if rec else (0.0, None)
    ia, sb, verdict = res.get("is_aerial"), res.get("shows_buildings_or_terrain"), res.get("verdict")
    if "error" in res or not isinstance(ia, bool) or not isinstance(sb, bool) or verdict not in ("proceed", "reject"):
        log.warning("scene gate failed open: %s", str(res.get("error") or res)[:200])
        return _result("proceed", "fail-open", t0, reason="scene check gave no usable answer; proceeding", usd=usd, ms=ms, name=name, call=rec)
    ok = verdict == "proceed" and ia and sb          # proceed only for aerial imagery of buildings or terrain
    dmg = res.get("disaster_damage_visible"); dmg = dmg if dmg in ("yes", "no", "unclear") else "unclear"
    reason = _words(res.get("reason"), 20)
    if not ok and verdict == "proceed": reason = reason or ("not taken from above" if not ia else "no buildings or terrain visible")
    return _result("proceed" if ok else "reject", "model", t0, scene=_words(res.get("scene"), 12), reason=reason, usd=usd, ms=ms,
                   is_aerial=ia, shows_buildings_or_terrain=sb, disaster_damage_visible=dmg, name=name, call=rec)

def line(g):
    """One progress-log line, e.g. 'Scene check · Muse Spark · 1.8 s: aerial view of flooded houses → proceeding'."""
    who = {"model": "Muse Spark", "fail-open": "Muse Spark", "local": "local pre-check", "forced": "skipped"}[g["stage"]]
    head = f"Scene check · {who}" + (f" · {g['seconds']} s" if g["stage"] != "forced" else "")
    if g["stage"] == "forced": return f"{head}: analysing anyway at your request" + (f" ({g['name']})" if g.get("name") else "")
    if g["stage"] == "fail-open": return f"{head}: {g['reason']}" + (f" ({g['name']})" if g.get("name") else "")
    tail = "proceeding" if g["verdict"] == "proceed" else f"not an aerial disaster image, skipped ({g['reason']})"
    return f"{head}: {g['scene'] or g.get('name') or 'image'} → {tail}"
