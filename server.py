"""Aftershock web server. Run: uvicorn server:app --port 8000"""
import asyncio, base64, copy, io, json, os, re, shutil, subprocess, tempfile, threading, time, uuid, glob
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from aftershock import pipeline, render, voice, mission, meta

app = FastAPI(title="Aftershock")
JOBS = {}
HERE = os.path.dirname(os.path.abspath(__file__))

def _b64(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=84); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def keyframes(path, n=3):
    d = tempfile.mkdtemp()
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout or 0)
    outs = []
    for i in range(n):
        t = dur * (i + 0.5) / n; o = f"{d}/f{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", o], check=True)
        outs.append((round(t, 1), o))
    return outs

def _run(job, frames, rows, cols, hazard):
    try:
        out = []
        for t, im in frames:
            job["status"] = f"surveying frame at {t}s" if t is not None else "surveying image"
            res, masks = pipeline.survey(im, rows, cols, hazard, progress=lambda *a: job.__setitem__("progress", job.get("progress", 0) + 1))
            res["frame_time"] = t; res["overlay"] = _b64(render.overlay(im, res, masks)); out.append(res)
        job.update(status="done", result=out)
    except Exception as e:
        job.update(status="error", error=repr(e)[:400])

@app.post("/api/survey")
async def survey(request: Request, file: UploadFile = File(...), rows: int = Form(2), cols: int = Form(2), hazard: str = Form("auto")):
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set; use the sample results")
    _budget_and_cooldown(request)
    data = await file.read(); jid = uuid.uuid4().hex[:10]; JOBS[jid] = {"status": "queued"}
    if (file.content_type or "").startswith("video"):
        p = tempfile.mktemp(suffix=os.path.splitext(file.filename or ".mp4")[1]); open(p, "wb").write(data)
        frames = [(t, Image.open(f).convert("RGB")) for t, f in keyframes(p)]
    else:
        frames = [(None, _open_image(data, file.filename))]
    threading.Thread(target=_run, args=(JOBS[jid], frames, max(1, min(rows, 4)), max(1, min(cols, 4)), hazard), daemon=True).start()
    return {"job": jid}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404)
    return JOBS[jid]

@app.post("/api/voice")
async def radio(request: Request, file: UploadFile = File(...), languages: str = Form("")):
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set")
    _budget_and_cooldown(request)
    p = tempfile.mktemp(suffix=os.path.splitext(file.filename or ".wav")[1] or ".wav"); open(p, "wb").write(await file.read())
    try: return voice.intake(p, [l.strip() for l in languages.split(",") if l.strip()] or None)
    finally: os.remove(p)

@app.get("/api/samples")
def samples():
    out = []
    for f in sorted(glob.glob(f"{HERE}/samples/*_result.json")):
        r = json.load(open(f)); stem = f[:-5]
        if os.path.exists(stem + ".jpg"): r["overlay"] = "/samples/" + os.path.basename(stem) + ".jpg"
        r["name"] = os.path.basename(f).replace("_result.json", ""); out.append(r)
    for f in sorted(glob.glob(f"{HERE}/samples/*_voice.json")):
        r = json.load(open(f)); r["name"] = os.path.basename(f).replace("_voice.json", ""); r["kind"] = "voice"; out.append(r)
    return out

# ------------------------------------------------------------------ Mission mode
MISSION_ID = os.environ.get("MISSION", "michael-sortie"); MDIR = f"{HERE}/missions/{MISSION_ID}"
STATE = {"m": None}; LIVE = {"busy": False, "runs": meta._BASE.get("live_runs", 0), "by_ip": {}}
MAX_LIVE = int(os.environ.get("AFTERSHOCK_MAX_LIVE", "6"))             # total live runs (persisted when AFTERSHOCK_SPEND_FILE is set)
IP_COOLDOWN_S = int(os.environ.get("AFTERSHOCK_IP_COOLDOWN_S", "0"))   # min seconds between live runs from one address
RESULTS_FIGURES = [  # quoted from docs/RESULTS.md (zero-shot, held-out public data)
    {"k": "Severe-damage recall", "v": "90/100", "n": "Spark prompt A, RescueNet T5 crops"},
    {"k": "Severe-damage precision", "v": "90/118", "n": "76%; over-flags rather than misses"},
    {"k": "Within one grade", "v": "191/200", "n": "exact 4-class grade 132/200 (66%)"},
    {"k": "Error when graders disagree", "v": "42.9%", "n": "12/28, vs 15.1% (26/172) when they agree"},
    {"k": "Destroyed-tile scan recall", "v": "28/30", "n": "precision 28/40 (T4a)"},
    {"k": "Blocked-road recall", "v": "18/20", "n": "precision 18/25 (T4b)"},
    {"k": "Flooded-road precision", "v": "22/22", "n": "recall 22/25, FloodNet (T8)"},
    {"k": "SAM finds destroyed buildings", "v": "3/13", "n": "vs 14/14 intact/minor: area scan covers the gap"},
]

# ---- per-visitor sessions: uploads go into a private copy of the mission; the shared demo mission is never changed by them
UPLOAD_ROOT = f"{MDIR}/uploads"                                         # git-ignored; wiped at start-up, nothing is persisted
MAX_SESSIONS = int(os.environ.get("AFTERSHOCK_MAX_SESSIONS", "50"))
SESSIONS = OrderedDict()                                                # session id -> {"m": mission copy, "dir": upload dir}
_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MB = 1024 * 1024
MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, MAX_AUDIO_BYTES = 20 * MB, 150 * MB, 15 * MB
MAX_AUDIO_S, MAX_FILES_PER_UPLOAD, MAX_SESSION_FRAMES = 180, 4, 12
LIVE_MAX_BUILDINGS = lambda: int(os.environ.get("LIVE_MAX_BUILDINGS", "8"))

def _sid(request):
    sid = (request.headers.get("x-aftershock-session") or "").strip()
    return sid if _SID.match(sid) else None

def _session(request, create=False):
    sid = _sid(request)
    if not sid: return None
    if sid in SESSIONS:
        SESSIONS.move_to_end(sid); return SESSIONS[sid]
    if not create: return None
    ses = {"id": sid, "m": copy.deepcopy(_mission()), "dir": f"{UPLOAD_ROOT}/{sid}", "busy": False}
    ses["m"]["session"] = {"uploads": 0, "radio": 0}
    SESSIONS[sid] = ses
    while len(SESSIONS) > MAX_SESSIONS:                                  # LRU eviction, never a session with a job running
        old = next((k for k, v in SESSIONS.items() if not v["busy"] and k != sid), None)
        if old is None: break
        _drop_session(old)
    return ses

def _drop_session(sid):
    ses = SESSIONS.pop(sid, None)
    if ses: shutil.rmtree(ses["dir"], ignore_errors=True)

def _decorate(m):
    for f in m["frames"]:
        v = f"?v={int(f.get('t', 0))}"
        if f.get("path"):    # uploaded frames and session re-surveys live in the session's own folder
            f["img"], f["thumb"], f["mask"] = f["path"] + ".jpg" + v, f["path"] + "_thumb.jpg" + v, f["path"] + "_mask.png" + v
            continue
        base = f"/missions/{MISSION_ID}/{'live' if f.get('live') else 'frames'}/{f['fid']}"
        f["img"], f["thumb"], f["mask"] = base + ".jpg" + v, f"/missions/{MISSION_ID}/frames/{f['fid']}_thumb.jpg", base + "_mask.png" + v
    m["results_figures"] = RESULTS_FIGURES; return m

def _mission(request=None):
    """The visitor's private mission copy if they have uploaded something, otherwise the shared demo mission."""
    if request is not None:
        ses = _session(request)
        if ses: return ses["m"]
    if STATE["m"] is None:
        p = f"{MDIR}/mission.json"
        if not os.path.exists(p): raise HTTPException(404, "mission not built: run tools/build_mission.py")
        STATE["m"] = _decorate(json.load(open(p)))
    return STATE["m"]

def _requeue(m):
    m["queue"] = mission.build_queue(m["frames"], m["radio"]); m["frame_summary"] = {f["fid"]: mission.frame_summary(f) for f in m["frames"]}

@app.get("/api/mission")
def get_mission(request: Request): return _mission(request)

@app.post("/api/mission/reset")
def reset_mission(request: Request):
    """With a private copy: drop it (and its uploads) and go back to the demo mission. Otherwise reset the shared demo mission."""
    sid = _sid(request)
    if sid and sid in SESSIONS:
        if SESSIONS[sid]["busy"]: raise HTTPException(409, "your upload is still being processed; reset when it finishes")
        _drop_session(sid); return _mission()
    STATE["m"] = None; return _mission()

@app.post("/api/radio/{rid}/confirm")
def confirm_radio(rid: str, request: Request, body: dict = Body(default={})):
    """A person read the count back to the caller. Optionally correct the count. Raises the item's priority."""
    m = _mission(request); r = next((x for x in m["radio"] if x["id"] == rid), None)
    if not r: raise HTTPException(404)
    r["confirmed"] = bool(body.get("confirmed", True))
    if body.get("people_trapped") is not None: r["people_trapped"] = int(body["people_trapped"]); r["corrected"] = True
    if body.get("sector"):
        sec = body["sector"].upper(); f = next((x for x in m["frames"] if x["sector"] == sec), None)
        if f: r["sector"], r["frame"], r["sector_exact"] = sec, f["fid"], True; r["flags"] = [x for x in r["flags"] if "sector" not in x]
    _requeue(m); return m

@app.get("/api/cost")
def cost():
    """Model calls made by THIS server process (live runs, uploads, radio uploads), priced at list rates."""
    return {"session_usd": meta.spent(), "by_model": meta.summary(), "budget_usd": meta.BUDGET_USD,
            "live_runs": LIVE["runs"], "max_live": MAX_LIVE, "live_enabled": bool(os.environ.get("MODEL_API_KEY")),
            "precomputed_mission": _mission().get("cost"),
            "upload_limits": {"images_per_upload": MAX_FILES_PER_UPLOAD, "image_mb": MAX_IMAGE_BYTES // MB, "video_mb": MAX_VIDEO_BYTES // MB,
                              "audio_mb": MAX_AUDIO_BYTES // MB, "audio_s": MAX_AUDIO_S, "frames_per_session": MAX_SESSION_FRAMES,
                              "buildings_graded_per_frame": LIVE_MAX_BUILDINGS()}}

@app.post("/api/validate")
def validate(request: Request, body: dict = Body(...)):
    """Run the commander-plan validator on any plan (no model call). Used by the UI's tamper test."""
    m = _mission(request); clean, issues = mission.validate_plan(body.get("plan"), mission.evidence_registry(m["frames"], m["radio"]), m["queue"])
    return {"plan": clean, "issues": issues}

def _frame_source(fr):
    """Full-resolution RescueNet image if present (RESCUENET_DIR or data/rescuenet), else the committed 1600 px display frame."""
    name = os.path.basename(fr["file"]); rn = os.environ.get("RESCUENET_DIR", f"{HERE}/data/rescuenet")
    for p in (fr["file"] if os.path.isabs(fr["file"]) else f"{HERE}/{fr['file']}", f"{rn}/{name}"):
        if os.path.exists(p): return p
    return f"{MDIR}/frames/{fr['fid']}.jpg"

# ---- caps shared by Run live and every upload job
def _ip(request):
    return (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()

def _budget_and_cooldown(request, what="live run"):
    """Budget and per-visitor cooldown. Used by Run live, uploads, and the single-frame /api/survey and /api/voice."""
    if meta.spent() >= meta.BUDGET_USD: raise HTTPException(429, f"public demo API budget (${meta.BUDGET_USD}) used up. The cached mission still works")
    ip = _ip(request); last = LIVE["by_ip"].get(ip, 0)
    if IP_COOLDOWN_S and time.time() - last < IP_COOLDOWN_S:
        raise HTTPException(429, f"one {what} per {max(1, IP_COOLDOWN_S // 60)} min per visitor; try again in {int(IP_COOLDOWN_S - (time.time() - last))} s")
    LIVE["by_ip"][ip] = time.time()

def _start_live(request, what="live run"):
    """Gate one live run (Run live or one upload job): key, one at a time, total cap, budget, per-visitor cooldown."""
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set: the cached mission still works")
    if LIVE["busy"]: raise HTTPException(409, "a live run or upload is already in progress; try again in a minute")
    if LIVE["runs"] >= MAX_LIVE: raise HTTPException(429, f"public demo cap reached ({MAX_LIVE} live runs or uploads). The cached mission still works")
    _budget_and_cooldown(request, what)
    LIVE.update(busy=True, runs=LIVE["runs"] + 1); meta._BASE["live_runs"] = LIVE["runs"]; meta.save_persisted(live_runs=LIVE["runs"])
    jid = uuid.uuid4().hex[:10]; JOBS[jid] = {"status": "running", "events": [], "t0": time.time()}
    return jid

def _commander(m):
    try: return mission.run_commander(m["queue"], m["radio"], m["frames"])
    except Exception as e:   # e.g. budget cap hit mid-run: same labelled rule-based fallback as an invalid Wonder plan
        plan = mission.fallback_plan(m["queue"])
        return dict(source="rule-based fallback (Wonder unavailable)", plan=plan, issues=[f"model error: {repr(e)[:160]}"],
                    input=mission.commander_input(m["queue"], m["radio"], m["frames"]), raw={"error": repr(e)[:160]}, seconds=0, usage=None,
                    tasks_ok=sum(t["status"] == "ok" for a in plan["assignments"] for t in a["tasks"]), tasks_rejected=0, teams=mission.TEAMS)

def _job(job, ses, work):
    """Run `work(m, ev)` on a copy of the right mission, stream model calls and stages, re-plan, and commit only on success."""
    calls = []
    def ev(e):
        job["events"].append(e)
        if e.get("event") == "end": calls.append(e)
    meta.LISTENERS.append(ev)
    try:
        m = copy.deepcopy(ses["m"] if ses else _mission())
        focus = work(m, ev)
        _requeue(m)
        ev({"event": "stage", "text": "Muse Wonder drafts team assignments from the queue (structured outputs only, no pixels)"})
        cmd = _commander(m); cmd["calls"] = []
        ev({"event": "stage", "text": f"Validator: {cmd['tasks_ok']} tasks accepted, {cmd['tasks_rejected']} rejected, {len(cmd['issues'])} issues"
                                      + ("" if cmd["source"].startswith("muse") else f" · {cmd['source']}")})
        m["commander"] = cmd
        lv = m["cost"].setdefault("live", {"runs": 0, "usd": 0.0, "calls": 0})
        lv.update(runs=lv["runs"] + 1, usd=round(lv["usd"] + sum(c["usd"] for c in calls), 4), calls=lv["calls"] + len(calls))
        if ses: ses["m"] = _decorate(m)
        else: STATE["m"] = _decorate(m)
        job.update(status="done"); ev({"event": "done", "focus": focus})
    except Exception as e:
        msg = str(e) if isinstance(e, (meta.BudgetExceeded, ValueError)) else repr(e)
        job.update(status="error", error=msg[:300]); ev({"event": "error", "text": msg[:300]})
        if ses and not any(f.get("upload") for f in ses["m"]["frames"]) and not any(r.get("upload") for r in ses["m"]["radio"]):
            _drop_session(ses["id"])   # a first upload that failed leaves no private copy behind
    finally:
        meta.LISTENERS.remove(ev); LIVE["busy"] = False
        if ses: ses["busy"] = False

def _live_job(job, fid, replan_only=False, ses=None):
    def work(m, ev):
        if replan_only: return None
        cur = next((x for x in m["frames"] if x["fid"] == fid), None)
        if cur is None: raise ValueError(f"no frame {fid} in this mission")
        if cur.get("upload"):
            fr = {"fid": fid, "sector": cur["sector"], "source": cur.get("source")}; src = f"{ses['dir']}/{fid}_src.jpg"
            out_dir, url = ses["dir"], f"/missions/{MISSION_ID}/uploads/{ses['id']}/{fid}"
        else:
            man = json.load(open(f"{MDIR}/manifest.json")); fr = next(x for x in man["frames"] if x["fid"] == fid); src = _frame_source(fr)
            out_dir, url = (f"{ses['dir']}/live", f"/missions/{MISSION_ID}/uploads/{ses['id']}/live/{fid}") if ses else (f"{MDIR}/live", None)
        ev({"event": "stage", "text": f"Surveying {fid} (Sector {fr['sector']}) live: SAM 3.1 segments, Spark scans 4 zones, Spark A + B grade each building"})
        img = Image.open(src).convert("RGB")
        f = mission.survey_frame(fr, img, out_dir, max_buildings=LIVE_MAX_BUILDINGS())
        f["t"] = time.time(); f["label_check"] = cur.get("label_check")
        if cur.get("upload"): f.update(upload=True, path=url, source=cur.get("source"))
        else:
            f["live"] = True
            if url: f["path"] = url
        if f["label_check"] is None: f.pop("label_check")
        m["frames"] = [f if x["fid"] == fid else x for x in m["frames"]]
        ev({"event": "stage", "text": f"{fid}: {len(f['buildings'])} buildings, {sum(z['road_blocked'] or z['flooded_road'] for z in f['zones'])} blocked-road zones. Rebuilding the mission queue"})
        return fid
    _job(job, ses, work)

@app.post("/api/live")
def live(request: Request, body: dict = Body(...)):
    ses = _session(request)
    if ses and ses["busy"]: raise HTTPException(409, "your upload is still being processed")
    jid = _start_live(request)
    if ses: ses["busy"] = True
    threading.Thread(target=_live_job, args=(JOBS[jid], body.get("fid", "F03"), bool(body.get("replan_only")), ses), daemon=True).start()
    return {"job": jid}

# ---- "Test with your own data": uploads into the visitor's private mission copy
async def _read_capped(file, limit, kind):
    data = await file.read(limit + 1)
    if len(data) > limit: raise HTTPException(413, f"{file.filename or kind} is larger than the {limit // MB} MB limit for {kind}")
    if not data: raise HTTPException(400, f"{file.filename or kind} is empty")
    return data

def _open_image(data, name=None):
    try:
        im = Image.open(io.BytesIO(data)); im.load(); return im.convert("RGB")
    except Exception:
        raise HTTPException(400, f"{name or 'the file'} is not an image we can open (use JPEG, PNG or WebP)")

def _no_key_message():
    return "Uploads need MODEL_API_KEY on the server (use the hosted demo, or run it with your own Meta Model API key). The cached mission still works"

def _check_request_size(request, limit):
    try: n = int(request.headers.get("content-length") or 0)
    except ValueError: n = 0
    if n > limit + MB: raise HTTPException(413, f"upload too large ({n // MB} MB); limits: images {MAX_IMAGE_BYTES // MB} MB each, video {MAX_VIDEO_BYTES // MB} MB, audio {MAX_AUDIO_BYTES // MB} MB")

@app.post("/api/upload/frames")
async def upload_frames(request: Request, files: list[UploadFile] = File(...)):
    """Survey up to 4 drone images (or one video -> 3 keyframes) into the visitor's private mission, then re-plan. Streams like /api/live."""
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, _no_key_message())
    if not _sid(request): raise HTTPException(400, "missing X-Aftershock-Session header; reload the page")
    _check_request_size(request, max(MAX_VIDEO_BYTES, MAX_FILES_PER_UPLOAD * MAX_IMAGE_BYTES))
    if not files: raise HTTPException(400, "choose at least one image or video")
    if len(files) > MAX_FILES_PER_UPLOAD: raise HTTPException(413, f"at most {MAX_FILES_PER_UPLOAD} images per upload")
    videos = [f for f in files if (f.content_type or "").startswith("video")]
    if videos and len(files) > 1: raise HTTPException(400, "upload one video on its own, or up to 4 images")
    frames = []   # (name, PIL image)
    if videos:
        v = videos[0]; data = await _read_capped(v, MAX_VIDEO_BYTES, "video")
        p = tempfile.mktemp(suffix=os.path.splitext(v.filename or ".mp4")[1] or ".mp4"); open(p, "wb").write(data)
        try:
            kf = keyframes(p)
            frames = [(f"{v.filename or 'video'} @ {t}s", Image.open(f).convert("RGB")) for t, f in kf]
            for _, f in kf: os.remove(f)
        except Exception:
            raise HTTPException(400, f"could not read video {v.filename or ''}: try MP4 (H.264) or upload images")
        finally: os.remove(p)
    else:
        for f in files:
            if f.content_type and not f.content_type.startswith("image") and f.content_type != "application/octet-stream":
                raise HTTPException(400, f"{f.filename} is not an image or video")
            frames.append((f.filename or "image", _open_image(await _read_capped(f, MAX_IMAGE_BYTES, "images"), f.filename)))
    ses0 = _session(request)
    have = sum(1 for f in (ses0["m"]["frames"] if ses0 else []) if f.get("upload"))
    if have + len(frames) > MAX_SESSION_FRAMES:
        raise HTTPException(413, f"at most {MAX_SESSION_FRAMES} of your own frames per visitor (you have {have}); use Reset to demo mission to start over")
    if ses0 and ses0["busy"]: raise HTTPException(409, "your previous upload is still being processed")
    jid = _start_live(request, "upload")
    ses = _session(request, create=True); ses["busy"] = True; os.makedirs(ses["dir"], exist_ok=True)

    def work(m, ev):
        n0 = sum(1 for f in m["frames"] if f.get("upload")); first = None
        ev({"event": "stage", "text": f"Your data: {len(frames)} frame(s). Each one: SAM 3.1 segments buildings, Spark scans 4 zones, Spark A + B grade up to {LIVE_MAX_BUILDINGS()} buildings"})
        for k, (name, img) in enumerate(frames, 1):
            fid, sec = f"U{n0 + k:02d}", f"U{n0 + k}"; first = first or fid
            ev({"event": "stage", "text": f"Surveying {fid} (Sector {sec}): {name}"})
            src = img.copy(); src.thumbnail((4000, 4000)); src.save(f"{ses['dir']}/{fid}_src.jpg", quality=90)
            f = mission.survey_frame({"fid": fid, "sector": sec, "source": f"Your data: {name}"}, img, ses["dir"], max_buildings=LIVE_MAX_BUILDINGS())
            f.update(upload=True, t=time.time(), path=f"/missions/{MISSION_ID}/uploads/{ses['id']}/{fid}")
            m["frames"].append(f)
            ev({"event": "stage", "text": f"{fid}: {len(f['buildings'])} buildings, {sum(z['road_blocked'] or z['flooded_road'] for z in f['zones'])} blocked-road zones"})
        m["session"] = {"uploads": n0 + len(frames), "radio": m.get("session", {}).get("radio", 0)}
        ev({"event": "stage", "text": "Rebuilding the mission queue with your frames"})
        return first
    threading.Thread(target=_job, args=(JOBS[jid], ses, work), daemon=True).start()
    return {"job": jid, "frames": len(frames)}

def _audio_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try: return float(out)
    except ValueError: return None

@app.post("/api/upload/radio")
async def upload_radio(request: Request, file: UploadFile = File(...), languages: str = Form(""), sector: str = Form("")):
    """Transcribe a radio call (Voice Transcribe + Spark), add it as a radio card in the visitor's private mission, fuse and re-plan."""
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, _no_key_message())
    if not _sid(request): raise HTTPException(400, "missing X-Aftershock-Session header; reload the page")
    _check_request_size(request, MAX_AUDIO_BYTES)
    if file.content_type and not (file.content_type.startswith("audio") or file.content_type.startswith("video") or file.content_type == "application/octet-stream"):
        raise HTTPException(400, f"{file.filename} is not an audio file")
    data = await _read_capped(file, MAX_AUDIO_BYTES, "audio")
    ext = (os.path.splitext(file.filename or "")[1] or ".wav").lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext): ext = ".wav"
    tmp = tempfile.mktemp(suffix=ext); open(tmp, "wb").write(data)
    dur = _audio_seconds(tmp)
    if dur is None: os.remove(tmp); raise HTTPException(400, f"could not read {file.filename or 'the audio'} as audio (try WAV, MP3, M4A or OGG)")
    if dur > MAX_AUDIO_S: os.remove(tmp); raise HTTPException(413, f"radio call is {int(dur)} s long; the limit is {MAX_AUDIO_S // 60} minutes")
    sec = re.sub(r"[^A-Za-z0-9]", "", sector or "").upper()[:6]
    m0 = _mission(request)
    if sec and not any(f["sector"] == sec for f in m0["frames"]):
        os.remove(tmp); raise HTTPException(400, f"no Sector {sec} in this mission (sectors: {', '.join(f['sector'] for f in m0['frames'])})")
    ses0 = _session(request)
    if ses0 and ses0["busy"]: os.remove(tmp); raise HTTPException(409, "your previous upload is still being processed")
    try: jid = _start_live(request, "upload")
    except HTTPException: os.remove(tmp); raise
    ses = _session(request, create=True); ses["busy"] = True; os.makedirs(ses["dir"], exist_ok=True)
    langs = [l.strip() for l in languages.split(",") if l.strip()][:4]

    def work(m, ev):
        n = sum(1 for r in m["radio"] if r.get("upload")) + 1; rid = f"U-R{n}"
        keep = f"{ses['dir']}/{rid}{ext}"; shutil.move(tmp, keep)
        ev({"event": "stage", "text": f"Your radio call {rid} ({dur:.0f} s): Muse Voice Transcribe, then Spark extracts counts with source quotes"})
        intake = voice.intake(keep, langs or None)
        sectors = {f["sector"] for f in m["frames"]}; s2f = {f["sector"]: f["fid"] for f in m["frames"]}
        card = mission.radio_card(rid, intake, sectors, s2f, audio=f"uploads/{ses['id']}/{rid}{ext}", lang=(langs[0] if langs else None), synthetic=False)
        if sec:   # the visitor told us where the call is from
            card.update(sector=sec, frame=s2f.get(sec), sector_exact=True, sector_words=f"set by you: Sector {sec}")
            card["flags"] = [x for x in card["flags"] if "sector" not in x and "location did not match" not in x]
        card["upload"] = True
        m["radio"].append(card)
        m["session"] = {"uploads": m.get("session", {}).get("uploads", 0), "radio": n}
        ev({"event": "stage", "text": f"{rid}: {card.get('people_trapped') if card.get('people_trapped') is not None else '?'} trapped reported"
                                      + (f", Sector {card['sector']} ({card['frame']})" if card.get("sector") else ", no sector match") + ". Fusing into the queue"})
        return rid
    threading.Thread(target=_job, args=(JOBS[jid], ses, work), daemon=True).start()
    return {"job": jid}

@app.get("/api/live/{jid}/events")
async def live_events(jid: str):
    if jid not in JOBS: raise HTTPException(404)
    async def gen():
        i = 0; job = JOBS[jid]
        while True:
            evs = job["events"]
            while i < len(evs):
                yield f"data: {json.dumps(evs[i])}\n\n"; i += 1
            if job["status"] != "running" and i >= len(job["events"]): break
            await asyncio.sleep(0.25)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

os.makedirs(f"{MDIR}/live", exist_ok=True)
shutil.rmtree(UPLOAD_ROOT, ignore_errors=True); os.makedirs(UPLOAD_ROOT, exist_ok=True)   # persist nothing across restarts
app.mount("/missions", StaticFiles(directory=f"{HERE}/missions"), name="missions")
app.mount("/samples", StaticFiles(directory=f"{HERE}/samples"), name="samples")
app.mount("/", StaticFiles(directory=f"{HERE}/web", html=True), name="web")
