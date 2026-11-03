"""Aftershock web server. Run: uvicorn server:app --port 8000"""
import asyncio, base64, copy, io, json, os, subprocess, tempfile, threading, time, uuid, glob
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
async def survey(file: UploadFile = File(...), rows: int = Form(2), cols: int = Form(2), hazard: str = Form("auto")):
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set; use the sample results")
    data = await file.read(); jid = uuid.uuid4().hex[:10]; JOBS[jid] = {"status": "queued"}
    if (file.content_type or "").startswith("video"):
        p = tempfile.mktemp(suffix=os.path.splitext(file.filename or ".mp4")[1]); open(p, "wb").write(data)
        frames = [(t, Image.open(f).convert("RGB")) for t, f in keyframes(p)]
    else:
        frames = [(None, Image.open(io.BytesIO(data)).convert("RGB"))]
    threading.Thread(target=_run, args=(JOBS[jid], frames, max(1, min(rows, 4)), max(1, min(cols, 4)), hazard), daemon=True).start()
    return {"job": jid}

@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404)
    return JOBS[jid]

@app.post("/api/voice")
async def radio(file: UploadFile = File(...), languages: str = Form("")):
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set")
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

def _decorate(m):
    for f in m["frames"]:
        base = f"/missions/{MISSION_ID}/{'live' if f.get('live') else 'frames'}/{f['fid']}"; v = f"?v={int(f.get('t', 0))}"
        f["img"], f["thumb"], f["mask"] = base + ".jpg" + v, f"/missions/{MISSION_ID}/frames/{f['fid']}_thumb.jpg", base + "_mask.png" + v
    m["results_figures"] = RESULTS_FIGURES; return m

def _mission():
    if STATE["m"] is None:
        p = f"{MDIR}/mission.json"
        if not os.path.exists(p): raise HTTPException(404, "mission not built: run tools/build_mission.py")
        STATE["m"] = _decorate(json.load(open(p)))
    return STATE["m"]

def _requeue(m):
    m["queue"] = mission.build_queue(m["frames"], m["radio"]); m["frame_summary"] = {f["fid"]: mission.frame_summary(f) for f in m["frames"]}

@app.get("/api/mission")
def get_mission(): return _mission()

@app.post("/api/mission/reset")
def reset_mission(): STATE["m"] = None; return _mission()

@app.post("/api/radio/{rid}/confirm")
def confirm_radio(rid: str, body: dict = Body(default={})):
    """A person read the count back to the caller. Optionally correct the count. Raises the item's priority."""
    m = _mission(); r = next((x for x in m["radio"] if x["id"] == rid), None)
    if not r: raise HTTPException(404)
    r["confirmed"] = bool(body.get("confirmed", True))
    if body.get("people_trapped") is not None: r["people_trapped"] = int(body["people_trapped"]); r["corrected"] = True
    if body.get("sector"):
        sec = body["sector"].upper(); f = next((x for x in m["frames"] if x["sector"] == sec), None)
        if f: r["sector"], r["frame"], r["sector_exact"] = sec, f["fid"], True; r["flags"] = [x for x in r["flags"] if "sector" not in x]
    _requeue(m); return m

@app.get("/api/cost")
def cost():
    """Model calls made by THIS server process (live runs, radio uploads), priced at list rates."""
    return {"session_usd": meta.spent(), "by_model": meta.summary(), "budget_usd": meta.BUDGET_USD,
            "live_runs": LIVE["runs"], "max_live": MAX_LIVE, "live_enabled": bool(os.environ.get("MODEL_API_KEY")),
            "precomputed_mission": _mission().get("cost")}

@app.post("/api/validate")
def validate(body: dict = Body(...)):
    """Run the commander-plan validator on any plan (no model call). Used by the UI's tamper test."""
    m = _mission(); clean, issues = mission.validate_plan(body.get("plan"), mission.evidence_registry(m["frames"], m["radio"]), m["queue"])
    return {"plan": clean, "issues": issues}

def _frame_source(fr):
    """Full-resolution RescueNet image if present (RESCUENET_DIR or data/rescuenet), else the committed 1600 px display frame."""
    name = os.path.basename(fr["file"]); rn = os.environ.get("RESCUENET_DIR", f"{HERE}/data/rescuenet")
    for p in (fr["file"] if os.path.isabs(fr["file"]) else f"{HERE}/{fr['file']}", f"{rn}/{name}"):
        if os.path.exists(p): return p
    return f"{MDIR}/frames/{fr['fid']}.jpg"

def _live_job(job, fid, replan_only=False):
    calls = []
    def ev(e):
        job["events"].append(e)
        if e.get("event") == "end": calls.append(e)
    meta.LISTENERS.append(ev)
    try:
        m = copy.deepcopy(_mission())
        if not replan_only:
            man = json.load(open(f"{MDIR}/manifest.json")); fr = next(x for x in man["frames"] if x["fid"] == fid)
            ev({"event": "stage", "text": f"Surveying {fid} (Sector {fr['sector']}) live: SAM 3.1 segments, Spark scans 4 zones, Spark A + B grade each building"})
            img = Image.open(_frame_source(fr)).convert("RGB")
            f = mission.survey_frame(fr, img, f"{MDIR}/live", max_buildings=int(os.environ.get("LIVE_MAX_BUILDINGS", "8")))
            f["live"] = True; f["t"] = time.time(); f["label_check"] = next(x for x in m["frames"] if x["fid"] == fid).get("label_check")
            m["frames"] = [f if x["fid"] == fid else x for x in m["frames"]]
            ev({"event": "stage", "text": f"{fid}: {len(f['buildings'])} buildings, {sum(z['road_blocked'] or z['flooded_road'] for z in f['zones'])} blocked-road zones. Rebuilding the mission queue"})
        _requeue(m)
        ev({"event": "stage", "text": "Muse Wonder drafts team assignments from the queue (structured outputs only, no pixels)"})
        cmd = mission.run_commander(m["queue"], m["radio"], m["frames"]); cmd["calls"] = []
        ev({"event": "stage", "text": f"Validator: {cmd['tasks_ok']} tasks accepted, {cmd['tasks_rejected']} rejected, {len(cmd['issues'])} issues"})
        m["commander"] = cmd
        lv = m["cost"].setdefault("live", {"runs": 0, "usd": 0.0, "calls": 0})
        lv.update(runs=lv["runs"] + 1, usd=round(lv["usd"] + sum(c["usd"] for c in calls), 4), calls=lv["calls"] + len(calls))
        STATE["m"] = _decorate(m)
        job.update(status="done"); ev({"event": "done"})
    except Exception as e:
        job.update(status="error", error=repr(e)[:300]); ev({"event": "error", "text": repr(e)[:300]})
    finally:
        meta.LISTENERS.remove(ev); LIVE["busy"] = False

@app.post("/api/live")
def live(request: Request, body: dict = Body(...)):
    if not os.environ.get("MODEL_API_KEY"): raise HTTPException(503, "MODEL_API_KEY not set: the cached mission still works")
    if LIVE["busy"]: raise HTTPException(409, "a live run is already in progress")
    if LIVE["runs"] >= MAX_LIVE: raise HTTPException(429, f"public demo cap reached ({MAX_LIVE} live runs). The cached mission still works")
    if meta.spent() >= meta.BUDGET_USD: raise HTTPException(429, f"public demo API budget (${meta.BUDGET_USD}) used up. The cached mission still works")
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    last = LIVE["by_ip"].get(ip, 0)
    if IP_COOLDOWN_S and time.time() - last < IP_COOLDOWN_S:
        raise HTTPException(429, f"one live run per {IP_COOLDOWN_S // 60} min per visitor; try again in {int(IP_COOLDOWN_S - (time.time() - last))} s")
    LIVE["by_ip"][ip] = time.time()
    LIVE.update(busy=True, runs=LIVE["runs"] + 1); meta._BASE["live_runs"] = LIVE["runs"]; meta.save_persisted(live_runs=LIVE["runs"]); jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "events": [], "t0": time.time()}
    threading.Thread(target=_live_job, args=(JOBS[jid], body.get("fid", "F03"), bool(body.get("replan_only"))), daemon=True).start()
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
app.mount("/missions", StaticFiles(directory=f"{HERE}/missions"), name="missions")
app.mount("/samples", StaticFiles(directory=f"{HERE}/samples"), name="samples")
app.mount("/", StaticFiles(directory=f"{HERE}/web", html=True), name="web")
