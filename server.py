"""Aftershock web server. Run: uvicorn server:app --port 8000"""
import base64, io, json, os, subprocess, tempfile, threading, uuid, glob
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from aftershock import pipeline, render, voice

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

app.mount("/samples", StaticFiles(directory=f"{HERE}/samples"), name="samples")
app.mount("/", StaticFiles(directory=f"{HERE}/web", html=True), name="web")
