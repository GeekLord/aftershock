"""Offline tests for the scene gate (mocked Spark): reject before any SAM / survey work, proceed, fail open, and force=1.
Run: python -m pytest -q tests"""
import os, sys, time
import pytest
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server
from aftershock import gate, meta, mission
from test_uploads import client, _jpeg, _wait, A, B, GATE_CALLS, fake_survey_frame   # noqa: F401 (fixture)

PERSON = lambda: _jpeg(640, 480, (220, 30, 30))      # the mocked Spark calls a mostly-red image a portrait
AERIAL = lambda: _jpeg(640, 480, (120, 90, 60))

@pytest.fixture
def surveyed(monkeypatch):
    """Counts frames that reach the expensive survey; SAM must never be called directly by these flows."""
    n = []
    def counting(frame, img, out_dir, **kw): n.append(frame["fid"]); return fake_survey_frame(frame, img, out_dir, **kw)
    monkeypatch.setattr(mission, "survey_frame", counting)
    def no_sam(*a, **k): raise AssertionError("SAM called")
    monkeypatch.setattr(meta, "segment", no_sam)
    monkeypatch.setattr(server.pipeline, "survey", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pipeline.survey called")))
    return n

def test_local_precheck_needs_no_model():
    assert gate.local_check(Image.new("RGB", (200, 600))) and gate.local_check(Image.new("RGB", (2000, 300)))
    assert gate.local_check(Image.new("RGB", (1200, 900))) is None
    n0 = len(meta.LEDGER); g = gate.scene_check(Image.new("RGB", (100, 100)))
    assert g["verdict"] == "reject" and g["stage"] == "local" and g["usd"] == 0 and len(meta.LEDGER) == n0

def test_reject_is_fast_spends_no_sam_and_no_live_run(client, surveyed, monkeypatch):
    monkeypatch.setattr(server, "IP_COOLDOWN_S", 600)
    usd0 = meta.spent(); t0 = time.time()
    r = client.post("/api/upload/frames", headers=A, files=[("files", ("me.jpg", PERSON(), "image/jpeg"))])
    assert r.status_code == 200, r.text; d = r.json(); assert time.time() - t0 < 3
    assert d["job"] is None and d["frames"] == 0 and len(d["rejected"]) == 1
    g = d["rejected"][0]
    assert g["verdict"] == "reject" and g["stage"] == "model" and "portrait" in g["scene"] and g["reason"] and g["usd"] > 0 and "call" not in g
    assert max(GATE_CALLS[-1]) <= gate.MAX_SIDE                                   # downscaled copy sent
    assert surveyed == [] and server.LIVE["runs"] == 0 and not server.LIVE["by_ip"] and not server.SESSIONS and not server.LIVE["busy"]
    assert abs(meta.spent() - usd0 - g["usd"]) < 1e-4                              # only the gate's cost went to the budget
    assert "spark_gate" in client.get("/api/cost").json()["by_model"]
    # classic single frame: same verdict, no job, no cooldown stamp
    r = client.post("/api/survey", files={"file": ("me.jpg", PERSON(), "image/jpeg")}); d = r.json()
    assert r.status_code == 200 and d["job"] is None and d["rejected"] and d["gate"]["verdict"] == "reject" and not server.LIVE["by_ip"]
    # a rejected-only upload did not start the cooldown, so a real upload right after still runs (and counts)
    job = _wait(client.post("/api/upload/frames", headers=A, files=[("files", ("drone.jpg", AERIAL(), "image/jpeg"))]).json()["job"])
    assert surveyed == ["U01"] and server.LIVE["runs"] == 1

def test_proceed_and_mixed_upload(client, surveyed, monkeypatch):
    r = client.post("/api/upload/frames", headers=A, files=[("files", ("drone.jpg", AERIAL(), "image/jpeg")), ("files", ("me.jpg", PERSON(), "image/jpeg")),
                                                            ("files", ("drone2.jpg", AERIAL(), "image/jpeg"))])
    d = r.json(); assert r.status_code == 200 and d["job"] and d["frames"] == 2
    assert [g["name"] for g in d["rejected"]] == ["me.jpg"] and d["rejected"][0]["index"] == 1
    job = _wait(d["job"]); stages = [e["text"] for e in job["events"] if e["event"] == "stage"]
    assert stages[0].startswith("Scene check · Muse Spark") and "→ proceeding" in stages[0] and "not an aerial disaster image" in stages[1]
    assert job["events"][0]["event"] == "start" and job["events"][0]["tag"] == "spark_gate:scene"          # gate is the first step
    assert surveyed == ["U01", "U02"] and server.LIVE["runs"] == 1
    srcs = [f["source"] for f in client.get("/api/mission", headers=A).json()["frames"] if f.get("upload")]
    assert srcs == ["Your data: drone.jpg", "Your data: drone2.jpg"]
    # classic single frame proceeds and carries the gate line
    ran = []
    monkeypatch.setattr(server, "_run", lambda job, *a: (ran.append(1), job.update(status="done", result=[])))
    d = client.post("/api/survey", files={"file": ("drone.jpg", AERIAL(), "image/jpeg")}).json()
    assert d["job"] and d["gate"]["verdict"] == "proceed" and "→ proceeding" in server.JOBS[d["job"]]["gate_line"] and ran

def test_fail_open_on_model_error(client, surveyed, monkeypatch):
    def broken(*a, **k): return {"error": "503 upstream"}, None
    monkeypatch.setattr(meta, "chat_json", broken)
    g = gate.scene_check(Image.new("RGB", (800, 600), (220, 30, 30)))
    assert g["verdict"] == "proceed" and g["stage"] == "fail-open" and "proceeding" in gate.line(g)
    monkeypatch.setattr(meta, "chat_json", lambda *a, **k: ({"verdict": "maybe"}, None))      # unparseable answer
    assert gate.scene_check(Image.new("RGB", (800, 600)))["stage"] == "fail-open"
    def raises(*a, **k): raise meta.BudgetExceeded("cap")
    monkeypatch.setattr(meta, "chat_json", raises)
    assert gate.scene_check(Image.new("RGB", (800, 600)))["verdict"] == "proceed"
    monkeypatch.setattr(meta, "chat_json", broken)
    d = client.post("/api/upload/frames", headers=A, files=[("files", ("me.jpg", PERSON(), "image/jpeg"))]).json()
    assert d["job"] and not d["rejected"] and d["gate"][0]["stage"] == "fail-open"; _wait(d["job"]); assert surveyed == ["U01"]

def test_fail_open_on_timeout(monkeypatch):
    monkeypatch.setattr(gate, "TIMEOUT_S", 0.2)
    monkeypatch.setattr(meta, "chat_json", lambda *a, **k: (time.sleep(3), ({}, None))[1])
    t0 = time.time(); g = gate.scene_check(Image.new("RGB", (800, 600)))
    assert g["verdict"] == "proceed" and g["stage"] == "fail-open" and time.time() - t0 < 2.5

def test_force_skips_gate_and_counts_normally(client, surveyed, monkeypatch):
    n0 = len(GATE_CALLS)
    r = client.post("/api/upload/frames", headers=A, data={"force": "1"}, files=[("files", ("me.jpg", PERSON(), "image/jpeg"))])
    d = r.json(); assert r.status_code == 200 and d["job"] and not d["rejected"] and d["gate"][0]["stage"] == "forced"
    job = _wait(d["job"]); assert len(GATE_CALLS) == n0 and surveyed == ["U01"] and server.LIVE["runs"] == 1
    assert "analysing anyway" in next(e["text"] for e in job["events"] if e["event"] == "stage")
    monkeypatch.setattr(server, "_run", lambda job, *a: job.update(status="done", result=[]))
    d = client.post("/api/survey", data={"force": "1"}, files={"file": ("me.jpg", PERSON(), "image/jpeg")}).json()
    assert d["job"] and d["gate"]["stage"] == "forced" and len(GATE_CALLS) == n0 and server.LIVE["by_ip"]

def test_video_is_checked_once_on_its_middle_keyframe(client, surveyed, tmp_path):
    import subprocess, shutil
    if not shutil.which("ffmpeg"): pytest.skip("ffmpeg not installed")
    v = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0xdc1e1e:size=640x480:rate=10:duration=3", "-pix_fmt", "yuv420p", str(v)], check=True)
    n0 = len(GATE_CALLS)
    d = client.post("/api/upload/frames", headers=A, files=[("files", ("me.mp4", v.read_bytes(), "video/mp4"))]).json()
    assert len(GATE_CALLS) == n0 + 1 and d["job"] is None and len(d["rejected"]) == 1 and d["rejected"][0]["name"] == "me.mp4"
    assert surveyed == [] and server.LIVE["runs"] == 0
