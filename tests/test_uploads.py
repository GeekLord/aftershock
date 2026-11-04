"""Offline tests for "test with your own data" (no network: every model call is mocked).
Covers per-visitor session isolation, upload caps, and queue / radio fusion with an uploaded frame.
Run: python -m pytest -q tests"""
import io, os, sys, time, wave
import pytest
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import server
from aftershock import mission, meta

A, B = {"X-Aftershock-Session": "sessionAAAA1"}, {"X-Aftershock-Session": "sessionBBBB2"}

def _jpeg(w=640, h=480, color=(120, 90, 60)):
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, "JPEG"); return b.getvalue()

def _wav(seconds=1.0):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * int(16000 * seconds))
    return b.getvalue()

def fake_survey_frame(frame, img, out_dir, max_buildings=14, **kw):
    """Stands in for SAM + Spark: one destroyed building in zone 1, a blocked road in zone 2."""
    fid = frame["fid"]; os.makedirs(out_dir, exist_ok=True)
    disp = img.convert("RGB").resize((400, 300)); disp.save(f"{out_dir}/{fid}.jpg"); disp.save(f"{out_dir}/{fid}_thumb.jpg")
    Image.new("RGBA", (400, 300)).save(f"{out_dir}/{fid}_mask.png")
    W, H = img.size
    zones = [dict(id=f"{fid}-Z{k}", box=[0, 0, W // 2, H // 2], destroyed_present=(k == 3), road_blocked=(k == 2), flooded_road=False,
                  flooded_building=False, scan_evidence={"destroyed": "rubble" if k == 3 else None, "road": "debris on road" if k == 2 else None, "flood": None},
                  sam_capped=False, review=[], cell_score=0) for k in (1, 2, 3, 4)]
    b1 = dict(id=f"{fid}-B01", zone=f"{fid}-Z1", box=[10, 10, 200, 200], sam_conf=0.9, grade="destroyed",
              grades={"spark_a": "destroyed", "spark_b": "major", "wonder": None}, severe_votes="2/2", evidence="roof gone",
              review=False, review_reasons=[], agreement="Spark A=destroyed, Spark B=major (same severity band)")
    b2 = dict(b1, id=f"{fid}-B02", grade="no_damage", grades={"spark_a": "no_damage", "spark_b": "no_damage", "wonder": None},
              severe_votes="0/2", agreement="Spark A and B agree (no_damage)", box=[300, 300, 400, 400])
    return dict(fid=fid, sector=frame["sector"], source=frame.get("source"), size=[W, H], display=[400, 300], scale=400 / W,
                zones=zones, buildings=[b1, b2], seconds=0.1, calls=[], prompts_version="test")

def fake_commander(queue, radio, frames, model=None):
    """A grounded Wonder-style plan that cites the top items, put through the real validator."""
    reg = mission.evidence_registry(frames, radio)
    picks = queue[:2] + [it for it in queue if it["id"].startswith("U")][:2]   # the re-plan sees the new IDs in the rebuilt queue
    tasks = [{"target": it["id"], "action": "assess", "why": "Verify this item.", "evidence": [it["id"]]} for it in picks]
    plan, issues = mission.validate_plan({"assignments": [{"team": "T1", "tasks": tasks}], "holds": [], "summary": "test", "summary_evidence": []}, reg, queue)
    ok = sum(t["status"] == "ok" for a in plan["assignments"] for t in a["tasks"])
    return dict(source="muse-wonder-1.0", plan=plan, issues=issues, input=mission.commander_input(queue, radio, frames), raw={}, seconds=0,
                usage=None, tasks_ok=ok, tasks_rejected=0, teams=mission.TEAMS)

def fake_intake(path, languages=None):
    return dict(transcript="Base, unit four. Three people trapped in a collapsed house. We need medics.", duration_ms=1000,
                extracted={"english": "Three people trapped in a collapsed house. We need medics.", "location": None, "need": "medics",
                           "people_trapped": 3, "people_trapped_quote": "Three", "injured": None, "injured_quote": None},
                confirm=["confirm all counts by read-back before dispatch"])

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-not-a-key")
    monkeypatch.setattr(mission, "survey_frame", fake_survey_frame)
    monkeypatch.setattr(mission, "run_commander", fake_commander)
    monkeypatch.setattr(server.voice, "intake", fake_intake)
    monkeypatch.setattr(server, "MAX_LIVE", 100); monkeypatch.setattr(server, "IP_COOLDOWN_S", 0)
    monkeypatch.setattr(meta, "BUDGET_USD", 5.0)
    server.LIVE.update(busy=False, runs=0, by_ip={}); server.STATE["m"] = None
    for sid in list(server.SESSIONS): server._drop_session(sid)
    yield TestClient(server.app)
    for sid in list(server.SESSIONS): server._drop_session(sid)
    server.STATE["m"] = None

def _wait(jid, timeout=20):
    t0 = time.time()
    while server.JOBS[jid]["status"] == "running":
        assert time.time() - t0 < timeout, "job did not finish"; time.sleep(0.05)
    assert server.JOBS[jid]["status"] == "done", server.JOBS[jid].get("error")
    return server.JOBS[jid]

def _upload_frame(c, h=A, n=1):
    r = c.post("/api/upload/frames", headers=h, files=[("files", (f"mine{k}.jpg", _jpeg(), "image/jpeg")) for k in range(n)])
    assert r.status_code == 200, r.text; return _wait(r.json()["job"])

def test_session_isolation_and_reset(client):
    demo = client.get("/api/mission").json(); n_demo = len(demo["frames"])
    job = _upload_frame(client, A)
    assert job["events"][-1] == {"event": "done", "focus": "U01"}
    ma = client.get("/api/mission", headers=A).json()
    assert [f["fid"] for f in ma["frames"]][-1] == "U01" and len(ma["frames"]) == n_demo + 1
    u = ma["frames"][-1]
    assert u["sector"] == "U1" and u["upload"] and "label_check" not in u and "/uploads/sessionAAAA1/U01.jpg" in u["img"]
    assert ma["session"] == {"uploads": 1, "radio": 0}
    for h in (B, {}):   # another visitor, and a visitor without a session id, still see the untouched demo mission
        mb = client.get("/api/mission", headers=h).json()
        assert len(mb["frames"]) == n_demo and "session" not in mb and not any(f.get("upload") for f in mb["frames"])
    # Confirm acts on the session copy only
    rid = ma["radio"][0]["id"]
    assert client.post(f"/api/radio/{rid}/confirm", headers=A, json={"confirmed": True}).json()["radio"][0]["confirmed"]
    assert not client.get("/api/mission", headers=B).json()["radio"][0]["confirmed"]
    assert not server.STATE["m"]["radio"][0]["confirmed"]
    # files are on disk under the session dir, and reset removes copy and files
    d = server.SESSIONS["sessionAAAA1"]["dir"]; assert os.path.exists(f"{d}/U01.jpg") and os.path.exists(f"{d}/U01_src.jpg")
    back = client.post("/api/mission/reset", headers=A).json()
    assert len(back["frames"]) == n_demo and "sessionAAAA1" not in server.SESSIONS and not os.path.exists(d)

def test_lru_eviction(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_SESSIONS", 2)
    for k in range(3): _upload_frame(client, {"X-Aftershock-Session": f"lrusession{k}"})
    assert list(server.SESSIONS) == ["lrusession1", "lrusession2"]
    assert not os.path.exists(f"{server.UPLOAD_ROOT}/lrusession0")

def test_caps(client, monkeypatch):
    img = ("files", ("a.jpg", _jpeg(), "image/jpeg"))
    assert client.post("/api/upload/frames", files=[img]).status_code == 400                         # no session id
    assert client.post("/api/upload/frames", headers=A, files=[img] * 5).status_code == 413           # > 4 images
    r = client.post("/api/upload/frames", headers=A, files=[("files", ("notes.jpg", b"not really a jpeg", "image/jpeg"))])
    assert r.status_code == 400 and "not an image" in r.json()["detail"]
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 1000)
    r = client.post("/api/upload/frames", headers=A, files=[img]); assert r.status_code == 413 and "MB limit" in r.json()["detail"]
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 20 * server.MB)
    monkeypatch.setattr(server, "MAX_AUDIO_S", 1)
    r = client.post("/api/upload/radio", headers=A, files={"file": ("call.wav", _wav(2.0), "audio/wav")}); assert r.status_code == 413
    monkeypatch.setattr(server, "MAX_AUDIO_S", 180)
    assert client.post("/api/upload/radio", headers=A, files={"file": ("x.wav", b"garbage", "audio/wav")}).status_code == 400
    assert client.post("/api/upload/radio", headers=A, data={"sector": "Z9"}, files={"file": ("c.wav", _wav(), "audio/wav")}).status_code == 400
    assert not server.SESSIONS and server.LIVE["runs"] == 0                                            # rejected uploads cost no live run
    # live-run caps count uploads
    monkeypatch.setattr(server, "MAX_LIVE", 0)
    r = client.post("/api/upload/frames", headers=A, files=[img]); assert r.status_code == 429 and "cap" in r.json()["detail"]
    monkeypatch.setattr(server, "MAX_LIVE", 100); monkeypatch.setattr(meta, "BUDGET_USD", 0.0)
    for url, files in (("/api/upload/frames", [img]), ("/api/upload/radio", {"file": ("c.wav", _wav(), "audio/wav")}),
                       ("/api/survey", {"file": ("a.jpg", _jpeg(), "image/jpeg")}), ("/api/voice", {"file": ("c.wav", _wav(), "audio/wav")})):
        r = client.post(url, headers=A, files=files); assert r.status_code == 429 and "budget" in r.json()["detail"], (url, r.text)
    monkeypatch.setattr(meta, "BUDGET_USD", 5.0); monkeypatch.setattr(server, "IP_COOLDOWN_S", 600)
    _upload_frame(client, A)
    r = client.post("/api/upload/frames", headers=B, files=[img]); assert r.status_code == 429 and "per visitor" in r.json()["detail"]
    for url, files in (("/api/survey", {"file": ("a.jpg", _jpeg(), "image/jpeg")}), ("/api/voice", {"file": ("c.wav", _wav(), "audio/wav")})):
        assert client.post(url, files=files).status_code == 429, url
    assert server.LIVE["runs"] == 1
    monkeypatch.delenv("MODEL_API_KEY")
    r = client.post("/api/upload/frames", headers=A, files=[img]); assert r.status_code == 503 and "MODEL_API_KEY" in r.json()["detail"]
    assert client.get("/api/cost").json()["live_enabled"] is False

def test_queue_and_radio_fusion_with_uploaded_frame(client):
    _upload_frame(client, A)
    m = client.get("/api/mission", headers=A).json(); q = {i["id"]: i for i in m["queue"]}
    assert "U01-B01" in q and q["U01-B01"]["sector"] == "U1" and "U01-Z3" in q and "U01-B02" not in q
    assert q["U01-B01"]["frame"] == "U01" and m["frame_summary"]["U01"]["n_severe"] == 1
    assert any(t["target"].startswith("U01") for a in m["commander"]["plan"]["assignments"] for t in a["tasks"])   # re-planned with new IDs
    base = q["U01-B01"]["score"]
    r = client.post("/api/upload/radio", headers=A, data={"sector": "u1", "languages": "en"}, files={"file": ("call.wav", _wav(), "audio/wav")})
    assert r.status_code == 200, r.text; job = _wait(r.json()["job"]); assert job["events"][-1]["focus"] == "U-R1"
    m = client.get("/api/mission", headers=A).json(); card = m["radio"][-1]; q = {i["id"]: i for i in m["queue"]}
    assert card["id"] == "U-R1" and card["upload"] and card["sector"] == "U1" and card["frame"] == "U01" and card["people_trapped"] == 3
    assert card["audio"] == "uploads/sessionAAAA1/U-R1.wav" and not any("location did not match" in f for f in card["flags"])
    assert q["U01-B01"]["score"] == base + 3 and "U-R1" in q["U01-B01"]["radio"] and "U-R1" in q
    assert m["session"] == {"uploads": 1, "radio": 1}
    # the validator accepts a grounded claim citing the new IDs and rejects an ungrounded one
    plan = {"assignments": [{"team": "T1", "tasks": [
        {"target": "U01-B01", "action": "search_rescue", "why": "3 people trapped reported in a collapsed house.", "evidence": ["U01-B01", "U-R1"]},
        {"target": "U01-B02", "action": "search_rescue", "why": "Collapsed building, survivors likely.", "evidence": ["U01-B02"]}]}],
        "holds": [], "summary": "", "summary_evidence": []}
    st = [t["status"] for t in client.post("/api/validate", headers=A, json={"plan": plan}).json()["plan"]["assignments"][0]["tasks"]]
    assert st == ["ok", "rejected"]
    # confirming the uploaded call raises the fused item further (+5 instead of +3)
    m = client.post("/api/radio/U-R1/confirm", headers=A, json={"confirmed": True}).json()
    assert {i["id"]: i for i in m["queue"]}["U01-B01"]["score"] == base + 5
    # Run live on an uploaded frame re-surveys it inside the session
    r = client.post("/api/live", headers=A, json={"fid": "U01"}); assert r.status_code == 200; _wait(r.json()["job"])
    f = next(x for x in client.get("/api/mission", headers=A).json()["frames"] if x["fid"] == "U01")
    assert f["upload"] and not f.get("live") and "/uploads/sessionAAAA1/U01.jpg" in f["img"]

def test_video_becomes_three_keyframes(client, tmp_path):
    import subprocess, shutil
    if not shutil.which("ffmpeg"): pytest.skip("ffmpeg not installed")
    v = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-pix_fmt", "yuv420p", str(v)], check=True)
    r = client.post("/api/upload/frames", headers=A, files=[("files", ("clip.mp4", v.read_bytes(), "video/mp4"))])
    assert r.status_code == 200 and r.json()["frames"] == 3, r.text; _wait(r.json()["job"])
    m = client.get("/api/mission", headers=A).json()
    assert [f["sector"] for f in m["frames"] if f.get("upload")] == ["U1", "U2", "U3"]
