"""Offline tests (no API calls): sector matching, queue building and the commander-plan validator.
Run: python -m pytest -q tests  (or python tests/test_mission.py)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aftershock import mission
M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "missions/michael-sortie/mission.json")

def load():
    m = json.load(open(M)); return m["frames"], m["radio"], m["queue"]

def test_sector_matching():
    S = {"A1", "A2", "B3", "C1"}
    assert mission.match_sector("Unit two at Sector C1.", S)[0] == "C1"
    assert mission.match_sector("en el sector a a dos hay", S)[:3] == ("A2", "sector a a dos", False)   # real ASR output, inexact
    assert mission.match_sector("sector bravo three", S)[0] == "B3"
    assert mission.match_sector("the mosque in Nile Street", S)[0] is None

def test_validator_accepts_grounded_and_rejects_ungrounded():
    frames, radio, queue = load(); reg = mission.evidence_registry(frames, radio)
    sev = next(i for i in queue if i["kind"] == "building" and i["grade"] in ("major", "destroyed"))
    calm = next(b["id"] for f in frames for b in f["buildings"] if b["grade"] == "no_damage")
    rad = next(r for r in radio if r.get("frame") and r.get("people_trapped"))
    plan = {"assignments": [
        {"team": "T1", "tasks": [
            {"target": sev["id"], "action": "assess", "why": "Building graded severe; verify.", "evidence": [sev["id"]]},          # ok
            {"target": "F07-B99", "action": "search_rescue", "why": "Search it.", "evidence": ["F07-B99"]},                          # invented ID
            {"target": calm, "action": "search_rescue", "why": "Collapsed house, search.", "evidence": [calm]},                      # damage claim unsupported
        ]},
        {"team": "T2", "tasks": [
            {"target": rad["id"], "action": "medical", "why": f"{rad['people_trapped'] + 3} people trapped reported.", "evidence": [rad["id"]]},  # wrong count
            {"target": sev["id"], "action": "assess", "why": "Second team on same target.", "evidence": [sev["id"]]},          # duplicate
            {"target": calm, "action": "search_rescue", "why": "Search for survivors.", "evidence": [calm]},                    # people claim, no radio
        ]},
        {"team": "T9", "tasks": []}], "holds": [], "summary": "x", "summary_evidence": ["Z-404"]}
    clean, issues = mission.validate_plan(plan, reg, queue)
    st = [t["status"] for a in clean["assignments"] for t in a["tasks"]]
    assert st == ["ok", "rejected", "rejected", "rejected", "rejected", "rejected"], st
    assert any("unknown team T9" in i for i in issues) and any("Z-404" in i for i in issues)

def test_queue_radio_boost_and_confirmation():
    frames, radio, _ = load()
    for r in radio: r["confirmed"] = False
    q0 = {i["id"]: i["score"] for i in mission.build_queue(frames, radio)}
    rad = next(r for r in radio if r.get("frame") and r.get("people_trapped")); rad["confirmed"] = True
    q1 = {i["id"]: i["score"] for i in mission.build_queue(frames, radio)}
    boosted = [k for k in q1 if k.startswith(rad["frame"] + "-")]
    assert boosted and all(q1[k] == q0[k] + 2 for k in boosted) and q1[rad["id"]] == q0[rad["id"]] + 2

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
