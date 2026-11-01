"""Precompute a demo mission so the live demo is instant and reliable.
Usage (venv python):
  python tools/build_mission.py manifest            # write missions/<id>/manifest.json from the frame list below
  python tools/build_mission.py frame F01 [F02 ...] # survey frames (cached per frame; skips frames already done)
  python tools/build_mission.py radio               # synthesise + transcribe the demo radio calls
  python tools/build_mission.py assemble            # build queue + run the Wonder commander -> missions/<id>/mission.json
Environment:
  RESCUENET_DIR  folder holding the RescueNet v1.0 test images and labels, flat, named
                 RescueNet__test__test-org-img__<n>.jpg and RescueNet__test__test-label-img__<n>_lab.png
                 (default: <repo>/data/rescuenet; data/ is git-ignored)
  MISSION        mission id under missions/ (default: michael-sortie)
  AFTERSHOCK_BUDGET_USD  stop surveying frames once cached spend reaches this (default: 5)
"""
import json, os, sys, glob, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from aftershock import meta, mission, voice

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MID = os.environ.get("MISSION", "michael-sortie"); MD = f"{HERE}/missions/{MID}"
RN = os.environ.get("RESCUENET_DIR", os.path.join(HERE, "data", "rescuenet"))
BUDGET = float(os.environ.get("AFTERSHOCK_BUDGET_USD", "5"))
# One sortie: consecutive RescueNet test-set frame numbers (capture order). Sector names are demo labels in flight order,
# NOT georeferenced positions.
FRAMES = [11914, 11916, 11918, 11924, 11939, 11943, 11946, 11955, 11959, 11961, 11965, 11979]
# Radio audio is generated once by tools/synth/ (Sarvam Bulbul v3 for English, ElevenLabs for Spanish and Arabic) and
# committed under missions/<id>/radio/. This step only transcribes and extracts it. See missions/michael-sortie/radio/LABEL.md.
RADIO = [  # synthetic demo calls from scripted, fictional text; clearly labelled synthetic in the UI
    ("R1", "en", "Base, this is Unit two at Sector C1. Two people trapped under a collapsed house, one of them injured. The street is covered in debris. We need a crane and medics."),
    ("R2", "es", "Base, aquí el equipo cinco en el sector A2. Hay cuatro personas atrapadas en una casa derrumbada, dos heridos. Necesitamos una ambulancia."),
    ("R3", "ar", "القاعدة، هنا الفريق تسعة. عشرة أشخاص محاصرون في المسجد في شارع النيل، ثلاثة مصابين. نحتاج إلى سيارة إسعاف."),
]

def spent_so_far():
    tot = 0.0
    for f in glob.glob(f"{MD}/cache/*.json"):
        d = json.load(open(f)); tot += sum(c["usd"] for c in d.get("calls", []))
    return round(tot, 4)

def manifest():
    rows = "ABC"; frames = []
    for i, n in enumerate(FRAMES):
        frames.append(dict(fid=f"F{i + 1:02d}", sector=f"{rows[i // 4]}{i % 4 + 1}", source=f"RescueNet v1.0 test/{n}.jpg",
                           file=f"{RN}/RescueNet__test__test-org-img__{n}.jpg", label=f"{RN}/RescueNet__test__test-label-img__{n}_lab.png"))
    m = dict(id=MID, title="Hurricane Michael sortie (12 frames)", disaster="Hurricane Michael, Florida, Oct 2018",
             dataset="RescueNet v1.0 (Rahnemoonfar, Chowdhury, Murphy; Scientific Data 2023)", licence="CC0 1.0",
             dataset_url="https://doi.org/10.6084/m9.figshare.c.6647354",
             note="Consecutive test-set frame numbers treated as one drone sortie. Sector names are demo labels in capture order, not map positions.",
             frames=frames)
    json.dump(m, open(f"{MD}/manifest.json", "w"), indent=1); print("manifest", len(frames), "frames")

def label_truth(path):
    """RescueNet pixel labels, used ONLY for an after-the-fact check (models never see them).
    v1.0 classes: 3 no damage, 4 minor, 5 major, 6 total destruction (see dataset version note)."""
    import numpy as np
    a = np.array(Image.open(path)); n = a.size; frac = lambda *c: float(sum((a == k).sum() for k in c)) / n
    return dict(severe_px_pct=round(100 * frac(5, 6), 2), destroyed_px_pct=round(100 * frac(6), 2), building_px_pct=round(100 * frac(3, 4, 5, 6), 2),
                severe_present=frac(5, 6) > 0.005)

def frame(fids):
    m = json.load(open(f"{MD}/manifest.json"))
    for fr in m["frames"]:
        if fr["fid"] not in fids: continue
        out = f"{MD}/cache/{fr['fid']}.json"
        if os.path.exists(out): print("cached", fr["fid"]); continue
        if spent_so_far() >= BUDGET: print("BUDGET reached", spent_so_far()); return
        img = Image.open(fr["file"]).convert("RGB")
        d = mission.survey_frame(fr, img, f"{MD}/frames", max_buildings=int(os.environ.get("MAX_BUILDINGS", "14")))
        d["label_check"] = label_truth(fr["label"])
        json.dump(d, open(out, "w"), indent=1)
        print(fr["fid"], fr["sector"], "buildings", len(d["buildings"]), "secs", d["seconds"], "usd", round(sum(c["usd"] for c in d["calls"]), 4), "total", spent_so_far(), flush=True)

def radio():
    out = []
    for rid, lang, text in RADIO:
        wav = f"{MD}/radio/{rid}.wav"
        if not os.path.exists(wav):
            sys.exit(f"missing {wav}: generate it with tools/synth/ (see radio/LABEL.md)")
        n0 = len(meta.LEDGER); r = voice.intake(wav, [lang]); r["calls"] = meta.LEDGER[n0:]
        out.append(dict(id=rid, lang=lang, audio=f"radio/{rid}.wav", script=text, intake=r)); print(rid, json.dumps(r["extracted"], ensure_ascii=False))
    json.dump(out, open(f"{MD}/radio/radio.json", "w"), indent=1, ensure_ascii=False)

def load_parts():
    m = json.load(open(f"{MD}/manifest.json"))
    frames = [json.load(open(f"{MD}/cache/{fr['fid']}.json")) for fr in m["frames"] if os.path.exists(f"{MD}/cache/{fr['fid']}.json")]
    sectors = {f["sector"] for f in frames}; s2f = {f["sector"]: f["fid"] for f in frames}
    rr = json.load(open(f"{MD}/radio/radio.json")) if os.path.exists(f"{MD}/radio/radio.json") else []
    radio = [mission.radio_card(r["id"], r["intake"], sectors, s2f, audio=r["audio"], lang=r["lang"]) for r in rr]
    return m, frames, radio, rr

def assemble(run_wonder=True):
    m, frames, radio, rr = load_parts()
    queue = mission.build_queue(frames, radio)
    prev = f"{MD}/commander.json"
    if run_wonder or not os.path.exists(prev):
        cmd = mission.run_commander(queue, radio, frames); cmd["calls"] = [c for c in meta.LEDGER if c["tag"].startswith("wonder:commander")]
        json.dump(cmd, open(prev, "w"), indent=1)
    cmd = json.load(open(prev))
    calls = [c for f in frames for c in f["calls"]] + [c for r in rr for c in r["intake"].get("calls", [])] + cmd.get("calls", [])
    out = dict(manifest=m, frames=frames, frame_summary={f["fid"]: mission.frame_summary(f) for f in frames}, radio=radio,
               queue=queue, commander=cmd, cost=dict(by_model=meta.summary(calls), total_usd=round(sum(c["usd"] for c in calls), 3), calls=len(calls)))
    json.dump(out, open(f"{MD}/mission.json", "w"), indent=1, ensure_ascii=False)
    print("queue", len(queue), "commander", cmd["source"], "ok", cmd["tasks_ok"], "rejected", cmd["tasks_rejected"], "issues", cmd["issues"])
    print("cost", json.dumps(out["cost"]))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "manifest": manifest()
    elif cmd == "frame": frame(sys.argv[2:])
    elif cmd == "radio": radio()
    elif cmd == "assemble": assemble(run_wonder="--no-wonder" not in sys.argv)
