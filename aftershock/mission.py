"""Mission mode: one drone sortie (many frames) -> one priority queue -> a Muse Wonder incident-commander plan.

Design rules (from docs/RESULTS.md):
- Perception happens only in the frozen, validated prompts (pipeline.py / prompts.py) and SAM.
- Aggregation and scoring are plain, inspectable code. Scores rank; they are not probabilities.
- Muse Wonder only orchestrates: it sees structured outputs (IDs, grades, flags, radio facts), never pixels, and every
  claim it makes must cite evidence IDs that exist and actually support the claim, or the claim is rejected.
"""
import json, os, re, time
import numpy as np
from PIL import Image
from . import meta, pipeline

SEV_BASE = {"destroyed": 6, "major": 4}
GRADE_ORDER = {"unknown": -1, "no_damage": 0, "minor": 1, "major": 2, "destroyed": 3}
COL = {"no_damage": (34, 197, 94), "minor": (234, 179, 8), "major": (249, 115, 22), "destroyed": (220, 38, 38), "unknown": (148, 163, 184)}
TEAMS = [
    {"team": "T1", "name": "Team 1 (heavy rescue / USAR)", "can": "search collapsed structures, shoring, extrication, crane coordination"},
    {"team": "T2", "name": "Team 2 (search + medical)", "can": "search damaged buildings, triage and treat casualties, light extrication"},
    {"team": "T3", "name": "Team 3 (route clearance + recon)", "can": "clear or bypass blocked roads, verify area-scan detections on foot, mark buildings"},
]

# ------------------------------------------------------------------ per-frame survey
def survey_frame(frame, img, out_dir, max_buildings=14, rows=2, cols=2, display_side=1600):
    """Run the validated survey on one frame and write the display image, mask layer and frame JSON."""
    fid = frame["fid"]; os.makedirs(out_dir, exist_ok=True)
    n0 = len(meta.LEDGER); t0 = time.time()
    res, masks = pipeline.survey(img, rows, cols, "auto", max_buildings=max_buildings)
    calls = meta.LEDGER[n0:]
    W, H = img.size; s = display_side / max(W, H); dw, dh = int(W * s), int(H * s)
    # display image, thumbnail, transparent mask layer coloured by grade
    disp = img.convert("RGB").resize((dw, dh)); disp.save(f"{out_dir}/{fid}.jpg", quality=82)
    th = disp.copy(); th.thumbnail((360, 270)); th.save(f"{out_dir}/{fid}_thumb.jpg", quality=78)
    lay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
    for b, grade in masks:
        x1, y1, x2, y2 = [int(v * s) for v in b["box"]]
        if x2 - x1 < 2 or y2 - y1 < 2: continue
        m = Image.fromarray((b["mask"] * 150).astype(np.uint8)).resize((x2 - x1, y2 - y1))
        lay.paste(Image.new("RGBA", m.size, COL.get(grade, COL["unknown"]) + (255,)), (x1, y1), m)
    lay.save(f"{out_dir}/{fid}_mask.png", optimize=True)
    zones, buildings = [], []
    for cell in sorted(res["cells"], key=lambda c: (c["r"], c["c"])):
        zid = f"{fid}-Z{cell['r'] * cols + cell['c'] + 1}"; sc = cell["scan"]; fl = sc.get("flood", {})
        zones.append(dict(id=zid, box=cell["box"], destroyed_present=bool(sc["destroyed"].get("destroyed_present")),
                          road_blocked=bool(sc["road"].get("blocked")), flooded_road=bool(fl.get("flooded_road")),
                          flooded_building=bool(fl.get("flooded_building")),
                          scan_evidence={"destroyed": sc["destroyed"].get("evidence"), "road": sc["road"].get("evidence"), "flood": fl.get("evidence")},
                          sam_capped=cell["sam_capped"], review=cell["review"], cell_score=cell["score"]))
        for b in cell["buildings"]:
            buildings.append(dict(b, zone=zid))
    buildings.sort(key=lambda b: b["key"])
    for i, b in enumerate(buildings, 1):
        b["id"] = f"{fid}-B{i:02d}"; b.pop("key", None)
        g = b["grades"]; va, vb = g.get("spark_a"), g.get("spark_b")
        if b["grade"] == "unknown": b["agreement"] = "not graded"
        elif g.get("wonder"): b["agreement"] = f"Spark A={va}, Spark B={vb} disagree; Wonder tie-break={g['wonder']}"
        elif va == vb: b["agreement"] = f"Spark A and B agree ({va})"
        else: b["agreement"] = f"Spark A={va}, Spark B={vb} (same severity band)"
    return dict(fid=fid, sector=frame["sector"], source=frame.get("source"), size=[W, H], display=[dw, dh], scale=s,
                zones=zones, buildings=buildings, seconds=round(time.time() - t0, 1),
                calls=[{k: c[k] for k in ("model", "tag", "ms", "tokens_in", "tokens_out", "usd", "ok")} for c in calls],
                prompts_version=res["prompts_version"])

# ------------------------------------------------------------------ radio fusion
_LET = {"a": "A", "alpha": "A", "alfa": "A", "b": "B", "bravo": "B", "bee": "B", "be": "B", "c": "C", "charlie": "C", "see": "C", "ce": "C", "d": "D", "delta": "D"}
_NUM = {"1": 1, "one": 1, "uno": 1, "un": 1, "2": 2, "two": 2, "to": 2, "dos": 2, "deux": 2, "3": 3, "three": 3, "tres": 3, "trois": 3, "4": 4, "four": 4, "for": 4, "cuatro": 4, "quatre": 4}

def match_sector(text, sectors):
    """Find 'Sector B2' / 'sector bravo two' / 'sector B dos' / ASR-mangled 'sector a a dos' in free text.
    Returns (sector label, matched words, exact?) or (None, None, False)."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    for m in re.finditer(r"\bsect(?:or|eur)\b((?:\s+\w+){1,3})", t):
        toks = m.group(1).split(); words = "sector " + " ".join(toks)
        if re.fullmatch(r"[a-d][1-9]", toks[0]):
            lab = toks[0].upper(); exact = True
        else:
            let = next((i for i, w in enumerate(toks) if w in _LET), None)
            num = next((i for i, w in enumerate(toks) if let is not None and i > let and w in _NUM), None)
            if let is None or num is None: continue
            lab = f"{_LET[toks[let]]}{_NUM[toks[num]]}"; exact = num == let + 1
            words = "sector " + " ".join(toks[:num + 1])
        if lab in sectors: return lab, words, exact
    return None, None, False

def radio_card(rid, intake, sectors, sector_to_fid, audio=None, lang=None, synthetic=True):
    e = intake.get("extracted") or {}
    sec, words, exact = None, None, False
    for src in (intake.get("transcript"), e.get("location"), e.get("english")):
        sec, words, exact = match_sector(src, sectors)
        if sec: break
    return dict(id=rid, kind="radio", lang=lang, audio=audio, synthetic=synthetic, transcript=intake.get("transcript"),
                english=e.get("english"), location_text=e.get("location"), need=e.get("need"),
                people_trapped=e.get("people_trapped"), people_trapped_quote=e.get("people_trapped_quote"),
                injured=e.get("injured"), injured_quote=e.get("injured_quote"),
                sector=sec, sector_words=words, sector_exact=exact, frame=sector_to_fid.get(sec), confirmed=False,
                flags=list(intake.get("confirm") or []) + ([] if sec else ["location did not match any sector: place it by hand"])
                      + ([f"sector read from '{words}' (inexact match): confirm location"] if sec and not exact else []))

# ------------------------------------------------------------------ mission queue
def build_queue(frames, radio):
    """One priority queue across all frames. Transparent integer score; a ranking aid, not a probability."""
    zones = {z["id"]: z for f in frames for z in f["zones"]}
    radio_by_frame = {}
    for r in radio:
        if r.get("frame"): radio_by_frame.setdefault(r["frame"], []).append(r)
    items = []
    for f in frames:
        fid = f["fid"]; rads = radio_by_frame.get(fid, []); severe_zones = set()
        for b in f["buildings"]:
            disputed = "assessors disagree on severity" in b["review_reasons"]
            if b["grade"] not in SEV_BASE and not disputed: continue
            z = zones[b["zone"]]; score = SEV_BASE.get(b["grade"], 2); why = []; ev = [b["id"]]
            if b["grade"] in SEV_BASE:
                why.append(f"graded {b['grade']} ({b['severe_votes']} severe votes)"); severe_zones.add(z["id"])
            else: why.append(f"graded {b['grade']} but assessors split on severity")
            if z["destroyed_present"] and b["grade"] in SEV_BASE: score += 1; why.append("area scan of its zone also sees destruction"); ev.append(z["id"])
            items.append(dict(id=b["id"], kind="building", frame=fid, sector=f["sector"], grade=b["grade"], base=score,
                              agreement=b["agreement"], review=b["review_reasons"], why=why, evidence=ev, zone=z["id"], box=b["box"]))
        for z in f["zones"]:
            if z["destroyed_present"] and z["id"] not in severe_zones:
                items.append(dict(id=z["id"], kind="zone", frame=fid, sector=f["sector"], grade="destroyed (area scan)", base=5,
                                  agreement="area scan only; no SAM building graded severe here",
                                  review=["area scan sees destruction the building grader did not (SAM misses flattened buildings)"],
                                  why=["Spark area scan: destroyed structure present"], evidence=[z["id"]], zone=z["id"], box=z["box"]))
    for it in items:   # road access and radio fusion
        z = zones[it["zone"]]
        it["access"] = "blocked" if (z["road_blocked"] or z["flooded_road"]) else "open"
        it["access_evidence"] = z["id"] if it["access"] == "blocked" else None
        it["access_why"] = ("road under floodwater" if z["flooded_road"] else "road likely impassable") if it["access"] == "blocked" else "no blocked road seen in its zone"
        it["radio"] = []; it["score"] = it["base"]
        for r in radio_by_frame.get(it["frame"], []):
            if r.get("people_trapped"):
                bump = 5 if r["confirmed"] else 3
                it["score"] += bump; it["radio"].append(r["id"]); it["evidence"].append(r["id"])
                it["why"].append(f"radio {r['id']}: {r['people_trapped']} trapped reported in {r['sector']} ({'confirmed by read-back' if r['confirmed'] else 'NEEDS HUMAN CONFIRMATION'})")
    for r in radio:
        trapped = r.get("people_trapped") or 0
        items.append(dict(id=r["id"], kind="radio", frame=r.get("frame"), sector=r.get("sector"), grade="radio report",
                          base=7 if trapped else 3, score=(7 if trapped else 3) + (2 if r["confirmed"] else 0),
                          agreement="confirmed by read-back" if r["confirmed"] else "count needs human confirmation",
                          review=[] if r["confirmed"] else ["count not confirmed: read back before dispatch"],
                          why=[f"{trapped} trapped, {r.get('injured') or 0} injured reported" + (f" in {r['sector']}" if r.get("sector") else " (no sector match)"),
                               f"need: {r.get('need')}"],
                          evidence=[r["id"]], zone=None, access="unknown", access_evidence=None, access_why="", radio=[r["id"]]))
    items.sort(key=lambda i: (-i["score"], -GRADE_ORDER.get(i["grade"], 2), i["id"]))
    for k, it in enumerate(items, 1): it["rank"] = k
    return items

def frame_summary(f):
    sev = [b["grade"] for b in f["buildings"] if b["grade"] in GRADE_ORDER]
    worst = max(sev, key=lambda g: GRADE_ORDER[g]) if sev else "no_damage"
    if any(z["destroyed_present"] for z in f["zones"]) and GRADE_ORDER[worst] < 3: worst = "destroyed"
    return dict(worst=worst, n_buildings=len(f["buildings"]), n_severe=sum(g in SEV_BASE for g in sev),
                n_review=sum(b["review"] for b in f["buildings"]), roads_blocked=sum(z["road_blocked"] or z["flooded_road"] for z in f["zones"]),
                destroyed_zones=sum(z["destroyed_present"] for z in f["zones"]))

# ------------------------------------------------------------------ incident commander (Muse Wonder)
COMMANDER_PROMPT = """You are the planning assistant to a disaster-response incident commander. You do NOT look at images.
You receive ONLY structured outputs from validated tools: a mission priority queue (buildings and zones with IDs, damage grades
from two independent graders, review flags, road access from an area scan) and radio reports (counts extracted with source quotes,
not yet confirmed unless marked). Draft initial assignments for the teams below.

Rules:
- Order work by: reported trapped people first, then severity (destroyed > major), then road access.
- If an item's zone road is blocked, either send the route-clearance team first or say the team must approach on foot; cite the zone ID.
- Every task must cite evidence IDs that appear in the input (building IDs like F03-B02, zone IDs like F03-Z1, radio IDs like R1).
  Cite the target ID itself. Cite a radio ID for any claim about people, a zone ID for any claim about roads, and an item graded
  destroyed/major for any claim about collapse or damage. Never invent IDs, counts or facts.
- Do not mention people, survivors or casualties for a target unless you cite a radio ID for it.
- People counts from radio are unconfirmed unless marked confirmed: say "reported", never state them as fact.
- Items flagged for review should be verified, not assumed.
- Each team gets 2 to 4 tasks in order. Do not give the same target to two teams.

Teams: {teams}

Input (JSON): {data}

Reply ONLY JSON:
{{"assignments":[{{"team":"T1|T2|T3","tasks":[{{"target":"<ID>","action":"search_rescue|medical|assess|clear_route|verify",
"why":"<=25 words","evidence":["<ID>", "..."]}}]}}],
"holds":[{{"target":"<ID>","why":"<=20 words","evidence":["<ID>"]}}],
"summary":"<=45 words overall intent","summary_evidence":["<ID>"]}}"""

CLAIMS = [  # (pattern in the claim text, test that at least one cited item supports it, rejection reason)
    (re.compile(r"trapped|people|person|casualt|injur|victim|survivor", re.I), lambda ev: any(e["kind"] == "radio" for e in ev), "claim about people cites no radio report"),
    (re.compile(r"road|route|block|impassab|access|bypass|on foot", re.I), lambda ev: any(e["kind"] == "zone" and (e["road_blocked"] or e["flooded_road"]) for e in ev) or any(e["kind"] == "zone" for e in ev), "claim about roads cites no zone"),
    (re.compile(r"collaps|destroy|flatten|major damage|severe", re.I), lambda ev: any(e.get("grade") in ("destroyed", "major") or (e["kind"] == "zone" and e["destroyed_present"]) for e in ev), "claim about damage cites no item graded major/destroyed"),
]

def evidence_registry(frames, radio):
    reg = {}
    for f in frames:
        for z in f["zones"]: reg[z["id"]] = dict(z, kind="zone", frame=f["fid"])
        for b in f["buildings"]: reg[b["id"]] = dict(kind="building", grade=b["grade"], zone=b["zone"], frame=f["fid"])
    for r in radio: reg[r["id"]] = dict(kind="radio", frame=r.get("frame"), confirmed=r["confirmed"],
                                        people_trapped=r.get("people_trapped"), injured=r.get("injured"))
    return reg

def commander_input(queue, radio, frames, top=18):
    zones = {z["id"]: z for f in frames for z in f["zones"]}
    q = []
    for it in queue[:top]:
        q.append({k: it[k] for k in ("rank", "id", "kind", "sector", "grade", "score", "access", "access_evidence", "radio") if it.get(k) is not None}
                 | {"review_flag": bool(it["review"])})
    blocked = [{"zone": z["id"], "road": "flooded" if z["flooded_road"] else "blocked"} for z in zones.values() if z["road_blocked"] or z["flooded_road"]]
    rad = [{"id": r["id"], "sector": r.get("sector"), "people_trapped_reported": r.get("people_trapped"), "injured_reported": r.get("injured"),
            "need": r.get("need"), "confirmed": r["confirmed"], "source_quote": r.get("people_trapped_quote")} for r in radio]
    return {"frames_in_flight_order": [f"{f['fid']}={f['sector']}" for f in frames], "priority_queue": q, "radio_reports": rad, "blocked_roads": blocked}

def validate_plan(plan, reg, queue):
    """Check every claim against the evidence registry. Returns (clean_plan, issues). Unsupported tasks are rejected, not repaired."""
    issues = []; seen = {}; teams = {t["team"] for t in TEAMS}; out = {"assignments": [], "holds": [], "summary": "", "summary_evidence": []}
    if not isinstance(plan, dict) or not isinstance(plan.get("assignments"), list):
        return None, ["plan is not valid JSON of the required shape"]
    def check(task, where):
        tgt = task.get("target"); evid = [e for e in (task.get("evidence") or []) if isinstance(e, str)]; why = str(task.get("why", ""))
        probs = []
        if tgt not in reg: probs.append(f"target {tgt} does not exist")
        bad = [e for e in evid if e not in reg]
        if bad: probs.append(f"cites unknown evidence {bad}")
        if not evid: probs.append("cites no evidence")
        if tgt in reg and tgt not in evid: evid = [tgt] + evid        # the target is always implicit evidence
        ev = [reg[e] for e in evid if e in reg]
        for pat, ok, reason in CLAIMS:
            if pat.search(why) and not ok(ev): probs.append(reason)
        counts = {str(reg[e].get(k)) for e in evid if e in reg and reg[e]["kind"] == "radio" for k in ("people_trapped", "injured")}
        for n in re.findall(r"\b(\d+)\s+(?:reported\s+)?(?:people|persons|trapped|injured|victims|casualties)", why):
            if n not in counts: probs.append(f"count {n} does not match any cited radio report")
        return dict(task, evidence=evid, status="rejected" if probs else "ok", problems=probs)
    for a in plan["assignments"]:
        team = a.get("team")
        if team not in teams: issues.append(f"unknown team {team}"); continue
        tasks = []
        for t in a.get("tasks") or []:
            c = check(t, team)
            if c.get("target") in seen and c["status"] == "ok":
                c["status"] = "rejected"; c["problems"].append(f"target already assigned to {seen[c['target']]}")
            if c["status"] == "ok": seen[c["target"]] = team
            else: issues.append(f"{team} task on {t.get('target')} rejected: {'; '.join(c['problems'])}")
            tasks.append(c)
        out["assignments"].append({"team": team, "tasks": tasks})
    for h in plan.get("holds") or []:
        c = check(h, "hold"); out["holds"].append(c)
        if c["status"] != "ok": issues.append(f"hold on {h.get('target')} rejected: {'; '.join(c['problems'])}")
    out["summary"] = str(plan.get("summary", ""))[:400]
    se = [e for e in plan.get("summary_evidence") or [] if isinstance(e, str)]
    bad = [e for e in se if e not in reg]
    if bad: issues.append(f"summary cites unknown evidence {bad}")
    out["summary_evidence"] = [e for e in se if e in reg]
    covered = set(seen) | {h["target"] for h in out["holds"] if h["status"] == "ok"}
    for it in queue[:5]:
        if it["id"] not in covered: issues.append(f"top-5 item {it['id']} (rank {it['rank']}) not assigned or held: human must decide")
    return out, issues

def fallback_plan(queue):
    """Rule-based plan used only if Wonder fails or returns an invalid plan. Plain code, clearly labelled."""
    out = {"assignments": [{"team": t["team"], "tasks": []} for t in TEAMS], "holds": [],
           "summary": "Rule-based fallback: highest scores first; route team takes blocked-access items.", "summary_evidence": []}
    by = {a["team"]: a for a in out["assignments"]}
    for it in queue[:9]:
        team = "T3" if it.get("access") == "blocked" and len(by["T3"]["tasks"]) < 3 else ("T1" if it["kind"] in ("zone", "radio") or it["grade"] == "destroyed" else "T2")
        if len(by[team]["tasks"]) >= 3: team = min(by, key=lambda k: len(by[k]["tasks"]))
        ev = list(dict.fromkeys(it["evidence"] + ([it["access_evidence"]] if it.get("access_evidence") else [])))
        by[team]["tasks"].append(dict(target=it["id"], action="clear_route" if team == "T3" else "search_rescue", why="; ".join(it["why"])[:140], evidence=ev, status="ok", problems=[]))
    return out

def run_commander(queue, radio, frames, model="muse-wonder-1.0"):
    reg = evidence_registry(frames, radio); data = commander_input(queue, radio, frames)
    prompt = COMMANDER_PROMPT.format(teams=json.dumps(TEAMS), data=json.dumps(data, separators=(",", ":")))
    t0 = time.time(); raw, usage = meta.chat_json(prompt, [], model, retries=2, tag="wonder:commander", timeout=300)
    plan, issues = (None, [f"model error: {raw['error']}"]) if "error" in raw else validate_plan(raw, reg, queue)
    ok_tasks = sum(t["status"] == "ok" for a in (plan or {}).get("assignments", []) for t in a["tasks"])
    source = "muse-wonder-1.0"
    if plan is None or ok_tasks == 0:
        plan = fallback_plan(queue); source = "rule-based fallback (Wonder output invalid)"
    rejected = sum(t["status"] != "ok" for a in plan["assignments"] for t in a["tasks"])
    return dict(source=source, plan=plan, issues=issues, input=data, raw=raw, seconds=round(time.time() - t0, 1),
                usage=usage, tasks_ok=sum(t["status"] == "ok" for a in plan["assignments"] for t in a["tasks"]), tasks_rejected=rejected, teams=TEAMS)
