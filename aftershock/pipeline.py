"""Aftershock survey pipeline.
Perception is done ONLY by frozen, validated prompts and SAM. Orchestration is plain code.
Output is a ranked list of grid cells (triage priority) with evidence and review flags, never a certified damage assessment."""
import concurrent.futures as cf, time
import numpy as np
from PIL import Image
from . import meta, prompts as P

SEVERE = {"major", "destroyed"}
RANK = {"no_damage": 0, "minor": 1, "major": 2, "destroyed": 3}

def grid(img, rows, cols):
    W, H = img.size
    return [dict(r=r, c=c, box=(c * W // cols, r * H // rows, (c + 1) * W // cols, (r + 1) * H // rows))
            for r in range(rows) for c in range(cols)]

def crop_ctx(img, box, margin=0.25):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1; W, H = img.size
    return img.crop((max(0, int(x1 - w * margin)), max(0, int(y1 - h * margin)), min(W, int(x2 + w * margin)), min(H, int(y2 + h * margin))))

def grade_building(img, b, wonder_tiebreak=True):
    c = crop_ctx(img, b["box"])
    with cf.ThreadPoolExecutor(2) as ex:
        fa = ex.submit(meta.chat_json, P.GRADE_A, [c], "muse-spark-1.3", 768)
        fb = ex.submit(meta.chat_json, P.GRADE_B, [c], "muse-spark-1.3", 768)
        a, _ = fa.result(); bb, _ = fb.result()
    ga, gb = a.get("grade"), bb.get("grade")
    votes = [g for g in (ga, gb) if g in RANK]
    disagree = len(votes) == 2 and ((ga in SEVERE) != (gb in SEVERE))
    gw = None
    if disagree and wonder_tiebreak:
        w, _ = meta.chat_json(P.GRADE_A, [c], "muse-wonder-1.0", 768); gw = w.get("grade")
        if gw in RANK: votes.append(gw)
    severe_votes = sum(v in SEVERE for v in votes)
    grade = max(votes, key=lambda g: RANK[g]) if votes else "unknown"   # recall-oriented: worst plausible grade
    reasons = []
    if disagree: reasons.append("assessors disagree on severity")
    w_px = b["box"][2] - b["box"][0]; h_px = b["box"][3] - b["box"][1]
    if min(w_px, h_px) < 80: reasons.append("building small in frame")
    if not votes: reasons.append("no valid assessment")
    return dict(box=[round(v) for v in b["box"]], sam_conf=round(b["conf"], 3), grade=grade,
                grades={"spark_a": ga, "spark_b": gb, "wonder": gw}, severe_votes=f"{severe_votes}/{len(votes)}",
                evidence=a.get("evidence") or bb.get("evidence") or "", review=bool(reasons), review_reasons=reasons)

def survey(img, rows=2, cols=2, hazard="auto", progress=lambda *a: None):
    t0 = time.time(); img = img.convert("RGB"); cells = grid(img, rows, cols)
    calls = {"sam": 0, "spark": 0, "wonder": 0}

    def scan(cell):
        tile = img.crop(cell["box"])
        with cf.ThreadPoolExecutor(4) as ex:
            fs = {"destroyed": ex.submit(meta.chat_json, P.TILE_DESTROYED, [tile]),
                  "road": ex.submit(meta.chat_json, P.TILE_ROAD, [tile])}
            if hazard in ("auto", "flood"): fs["flood"] = ex.submit(meta.chat_json, P.TILE_FLOOD, [tile])
            fsam = ex.submit(meta.segment, tile, "building")
            res = {k: f.result()[0] for k, f in fs.items()}; blds = fsam.result()
        x0, y0 = cell["box"][:2]
        for b in blds: b["box"] = [b["box"][0] + x0, b["box"][1] + y0, b["box"][2] + x0, b["box"][3] + y0]
        cell.update(scan=res, sam_n=len(blds), sam_capped=len(blds) >= 16, _tile_blds=blds, _blds=[])
        progress("scanned", cell["r"], cell["c"])
        return cell
    with cf.ThreadPoolExecutor(6) as ex:
        fwhole = ex.submit(meta.segment, img, "building")   # whole-frame pass keeps buildings that straddle tile edges intact
        cells = list(ex.map(scan, cells)); whole = fwhole.result()
    calls["sam"] += len(cells) + 1; calls["spark"] += len(cells) * (3 if hazard in ("auto", "flood") else 2)
    # merge: whole-frame detections first, then tile detections not already covered (dedupe by overlap of the smaller box)
    def inter(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1]); return max(0, w) * max(0, h)
    def area(a): return (a[2] - a[0]) * (a[3] - a[1])
    merged = list(whole)
    for cell in cells:
        for b in cell["_tile_blds"]:
            if all(inter(b["box"], m["box"]) / max(1, min(area(b["box"]), area(m["box"]))) < 0.5 for m in merged): merged.append(b)
        cell.pop("_tile_blds")
    for b in merged:
        cx, cy = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
        for cell in cells:
            x1, y1, x2, y2 = cell["box"]
            if x1 <= cx < x2 and y1 <= cy < y2: cell["_blds"].append(b); break

    jobs = [(cell, b) for cell in cells for b in cell["_blds"]]
    with cf.ThreadPoolExecutor(12) as ex:
        graded = list(ex.map(lambda cb: (cb[0], cb[1], grade_building(img, cb[1])), jobs))
    masks = []
    for cell, b, g in graded:
        cell.setdefault("buildings", []).append(g); masks.append((b, g["grade"]))
        calls["spark"] += 2; calls["wonder"] += g["grades"]["wonder"] is not None
    for cell in cells:
        cell.setdefault("buildings", []); s = cell["scan"]; bl = cell["buildings"]
        n_sev = sum(b["grade"] in SEVERE for b in bl); n_rev = sum(b["review"] for b in bl)
        destroyed_area = bool(s["destroyed"].get("destroyed_present"))
        road_blocked = bool(s["road"].get("blocked"))
        fl = s.get("flood", {}); flooded_road = bool(fl.get("flooded_road")); flooded_bld = bool(fl.get("flooded_building"))
        # Transparent priority score: a ranking aid, not a probability.
        score = 3 * destroyed_area + 2 * n_sev + 2 * flooded_bld + 1 * (road_blocked or flooded_road)
        reasons = []
        if destroyed_area: reasons.append("destroyed structure seen in area scan")
        if n_sev: reasons.append(f"{n_sev} building(s) graded major/destroyed")
        if flooded_bld: reasons.append("building(s) in floodwater")
        if road_blocked: reasons.append("road likely impassable")
        if flooded_road: reasons.append("road under floodwater")
        review = []
        if n_rev: review.append(f"{n_rev} building assessment(s) need a human look")
        if destroyed_area and n_sev == 0: review.append("area scan sees destruction the building grader did not (SAM misses flattened buildings)")
        if cell["sam_capped"]: review.append("dense area: segmentation hit its 16-object limit")
        evid = [s["destroyed"].get("evidence"), s["road"].get("evidence"), fl.get("evidence")]
        cell.update(score=score, reasons=reasons, review=review, access="blocked" if (road_blocked or flooded_road) else "open",
                    evidence=[e for e in evid if e])
        cell.pop("_blds", None)
    ranked = sorted(cells, key=lambda c: (-c["score"], c["r"], c["c"]))
    for i, c in enumerate(ranked, 1): c["rank"] = i
    return dict(size=img.size, grid=[rows, cols], cells=ranked, calls=calls, seconds=round(time.time() - t0, 1),
                prompts_version=P.VERSION), masks
