# Aftershock de-risk test, 1 Nov 2026 ~05:00 IST
Clips (5 fps, 10 s, 854x480, 50 frames each): Taiwan 2016 collapse drone (CC BY 3.0, Wikimedia), Irma St Maarten helicopter (CC BY-SA 4.0 per API; page says CC0).
SAM 3.1 (concept prompts, ~4-6 s per request, billed per frame, min 50):
- Taiwan "building": 16 objects, all 16 tracked on >=40/50 frames, ids stable while the camera orbits; the tilted collapsed tower is tracked as a building. Mean conf 0.82.
- Taiwan "vehicle": 16 objects, 14 tracked >=40 frames (fire engines, ladder trucks).
- Taiwan "person": only 3 found (rescuers too small at 480p). Weak, as expected.
- "collapsed building", "rubble", "fire truck": 0 matches. SAM finds objects; it does not judge damage.
- Every busy prompt returned exactly 16 objects, so it looks like a 16-objects-per-request cap. Work around it with tiles/crops or box prompts.
- Irma: buildings and boats found, but ids reset at helicopter scene cuts (expected).
Muse Spark 1.3 grading, 24 RescueNet_BDA scene images (USF-IAE on HF, 8 per class; labels look scene-level and noisy):
- Accuracy 12/24 (chance 8/24). Collapsed recall 8/8. Damaged 1/8 (5 called collapsed, 2 intact). Intact 3/8.
- Confidence was always 0.85-0.99, so it is not calibrated. Don't show raw confidence.
- About 2.4K tokens per image (716 reasoning).
Verdict: tracking passes Muse's rule (buildings + vehicles stable). Grading is the real risk. Next: per-building crops (SAM box, then crop, then Spark) scored against RescueNet pixel-level building-damage masks; triage-first framing (flag collapsed and needs-check).

# Extended tests, 1 Nov 2026 ~05:00 IST, RescueNet v1.0 test split (pixel labels; classes 3-6 = building no/minor/major/total)
T1 Spark per-building crops (60, 15 per class): 4-class 41/60 (68%); within one level 55/60 (92%); severe (major+destroyed) vs not 48/60 (80%), severe precision 26/34, recall 26/30. Model's own needs_human_review flag never fired (0/60), so build our own uncertainty signal.
T3 SAM 3.1 "building" recall of labelled buildings (12 images): no-damage 4/4, minor 10/10, major 2/5, destroyed 3/13. "damaged building" prompt is worse. SAM misses destroyed buildings, the most important ones.
T4a Fix: Spark 3x3 tile scan for destroyed buildings (60 tiles): recall 28/30, precision 28/40.
T2/T4b Roads (asolodin/rescuenet-roads test; class 8 = road buried/covered, verified visually): naive prompt recall 6/20; prompt that tells Spark roads may be buried: recall 18/20, precision 18/25, acc 31/40.
Architecture this implies: Spark tile scan finds destroyed areas and blocked roads; SAM tracks standing buildings and vehicles; Spark grades each building crop; Wonder ranks; uncertainty via our own logic, not the model's flag.

# Round 3, 1 Nov 2026 05:20-05:55 IST (inspired by Factory Droid 1.0: script mode + Agent Factory self-improvement loop)
T5 200 new building crops (50/class), RescueNet test:
- Spark 1.3 prompt A: 4-class 132/200, within-1 191/200, severe-bin 162/200, severe prec 90/118, recall 90/100.
- Spark 1.3 prompt B (structured checklist): 131, 190, 160, prec 95/130, rec 95/100.
- Wonder 1.0 prompt A: 129, 191, 161, prec 96/131, rec 96/100. Wonder no better than Spark on grading, costs ~4x.
- 3-way vote: no gain (132, prec 91/122, rec 91/100).
- Disagreement as review flag (severe level): all three agree on 172/200; error 26/172 (15%) when they agree vs 12/28 (43%) when they disagree. Usable review trigger.
- "Any model says severe" policy: recall 97/100, flags 141/200.
T6 Script mode: one Wonder request (49 s, ~10k tokens) wrote a 238-line survey program; it ran on 3 full images first try (after fixing my harness), 18-29 tool calls each, 9-19 s, valid JSON. But its self-written vision prompts over-flagged destroyed tiles (debris-heavy images: 4/9, 0/9, 7/9 tile agreement). Lesson: planner should orchestrate our validated primitives, not write its own vision prompts.
T7 Self-improvement loop (Wonder reads dev errors, rewrites prompt, keep if better on dev, test on untouched held-out 100): held-out severe precision 43/57 -> 43/52, recall 43/49 unchanged, severe-bin 80 -> 85, 4-class 65 -> 65. Small n. Iteration 2 did not help. Note: sending Wonder the full list of model evidence strings triggered content_policy_violation (prompt) every time; individual strings were fine; sent truth/pred pairs only.

---
## Round 4 (2026-11-01 IST)

### T8 — Flood generalisation on FloodNet (Hurricane Harvey, different dataset, different labellers)
Data: 60 images from torchgeo/floodnet `Train/Labeled` (30 Flooded, 30 Non-Flooded; seed 5). No training was done, so using the train split is fine for zero-shot evaluation. Ground truth from pixel masks (class 1 building-flooded, 3 road-flooded; >20,000 px = present).
| Task | Recall | Precision | Acc |
|---|---|---|---|
| Image: any flooded building (Spark) | 14/24 = 58% (CI 39–76) | 14/15 = 93% (CI 70–99) | 49/60 |
| Image: any flooded road (Spark) | 22/25 = 88% (CI 70–96) | 22/22 = 100% (CI 85–100) | 57/60 |
| Building crop: flooded? (40 flooded / 40 dry) | 16/40 = 40% (CI 26–55) | 16/19 = 84% | 53/80 |
| SAM "building" recall | flooded 86/90 = 96%, dry 25/25 | — | 7/12 images hit the 16-object cap |

Observations:
1. **Flooded roads generalise very well**; zero false positives. Strongest flood signal for routing.
2. **Spark is conservative on flooded buildings** (high precision, low recall). Visual inspection of false negatives: FloodNet labels a building "flooded" when it sits in a flooded neighbourhood (water in streets/yards), even with no visible water against the walls. Our prompt required water touching the walls. This is largely an **operational-definition mismatch**, not only a model error. Paper point: zero-shot VLM triage is only as good as the alignment between prompt definitions and the dataset's labelling protocol.
3. **SAM behaves oppositely across disasters**: finds flooded-but-intact buildings (96%) but misses destroyed ones (hurricane-wind debris, earlier T3). SAM's "building" concept = intact roof outline. So SAM recall is a function of damage *type*, a useful, citable finding.
4. The 16-object SAM cap is hit in dense suburbs (7/12); tile before segmenting.

### T9 — Multilingual radio notes through Muse Voice Transcribe
8 scripted rescue calls (English, Spanish, Hindi, Tagalog, Bengali, Turkish, French, Arabic), synthesised with neural TTS (edge-tts), then degraded to "radio" (300–3000 Hz band-pass, limiter, pink noise). Transcribed via `/v1/asr/transcribe`, then Spark extracted people count, injured yes/no, need (English).
| Condition | Mean CER | Fields correct |
|---|---|---|
| Clean | 0.052 | 24/24 |
| Radio noise | 0.077 | 24/24 |
| Radio + languageBias | 0.033 | 23/24 |
Observations:
1. Language bias helps transcription (CER 0.077 → 0.033) on noisy audio.
2. **The one error is safety-relevant**: Arabic "الفريق تسعة. عشرة أشخاص" ("team nine. ten people") lost its sentence boundary, and the extractor read "nineteen people". A dropped full stop turned into a wrong head-count. Mitigation for the product: show counts with source span and require confirm/read-back; flag adjacent number words.
3. Tagalog: "Team Lima" kept as a name (lima = five); disfluent duplicates ("ito ang ito ang") in clean audio. Translation of call-signs is ambiguous; preserve the original text beside the English.
4. Caveat: TTS speech is far cleaner than real radio (no accents, crosstalk, clipping bursts, background sirens). Treat as an upper bound. Real recordings needed before any claim.

### T10 — Overlay rendering
`render.py` decodes SAM one-bit masks with `meta-sam-parser` (mask payload is local to `bounds` left/top/right/bottom, in the submitted image's pixel space). Output `overlay_11121.jpg` is crisp and demo-ready. The figure also shows the T3 finding visually: intact roofs get masks; debris fields where houses stood get none. Strong paper figure.

### Statistical note
Wilson 95% CIs added. The T7 prompt-revision gain (severe precision 75.4% → 82.7%) has overlapping CIs (62.9–84.8 vs 70.3–90.6): **not significant at n=100**. Report as preliminary.

### Prototype note (2026-11-01)
With the quote-required extraction prompt (`RADIO_EXTRACT`), the Arabic call that earlier produced "nineteen" returned 10 people with source quote "عشرة" (n=1, anecdotal). Quote grounding is a cheap mitigation worth testing at scale.
Run-to-run variance: re-surveying the same image gave different area-scan answers for one of four tiles. Variance needs measuring before publication.

---
## Round 5: Mission mode (2026-11-01 IST)
Details and the full mission record: [MISSION_MODE.md](MISSION_MODE.md).
- 12-frame RescueNet sortie surveyed end to end: 302 model calls, $1.90 at list prices, 8–76 s per frame with 12 parallel workers.
- Muse Wonder as planner over structured outputs only: 3 runs, 10/12/11 tasks, **0 rejected** by the evidence validator; it cited radio IDs for every people claim and zone IDs for blocked access. Commander latency 45–91 s (9.5k–13.7k output tokens, mostly reasoning), so plans are drafted in the background, not per click.
- Validator unit tests (tests/test_mission.py) confirm rejection of invented IDs, unsupported damage claims, people claims without radio, wrong counts, duplicate targets.
- ASR turned "sector A2" (Spanish TTS through radio noise) into "sector a a dos"; a tolerant matcher recovers A2 but flags it inexact. Location words need the same read-back as counts.
- Frame-level label check (n=12): severe frames 8/8 flagged; 3/4 non-severe frames also flagged (area-scan over-flagging).
- Run-to-run variance seen again: F09-B01 was "A and B agree (major)" in the cached run and "A/B disagree → Wonder tie-break" live.
