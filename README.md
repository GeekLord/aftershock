# Aftershock

**Find the worst first, with evidence.** Aftershock turns a post-disaster drone sortie and multilingual radio calls into **one mission-level priority queue** and a **draft team plan**, and it says plainly when a person needs to check.

![Mission mode](docs/screenshots/mission_overview.png)

Built for the Meta Global AI Developer Hackathon (2026) on Meta's models:
- **SAM 3.1** segments buildings in every frame (masks shown on the map).
- **Muse Spark 1.3** grades each building twice with two independent frozen prompts (A and B), and scans every zone for destroyed structures, blocked roads and floodwater.
- **Muse Wonder 1.0** does two jobs: breaks ties when Spark A and B disagree on severity, and acts as the **incident-commander planner**. It sees only structured tool outputs (IDs, grades, flags, road access, radio facts), never pixels, and every claim it makes must cite evidence IDs that a validator checks.
- **Muse Voice Transcribe** turns radio calls (25 languages) into text; Spark extracts head-counts *with the exact source words*, so a person can read them back.

## What was built during the hackathon
1. **Survey pipeline** (`aftershock/pipeline.py`): tiling, SAM merge, two-prompt grading, disagreement flag, Wonder tie-break, area scans. Prompts are frozen and validated (`aftershock/prompts.py`, `docs/RESULTS.md`).
2. **Mission mode** (`aftershock/mission.py`, new on branch `mission-mode`):
   - Surveys every frame of a sortie, then builds **one priority queue across frames**. Every building has a stable ID (`F03-B04` = frame 3, building 4), every zone too (`F03-Z2`), and every radio call (`R1`). Each item carries its grade, how the graders agreed, review flags and road-access evidence.
   - Scores are a transparent integer (destroyed 6, major 4, area-scan destruction 5, +1 if the zone scan corroborates, +3 for an unconfirmed trapped-people radio report in the same sector, +5 once a person confirms it). They rank; they are not probabilities.
   - **Radio fusion**: calls become cards in the same queue. If the location names a sector ("Sector C1", or ASR-mangled "sector a a dos") the card attaches to that frame and raises its items' priority. Counts always show their source quote and "needs human confirmation" until someone presses Confirm.
   - **Incident commander**: Muse Wonder drafts assignments for three teams (heavy rescue, search + medical, route clearance). A validator rejects any task that cites an ID that does not exist, claims people without a radio ID, states a count that does not match the cited report, claims collapse without an item graded major/destroyed, claims a road problem without a zone, or double-assigns a target. If no task survives, a rule-based fallback plan is used and labelled as such.
   - **Tamper test** button: injects made-up tasks into Wonder's plan and shows the validator rejecting them (no model call).
3. **Precomputed demo mission** (`missions/michael-sortie/mission.json`) so the demo is instant, plus **Run live**: re-surveys the selected frame and re-plans with Wonder, streaming every model call (SAM, Spark scan, Spark A, Spark B, Wonder) with latency, tokens and cost over Server-Sent Events.
4. **Operations-console UI** (`web/index.html`, vanilla JS, no build step): sortie contact sheet with heat colouring, frame viewer with SAM-mask / box / road / zone toggles, team plan and queue with evidence links that jump to the frame and building, radio strip with transcript, quote, count and Confirm, a persistent honesty banner, and a measured-accuracy panel. The original single-image analyzer is still at `/classic.html`.
5. **Call ledger** (`aftershock/meta.py`): every model call is counted and priced at list rates; a per-process budget cap (`AFTERSHOCK_BUDGET_USD`, default 5) stops calls when reached.

## Why it is built this way
Every design choice comes from a measured result (full details in [docs/RESULTS.md](docs/RESULTS.md)):

| Finding | Design response |
|---|---|
| Exact 4-grade accuracy is 66%, but 95.5% of grades land within one level and severe-damage recall is 90% (n=200) | Rank by severe damage; never claim certified grades |
| Error is 15% when independent assessors agree, 43% when they disagree | Disagreement raises a ⚑ human-review flag |
| The model's own "needs review" flag fired 0/60 times | Review flags come from code, not self-report |
| SAM finds 96% of flooded intact buildings but misses flattened ones (3/13) | Zone scans look for destruction; zone-only detections become their own queue items |
| A planner model that wrote its own vision prompts over-flagged debris | Wonder plans only over structured outputs; perception prompts stay frozen |
| A lost full stop turned "team nine. ten people" into "nineteen" | Counts carry their source quote and must be read back |

## Run it
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export MODEL_API_KEY=...          # Meta Model API key (only needed for Run live / uploads)
uvicorn server:app --port 8000    # open http://localhost:8000
```
The precomputed mission loads without a key. To rebuild it (about $2 of API calls at list prices):
```bash
python tools/build_mission.py manifest
python tools/build_mission.py frame F01 F02 F03 F04 F05 F06 F07 F08 F09 F10 F11 F12
python tools/build_mission.py radio
python tools/build_mission.py assemble
```
`RESCUENET_DIR` must point at the RescueNet v1.0 test images and labels. Offline tests: `python tests/test_mission.py`.

API: `GET /api/mission`, `POST /api/radio/{id}/confirm`, `POST /api/live {fid, replan_only}`, `GET /api/live/{job}/events` (SSE), `POST /api/validate {plan}`, `GET /api/cost`, plus the original `/api/survey` and `/api/voice`.

## Layout
- `aftershock/meta.py`: Meta Model API clients (chat, SAM, ASR) and the call ledger
- `aftershock/prompts.py`: frozen, validated prompts, each with its validation result
- `aftershock/pipeline.py`: per-frame survey engine
- `aftershock/mission.py`: mission queue, radio fusion, Wonder commander and plan validator
- `aftershock/voice.py`: radio-call intake with quote-grounded extraction
- `tools/build_mission.py`: precomputes the demo mission
- `missions/michael-sortie/`: cached demo mission (frames, masks, radio audio, `mission.json`)
- `server.py`, `web/`: API and UI
- `eval/`, `tests/`: evaluation scripts behind every number, and offline tests
- `docs/`: results, mission-mode notes, screenshots

## Data sources and licences
- **RescueNet v1.0** (Hurricane Michael, 2018): Rahnemoonfar, Chowdhury, Murphy, *RescueNet: a high resolution UAV semantic segmentation dataset for natural disaster damage assessment*, Scientific Data 10, 913 (2023). Figshare collection 6647354, https://doi.org/10.6084/m9.figshare.c.6647354, **CC0 1.0**. The demo mission uses 12 test-set frames (11914, 11916, 11918, 11924, 11939, 11943, 11946, 11955, 11959, 11961, 11965, 11979), resized. Their pixel labels are used only for an after-the-fact check and are never shown to a model.
- **FloodNet** (Hurricane Harvey): Rahnemoonfar et al., IEEE Access 2021, BinaLab, **CDLA-Permissive-1.0**. Used in evaluation and one sample.
- Radio calls in the demo are **synthetic** (neural TTS via edge-tts, degraded to radio quality) and are labelled as such in the UI.

## Limits
Aftershock is a triage aid. It is not a structural assessment, does not make rescue decisions, and its scores are rankings, not probabilities. Sector names in the demo are labels in capture order, not georeferenced positions. Results are zero-shot on public datasets the models may have seen in training. The area scan over-flags debris (precision 70%); in the demo it flagged 3 of the 4 frames whose labels show no major/destroyed buildings. Re-running the same frame can change a grade (seen live on F09-B01). Voice results use synthetic speech; real radio will be harder.

## License
MIT (code). Data under the licences above.
