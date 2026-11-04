# Meta models and API usage in Aftershock

Aftershock uses four Meta models through the Meta Model API (`https://api.meta.ai/v1`, see `aftershock/meta.py`). Each section below says what the model does here, which code calls it, how it is called, which frozen prompts it uses and what we measured. Every number on this page comes from `docs/RESULTS.md`, `docs/MISSION_MODE.md`, or the saved call records of the precomputed mission (`missions/michael-sortie/`).

All prompts live in `aftershock/prompts.py` (version `2026-11-01`) and are frozen: they are not edited without re-running the evaluations in `eval/`. The incident-commander prompt (`COMMANDER_PROMPT`) lives in `aftershock/mission.py`.

## At a glance

| Model | Model id (as sent) | Endpoint | Called from | Job in Aftershock |
|---|---|---|---|---|
| SAM 3.1 | `sam-3.1` | `POST /v1/responses` (streamed) | `meta.segment()` ← `pipeline.survey()` | Segments buildings in each frame and each zone; masks drawn on the map |
| Muse Spark 1.3 | `muse-spark-1.3` | `POST /v1/chat/completions` | `meta.chat_json()` ← `pipeline.grade_building()`, `pipeline.survey()` (`scan`), `voice.intake()` | Grades each building twice (prompts A and B), scans each zone for destruction, blocked roads and floodwater, extracts radio facts with quotes |
| Muse Wonder 1.0 | `muse-wonder-1.0` | `POST /v1/chat/completions` | `meta.chat_json()` ← `pipeline.grade_building()` (tie-break), `mission.run_commander()` | Breaks severity ties between Spark A and B; drafts the team plan from structured outputs only |
| Muse Voice Transcribe | `muse-voice-transcribe-1.0` | `POST /v1/asr/transcribe` (multipart) | `meta.transcribe()` ← `voice.intake()` | Turns radio calls into text |

## SAM 3.1

- **What it does here.** Finds buildings. `pipeline.survey()` runs one whole-frame pass (`tag="sam:frame"`) and one pass per zone tile (`tag="sam:tile"`, 2×2 tiles per frame), then merges them: whole-frame detections first, tile detections added only if they overlap an existing box by less than half of the smaller box. Each kept building is cropped and sent to Spark for grading. The masks are coloured by grade and saved as the mask layer shown in the console.
- **Call.** `meta._segment()` posts to `/v1/responses` with `model: "sam-3.1"`, `stream: true`, `metadata.include_confidence: "true"`, and a user message holding the text prompt `"building"` plus the image as a JPEG data URL (longest side 1600 px). The server-sent event stream is parsed with `meta_sam_parser`; masks are decoded to rasters and rescaled to the original image.
- **Input / output.** In: one image + the noun `building`. Out: a list of `{box, conf, mask}`.
- **Prompt.** The concept noun `building` (no prompt from `prompts.py`).
- **Measured** (`docs/RESULTS.md`):
  - Recall of labelled RescueNet buildings (T3, 12 images): no damage 4/4, minor 10/10, major 2/5, **destroyed 3/13**. SAM finds intact roof outlines and misses flattened buildings, which is why the Spark zone scan exists.
  - FloodNet (T8): flooded buildings 86/90, dry 25/25. 7/12 images hit what looks like a 16-objects-per-request limit, so frames are tiled before segmenting.
  - Asking for "collapsed building" or "rubble" returned 0 objects in the de-risk test.

## Muse Spark 1.3

- **What it does here.**
  1. **Two independent grades per building** (`pipeline.grade_building()`): prompt `GRADE_A` (`tag="spark_a:grade"`) and prompt `GRADE_B` (`tag="spark_b:grade"`) run in parallel on the same crop (building box plus a 25% margin, longest side 768 px). If one says major/destroyed and the other does not, the building is flagged "assessors disagree on severity" and Wonder is asked to break the tie. The kept grade is the worst plausible one among the valid votes.
  2. **Zone scans** (`scan()` inside `pipeline.survey()`): each zone tile gets `TILE_DESTROYED` (`spark_scan:destroyed`), `TILE_ROAD` (`spark_scan:road`) and, in `auto`/`flood` mode, `TILE_FLOOD` (`spark_scan:flood`). Images are sent at longest side 1024 px.
  3. **Radio extraction** (`voice.intake()`): `RADIO_EXTRACT` + the transcript (`spark_extract:radio`) returns counts, location, need and an English translation, with the exact source words for every count.
- **Call.** `meta._chat_json()` posts to `/v1/chat/completions` with `model`, one user message (text + images as data URLs) and `response_format: {"type": "json_object"}`. It retries on HTTP 429/500/502/503 (up to 3 tries by default) and returns `{"error": ...}` on other failures.
- **Frozen prompts.** `GRADE_A`, `GRADE_B`, `TILE_DESTROYED`, `TILE_ROAD`, `TILE_FLOOD`, `RADIO_EXTRACT`.
- **Measured** (`docs/RESULTS.md`):
  - Grading, 200 RescueNet building crops (T5), prompt A: exact 4-class 132/200, within one level 191/200, severe recall 90/100, severe precision 90/118. Prompt B: severe recall 95/100, precision 95/130.
  - Disagreement as a review trigger: error 26/172 (15%) when graders agree vs 12/28 (43%) when they disagree.
  - The model's own "needs review" flag fired 0/60 times (T1), so review flags come from code.
  - Self-reported confidence was always 0.85–0.99 on 24 scene images with accuracy 12/24: not calibrated, not used.
  - Destroyed-structure tile scan (T4a, 60 tiles): recall 28/30, precision 28/40.
  - Blocked-road scan (T4b, 40 images): recall 18/20, precision 18/25.
  - Flood scan, FloodNet (T8, 60 images): flooded road recall 22/25, precision 22/22; flooded building recall 14/24, precision 14/15.
  - Radio extraction (T9, 8 languages, synthetic radio noise): fields correct 24/24 clean, 24/24 noisy, 23/24 noisy + languageBias. The one error read "team nine. ten people" as nineteen.

## Muse Wonder 1.0

- **What it does here.**
  1. **Tie-break** (`pipeline.grade_building()`, `tag="wonder:tiebreak"`): only when Spark A and B disagree on severity. Wonder gets the same crop and the frozen `GRADE_A` prompt; its grade is added as a third vote.
  2. **Incident commander** (`mission.run_commander()`, `tag="wonder:commander"`): drafts assignments for three teams (T1 heavy rescue, T2 search + medical, T3 route clearance). It gets **no images**, only `commander_input()`: the top 18 queue items (IDs, kind, sector, grade, score, road access and its zone, linked radio IDs, review flag), the radio reports (counts marked reported/confirmed, with source quote) and the list of blocked zones. Called with `retries=2`, `timeout=300` s.
- **Call.** Same `chat_json()` / `/v1/chat/completions` path as Spark, with `model: "muse-wonder-1.0"`.
- **Prompts.** `GRADE_A` (tie-break, from `prompts.py`); `COMMANDER_PROMPT` (planner, in `mission.py`).
- **Measured.**
  - As a grader (T5, 200 crops): exact 129/200, within one 191/200, severe recall 96/100, precision 96/131. No better than Spark at about 4× the cost, so it is used only for tie-breaks.
  - As planner (`docs/RESULTS.md`, Round 5): 3 runs gave 10, 12 and 11 tasks with **0 rejected** by the validator. Latency 45–91 s per plan (9.5k–13.7k output tokens, mostly reasoning), so plans are drafted in the background, not per click.
  - Precomputed plan (`missions/michael-sortie/commander.json`): 73.8 s, 1,854 prompt tokens, 13,713 completion tokens of which 13,110 reasoning, 10 tasks accepted, 0 rejected, 0 issues.
  - When Wonder wrote its own vision prompts (T6), it over-flagged debris; this is why Wonder plans over structured outputs and perception prompts stay frozen.

## Muse Voice Transcribe

- **What it does here.** Transcribes radio calls in the mission (`tools/build_mission.py radio`) and in the single-frame view's upload (`POST /api/voice`). `voice.intake()` first normalises the audio with ffmpeg to mono 24 kHz 16-bit WAV.
- **Call.** `meta._transcribe()` posts multipart to `/v1/asr/transcribe`: a `request` JSON part `{"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "languageBias": [...]}` (bias only when languages are given) and an `audio` part. Up to 3 tries.
- **Input / output.** In: WAV audio (+ optional language list). Out: `transcript`, `audioDurationMs`. The transcript then goes to Spark with `RADIO_EXTRACT`.
- **Code checks after extraction** (`voice.intake()`): a count whose quote is not found verbatim in the transcript is flagged; adjacent number words ("one two", "nine. ten") raise "confirm counts by read-back"; every card says "needs human confirmation" until a person presses Confirm.
- **Measured** (T9, 8 scripted calls, TTS audio): mean CER 0.052 clean, 0.077 with radio noise, 0.033 with radio noise + languageBias. In the mission, Spanish "sector A2" was heard as "sector A dos" and matched to A2 with an "inexact" flag; English "Sector C1" was heard as "sector C one C one". TTS audio is cleaner than real radio, so these are upper bounds.

## Calls, tokens, latency and cost: precomputed mission

Source: the per-call records saved during the build, i.e. `calls` in each frame of `missions/michael-sortie/mission.json` (copied from `missions/michael-sortie/cache/F01–F12.json`), `intake.calls` in `missions/michael-sortie/radio/radio.json`, and `calls` in `missions/michael-sortie/commander.json`. This is the same set `tools/build_mission.py assemble` sums. Costs are list prices set in `meta.PRICE`, `SAM_PER_IMAGE` and `ASR_PER_HOUR` (pay-as-you-go, standard mode, from https://dev.meta.ai/docs/pricing-rate-limits, checked 2026-11-01): Spark $1.25 / $4.25 per million input / output tokens, Wonder $4.00 / $20.00, SAM $2.50 per 1,000 images, Voice Transcribe $0.18 per audio hour.

| Model | Calls (all OK) | Input tokens | Output tokens | Latency median / mean / max | Cost (list) |
|---|---:|---:|---:|---|---:|
| SAM 3.1 | 60 | – | – | 3.1 s / 11.9 s / 47.9 s | $0.150 |
| Muse Spark 1.3 | 235 | 206,030 | 238,257 | 5.3 s / 6.1 s / 16.9 s | $1.270 |
| Muse Wonder 1.0 | 6 | 4,326 | 23,740 | 22.2 s / 28.8 s / 73.8 s | $0.492 |
| Muse Voice Transcribe | 3 | – | – | 2.5 s / 2.5 s / 2.9 s | $0.0015 |
| **Total** | **304** | | | | **$1.914** |

By job:

| Job (ledger tag) | Model | Calls | In tokens | Out tokens | Median latency | Cost |
|---|---|---:|---:|---:|---:|---:|
| `sam:frame` | SAM 3.1 | 12 | – | – | 6.5 s | $0.030 |
| `sam:tile` | SAM 3.1 | 48 | – | – | 2.6 s | $0.120 |
| `spark_scan:destroyed` | Spark | 48 | 50,256 | 39,467 | 4.9 s | $0.231 |
| `spark_scan:road` | Spark | 48 | 51,600 | 58,795 | 7.5 s | $0.314 |
| `spark_scan:flood` | Spark | 48 | 51,312 | 27,791 | 3.7 s | $0.182 |
| `spark_a:grade` | Spark | 44 | 24,887 | 42,901 | 5.2 s | $0.213 |
| `spark_b:grade` | Spark | 44 | 27,527 | 62,893 | 6.5 s | $0.302 |
| `spark_extract:radio` | Spark | 3 | 448 | 6,410 | 9.6 s | $0.028 |
| `wonder:tiebreak` | Wonder | 5 | 2,472 | 10,027 | 13.2 s | $0.210 |
| `wonder:commander` | Wonder | 1 | 1,854 | 13,713 | 73.8 s | $0.282 |
| `voice` | Voice Transcribe | 3 | – | – | 2.5 s | $0.0015 |

Notes on these numbers:
- Output tokens are the API's `completion_tokens`, which include reasoning tokens (the commander call: 13,110 of 13,713).
- Latency is wall time per call as measured by the client, including time spent waiting in the build's parallel thread pools (up to 12 workers). The large SAM maximum (47.9 s) reflects that contention, not SAM's own speed; the de-risk test saw about 4–6 s per SAM request.
- Per-frame survey time was 8.4–76.4 s.
- The `cost` block stored inside `mission.json` (and shown in the console header) says **302 calls, $1.903**: it counts 2 Voice and 2 extraction calls, while `radio/radio.json` now holds intake records for all three calls (R1–R3). The difference is R3's 2 calls ($0.011). The table above sums the saved records as they are today.
- This covers only the precomputed mission. `docs/MISSION_MODE.md` estimates total spend for the whole build, including one live run, one re-plan and one unmetered aborted commander attempt, at about $2.6–3.1.

## Call ledger and spending caps

- **Ledger** (`aftershock/meta.py`): every call goes through `_start()` / `_end()` and is appended to `LEDGER` with model, tag, latency, input/output tokens, cost and OK flag. Live runs stream these records to the browser over Server-Sent Events (`GET /api/live/{job}/events`). `GET /api/cost` returns this process's spend by job, the budget, the live-run count, and the precomputed mission's cost.
- **Budget cap**: `_start()` raises `BudgetExceeded` before any call once spend reaches `AFTERSHOCK_BUDGET_USD` (code default $5; Docker image default $3). With `AFTERSHOCK_SPEND_FILE` set, spend and the live-run count are written to that file after every call and read back at start-up, so the cap survives restarts and scale-to-zero. Without it, the cap is per process.
- **Live-run caps** (`server.py`, `POST /api/live`): refused if no `MODEL_API_KEY` (503), if a run is already in progress (409), if the run count has reached `AFTERSHOCK_MAX_LIVE` (code default 6; Docker default 3) or the budget is used up (429), or if the same visitor IP started a run within `AFTERSHOCK_IP_COOLDOWN_S` seconds (code default 0 = off; Docker default 600). A live run grades at most `LIVE_MAX_BUILDINGS` (default 8) buildings in the re-surveyed frame; the mission build caps at `MAX_BUILDINGS` (default 14) per frame.
- **Hosted demo** (https://aftershock.busy.one): configured through these environment variables at **$50 total API spend, 200 live runs, one live run per visitor IP every 120 s**. These values are deployment settings, not repository defaults. When a cap is hit the console says so and the cached mission keeps working.

## Plan validator

`mission.validate_plan()` checks every task and hold Wonder proposes against an evidence registry built from the mission (`evidence_registry()`: every zone, building and radio ID). A task is **rejected, not repaired**, if:
- its target ID does not exist, or it cites an evidence ID that does not exist, or it cites no evidence;
- its text mentions people, trapped, casualties, injured, victims or survivors but cites no radio report;
- it mentions roads, routes, blocking, access, bypass or "on foot" but cites no zone;
- it mentions collapse, destroyed, flattened, major damage or severe but cites no item graded major/destroyed (or a zone whose scan saw destruction);
- it states a number of people/trapped/injured that does not match any cited radio report;
- its target is already assigned to another team.

The task's own target always counts as cited evidence. The validator also reports (as issues, not rejections) unknown teams, unknown IDs in the summary, and any top-5 queue item that no team took and no hold covers ("human must decide"). The **Tamper test** button posts a copy of Wonder's plan with three invented tasks to `POST /api/validate` (no model call) and shows them rejected.

## Fallback when Wonder fails

In `mission.run_commander()`, if the Wonder call returns an error, or the validator leaves zero accepted tasks, the plan is replaced by `fallback_plan()`: plain code that takes the top 9 queue items, gives blocked-access items to T3 and the rest to T1/T2, and sets `source` to `"rule-based fallback (Wonder output invalid)"`. The console shows that source in the Team assignments header instead of "drafted by Muse Wonder 1.0", and the model error is kept in `issues`.

An HTTP 504 from Wonder takes this path: `_chat_json()` retries only on 429/500/502/503, so a 504 returns an error straight away and the labelled rule-based plan is used. Note the label says "Wonder output invalid" for both cases (bad output and HTTP error). A `BudgetExceeded` error is different: it is raised before the call and is not caught by `run_commander()`, so a live run that hits the budget ends with an error and the previous plan stays in place.

## Known limitations

- Results are zero-shot on public datasets the models may have seen in training. Sample sizes are small (n=12 to 200), and the T7 prompt-revision gain is not statistically significant.
- SAM misses flattened buildings (destroyed 3/13); zone scans compensate but over-flag debris (tile precision 28/40; 3 of 4 frames with no labelled severe damage were still flagged).
- Spark under-calls flooded buildings (image recall 14/24) and its confidence is not calibrated.
- Wonder planning takes 45–91 s and can time out or fail; the fallback is rule-based, not a model plan.
- Results vary run to run: re-surveying F09 changed F09-B01 from "A and B agree (major)" to a tie-break, and one of four tiles changed its scan answer on a re-survey.
- Radio audio is synthetic TTS, cleaner than real radio. ASR merges or drops sentence boundaries and can turn "one two" into "twelve"; counts and locations need read-back.
- Muse Voice Transcribe pricing in the ledger is computed from `audioDurationMs` at the list rate; a failed transcription is recorded at $0.
