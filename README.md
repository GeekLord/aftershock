# Aftershock

**Find the worst first, with evidence.** Aftershock turns post-disaster drone imagery and multilingual radio calls into a ranked list of where responders should look first, and it says plainly when a person needs to check.

Built for the Meta Global AI Developer Hackathon (2026) on Meta's models:
- **SAM 3.1** segments buildings (masks shown on the map).
- **Muse Spark 1.3** grades each building and scans each area for destroyed structures, blocked roads and floodwater, using frozen, validated prompts.
- **Muse Wonder 1.0** breaks ties when two independent assessments disagree on severity.
- **Muse Voice Transcribe** turns radio calls (25 languages) into text; Spark extracts head-counts *with the exact source words*, so a person can read them back.

## Why it is built this way
Every design choice comes from a measured result (full details in [docs/RESULTS.md](docs/RESULTS.md)):

| Finding | Design response |
|---|---|
| Exact 4-grade accuracy is 66%, but 95.5% of grades land within one level and severe-damage recall is 90% (n=200) | Rank by severe damage; never claim certified grades |
| Error is 15% when independent assessors agree, 43% when they disagree | Disagreement raises a ⚑ human-review flag |
| The model's own "needs review" flag fired 0/60 times | Review flags come from code, not self-report |
| SAM finds 96% of flooded intact buildings but misses flattened ones | A separate area scan looks for destruction; conflicts are flagged |
| A planner model that wrote its own vision prompts over-flagged debris | Perception prompts are frozen in `aftershock/prompts.py`; orchestration is plain code |
| A lost full stop turned "team nine. ten people" into "nineteen" | Counts carry their source quote and must be read back |

## Run it
```bash
pip install -r requirements.txt
export MODEL_API_KEY=...        # Meta Model API key
uvicorn server:app --port 8000  # open http://localhost:8000
```
Sample results load without a key (pick one from "Sample surveys").

## Layout
- `aftershock/meta.py`: Meta Model API clients (chat, SAM, ASR)
- `aftershock/prompts.py`: frozen, validated prompts, each with its validation result
- `aftershock/pipeline.py`: survey engine (tiling, SAM merge, grading, disagreement, ranking)
- `aftershock/voice.py`: radio-call intake with quote-grounded extraction
- `aftershock/render.py`: mask and rank overlay
- `server.py`, `web/`: API and single-page UI
- `eval/`: the evaluation scripts and raw results behind every number here
- `docs/`: results and research notes

## Limits
Aftershock is a triage aid. It is not a structural assessment, does not make rescue decisions, and its scores are rankings, not probabilities. It was tested zero-shot on public datasets (RescueNet, Hurricane Michael; FloodNet, Hurricane Harvey) that the underlying models may have seen during training. Voice results used synthetic speech with added noise; real radio will be harder.

## Data credits
RescueNet (Rahnemoonfar et al., 2023) and FloodNet (Rahnemoonfar et al., IEEE Access 2021, CDLA-Permissive-1.0). Sample images in `samples/` come from these datasets.

## License
MIT
