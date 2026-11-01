# Mission mode: demo mission record (built 2026-11-01 IST)

## Mission
- 12 RescueNet v1.0 test frames (Hurricane Michael, CC0), consecutive frame numbers 11914–11979, treated as one sortie. Sectors A1–C4 are labels in capture order, not map positions.
- Survey: 2×2 zones per frame, SAM 3.1 on each zone + whole frame, Spark scan (destroyed / road / flood) per zone, Spark A + B per building, Wonder tie-break on disagreement. Per-frame grading cap 14 buildings (not reached; max 9).
- Result: **44 SAM buildings** (2 destroyed, 18 major, 13 minor, 11 no damage), 20 severe, 6 flagged for review, 5 Wonder tie-breaks; **36/48 zones** with destruction in the area scan; **25/48 zones** with a blocked or flooded road. Queue: 45 items (buildings, zone-only detections, 3 radio cards).
- Radio: R1 English "Sector C1, two trapped, one injured" → attached to F09. R2 Spanish "sector A2" transcribed as "sector a a dos" → attached to F02 with an *inexact match* flag. R3 Arabic (earlier sample, mosque on Nile Street) → no sector match, held for a human.

## Top of the queue (precomputed, before any confirmation)
| Rank | ID | What | Score | Access |
|---|---|---|---|---|
| 1 | F02-B01 | destroyed (Spark A minor, B destroyed, Wonder destroyed) ⚑, + R2 | 10 | blocked (F02-Z1) |
| 2 | F02-B02 | major (A and B agree), + R2 | 8 | blocked (F02-Z2) |
| 3 | F02-B03 | major (A minor, B major, Wonder no_damage; worst plausible kept) ⚑, + R2 | 8 | blocked (F02-Z2) |
| 4 | F02-Z3 | area scan: destroyed, no SAM building ⚑, + R2 | 8 | open |
| 5 | F02-Z4 | area scan: destroyed, no SAM building ⚑, + R2 | 8 | open |
Confirming R1 by read-back lifts F09's items to 10 (tested via `POST /api/radio/R1/confirm`).

## Team plan (Muse Wonder 1.0, 73.8 s, validated: 10 tasks grounded, 0 rejected, 0 issues)
- T1 heavy rescue: F02-B01 (search, R2, approach on foot: F02-Z1) → F09-B01 (search, R1, crane) → F11-B02 (verify destroyed, F11-Z2 blocked)
- T2 search + medical: F02-B02 → F02-B03 (verify) → F03-B01
- T3 route + recon: verify F02-Z3, F02-Z4, F09-Z3; clear route F09-Z2
- Held: F09-Z4 (lower priority, blocked), R3 (no sector)
Two later Wonder re-plans (live run on F09; replan-only via curl) gave 12 and 11 tasks, again 0 rejected.

## After-the-fact label check (labels never shown to models; n=12, anecdotal)
- Frames whose labels contain major/destroyed building pixels: 8/8 flagged by Aftershock.
- Frames whose labels contain none: 3/4 still flagged (area scan sees debris as destruction; consistent with T4a precision 70%).

## Cost (list prices, standard mode)
Precomputed mission: 302 calls, **$1.90** (SAM 60 calls $0.15; Spark scans 144 $0.73; Spark A 44 $0.21; Spark B 44 $0.30; Wonder 6 $0.49 incl. commander; Voice 2 + extraction 2 $0.02).
Plus one live run (F09, ~45 calls) and one replan-only run ($0.20), and one aborted commander attempt whose calls were not metered (client timeout at 90 s; up to 2 Wonder calls). Estimated total spend for this build: about **$2.6–3.1**.
