"""Frozen, validated perception prompts. Do not edit without re-running eval/ on held-out data.
Each prompt lists the evaluation it was validated in (see docs/RESULTS.md)."""
VERSION = "2026-11-01"

# Validated: T1 (60 crops), T5 (200 crops): severe recall 90%, precision 76%, within-one-level 95.5%.
GRADE_A = ("You are a FEMA-trained damage assessor. This is a post-hurricane drone photo cropped around ONE building (the building centred in the frame). "
 "Grade that building on this scale: no_damage = roof and walls intact; minor = some shingles/roof covering lost, small debris, structure sound; "
 "major = significant roof loss exposing decking/rafters, partial wall failure, or structure shifted; destroyed = collapsed, flattened, or only slab/pilings remain. "
 "Look carefully before deciding; most standing houses are not destroyed. Reply ONLY JSON: "
 '{"grade":"no_damage|minor|major|destroyed","evidence":"<=25 words of visible evidence"}')

# Validated: T5 second opinion (structured checklist): severe recall 95%, precision 73%. Used for disagreement.
GRADE_B = 'Drone image, post-hurricane, centred on a single structure. First inspect: (1) roof covering - fraction missing; (2) roof structure - any exposed rafters or holes; (3) walls - standing, breached or gone; (4) footprint - is the structure still on its foundation. Then map to the Joint Damage Scale: no_damage (no visible change), minor (cosmetic: missing shingles, small tarps, <~25% covering lost), major (structural: large roof sections gone, exposed decking or rafters, wall breaches, shifted), destroyed (collapsed, flattened, swept off foundation, only slab or pilings). Reply ONLY JSON: {"roof_covering_missing_pct":0-100,"exposed_structure":true|false,"walls":"intact|breached|gone","on_foundation":true|false,"grade":"no_damage|minor|major|destroyed","evidence":"<=25 words"}'

# Validated: T4a (60 tiles): recall 93%, precision 70%. SAM misses destroyed buildings, so this tile scan is required.
TILE_DESTROYED = ("Post-hurricane drone image tile. Does it contain a DESTROYED building: collapsed, flattened into a debris pile, or only a slab/pilings left where a house stood? "
 "Standing houses with roof damage do NOT count. Reply ONLY JSON: {\"destroyed_present\":true|false,\"evidence\":\"<=20 words\"}")

# Validated: T4b (40 images): recall 90%, precision 72%.
TILE_ROAD = ("Post-disaster drone image. Roads may be BURIED under sand, debris, or wreckage so that the pavement is no longer visible; infer where roads run from driveways, "
 "rows of lots and utility lines. Is any road here impassable to a rescue vehicle (buried, covered by debris, trees, or water)? A buried/invisible road counts as blocked. "
 'Reply ONLY JSON: {"blocked":true|false,"evidence":"<=20 words"}')

# Validated: T8 FloodNet (60 images): flooded-road recall 88%, precision 100%; flooded-building recall 58% (definition-strict), precision 93%.
TILE_FLOOD = ("Post-disaster drone photo. Answer two questions about FLOODWATER (standing water where it should not be, not pools, ponds or rivers in their normal banks). "
 "1) Is any building standing in or surrounded by floodwater? 2) Is any road covered by floodwater? Reply ONLY JSON "
 '{"flooded_building":true|false,"flooded_road":true|false,"evidence":"<=20 words"}')

# Validated: T9 (8 languages, synthetic radio noise): 23-24/24 fields. Quotes are required so counts can be confirmed.
RADIO_EXTRACT = ("A rescue radio call transcript follows (any language). Extract facts in ENGLISH. For every number, copy the exact words from the transcript that it came from. "
 "Reply ONLY JSON {\"people_trapped\":int|null,\"people_trapped_quote\":\"exact source words\",\"injured\":int|null,\"injured_quote\":\"exact source words\","
 "\"location\":\"place as said\",\"need\":\"boat|helicopter|ambulance|crane|water|food|medical|other\",\"english\":\"faithful translation\"}. Transcript: ")
