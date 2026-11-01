# Aftershock: candidate paper notes

Working title: "Find the Worst First: A Measured Evaluation of Foundation Vision-Language and Segmentation Models for Zero-Shot Post-Disaster Triage"

## Candidate contributions
1. Zero-shot benchmark of Meta Muse Spark/Wonder + SAM 3.1 on two UAV disaster datasets (RescueNet, Michael; FloodNet, Harvey) with CIs.
2. Ordinal reframing: 4-class accuracy is modest (66%) but within-one-level is 95.5% and severe-recall 90%. Argue triage should be evaluated as ranking/recall, not exact damage class.
3. Inter-model disagreement as an uncertainty signal: error 15% when models agree vs 43% when they disagree (n=200). Native self-reported "needs review" flag fired 0/60.
4. Segmentation recall depends on damage type: SAM finds flooded intact buildings (96%) but misses destroyed ones. Pipelines must pair segmentation with tile-level VLM scans.
5. Planner-invented vision criteria fail: a model-written survey program ran correctly but its self-authored damage prompts over-flagged debris (tile agreement 0/9–7/9). Separate orchestration from validated perception tools.
6. Definition alignment: flood-building recall gap largely comes from label protocol vs prompt definition.
7. Multilingual voice intake: low CER across 8 languages under synthetic noise; one boundary-loss error changed a casualty count, a concrete failure mode.
8. Human-gated prompt self-improvement: dev/held-out split, modest non-significant gain.

## Threats to validity (must state)
- Possible training-data contamination: RescueNet/FloodNet are public; Meta models may have seen them.
- Small n (60–200 per test); single seeds; API model versions may change.
- Crude label thresholds (pixel counts) for image-level truth.
- Synthetic TTS audio, not real radio.
- RescueNet road "blocked" semantics checked visually only.
- No comparison to supervised baselines yet (FloodNet/RescueNet papers report supervised segmentation numbers; cite rather than rerun).

## Next experiments that would make it publishable
- Scale to ≥1,000 crops with fixed seeds; cost/latency table per image.
- Supervised baseline comparison from the dataset papers.
- Third disaster (earthquake, e.g. xBD/Turkey 2023 UAV) for generalisation.
- Real radio recordings (public emergency-comms archives) for T9.
- Prompt definition ablation on FloodNet (strict "touching walls" vs "in flooded area").
- Calibration analysis: disagreement rate vs error curve.
