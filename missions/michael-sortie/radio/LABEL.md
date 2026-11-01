# Synthetic audio: not real emergency calls

Every audio file in this folder (`R1.wav`, `R2.wav`, `R3.wav`) is **AI-generated speech**. Each was produced by text-to-speech from a scripted, fictional message (the scripts are in `radio.json`), then degraded to sound like a handheld radio (300–3000 Hz band-pass, limiter, pink noise). No real emergency call was recorded or used. The units, people and incidents described are invented.

| Call | Language | Source |
|---|---|---|
| R1 | English | AI-generated with Sarvam Bulbul v3 (Sarvam AI); labelled as AI-generated per Sarvam's Terms of Service §10.2 |
| R2 | Spanish | AI-generated with ElevenLabs (eleven_multilingual_v2, premade voice), free plan, non-commercial; credit: elevenlabs.io |
| R3 | Arabic | AI-generated with ElevenLabs (eleven_multilingual_v2, premade voice), free plan, non-commercial; credit: elevenlabs.io. Same audio as `samples/radio_arabic.wav` |

`samples/radio_english.wav` is AI-generated with Sarvam Bulbul v3 and `samples/radio_arabic.wav` is the ElevenLabs R3 call; both are labelled the same way.

Generation scripts: `tools/synth/` (keys read from `SARVAM_API_KEY` and `ELEVENLABS_API_KEY`). The radio effect uses a fixed pink-noise seed (2) so renders are repeatable.

## Licence
These audio files are **not covered by the repository's MIT licence**, which applies to code only. They remain subject to the terms of the TTS provider that generated them. The transcripts, extractions and model-call records in `radio.json` are outputs of the Aftershock pipeline.
