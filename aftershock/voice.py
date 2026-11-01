"""Radio-call intake: Muse Voice Transcribe + Spark extraction with source quotes.
Every count must be confirmed by a human against its quote (T9: a lost sentence boundary turned 'team nine. ten people' into 19)."""
import re, subprocess, tempfile, os
from . import meta, prompts as P
NUM_WORDS = re.compile(r"\b(nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\d+)\b", re.I)

def to_wav(path):
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1", out], check=True)
    return out

def intake(audio_path, languages=None):
    wav = to_wav(audio_path)
    try: tr = meta.transcribe(wav, languages)
    finally: os.remove(wav)
    text = tr.get("transcript", "")
    o, _ = meta.chat_json(P.RADIO_EXTRACT + text)
    flags = []
    for k in ("people_trapped", "injured"):
        q = (o.get(k + "_quote") or "").strip()
        if o.get(k) is not None and (not q or q not in text): flags.append(f"{k}: quote not found verbatim in transcript")
    if len(NUM_WORDS.findall(o.get("english", ""))) >= 2 or re.search(r"\d+\s*[\.,]?\s*\d+", text): flags.append("adjacent numbers: confirm counts by read-back")
    flags.append("confirm all counts by read-back before dispatch")
    return dict(transcript=text, duration_ms=tr.get("audioDurationMs"), extracted=o, confirm=flags)
