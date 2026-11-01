"""Synthetic radio-call generator for the Aftershock demo. Scripts are fictional. Output is AI-generated speech
and must stay labelled as such (see missions/michael-sortie/radio/LABEL.md). Run from the folder you want the files in."""
import json,base64,subprocess,requests,sys,os
text="Unit four to base. Three people trapped on the roof of the blue house on Maple Street, one elderly man injured, water rising fast, need a boat."
r=requests.post("https://api.sarvam.ai/text-to-speech",json=dict(text=text,language_code="en-IN",model="bulbul:v3",speaker="aditya",pace=1.1,temperature=0.8,speech_sample_rate=24000,output_audio_codec="wav"),headers={"api-subscription-key":os.environ["SARVAM_API_KEY"]},timeout=120)
if r.status_code!=200: print(r.status_code,r.text[:300]); sys.exit(1)
open("en_raw.wav","wb").write(b"".join(base64.b64decode(a) for a in r.json()["audios"]))
subprocess.run(["ffmpeg","-y","-loglevel","error","-i","en_raw.wav","-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1","radio_english_clean.wav"],check=True)
subprocess.run(["ffmpeg","-y","-loglevel","error","-i","radio_english_clean.wav","-f","lavfi","-i","anoisesrc=color=pink:amplitude=0.12:sample_rate=24000:seed=2",
 "-filter_complex","[0]highpass=f=300,lowpass=f=3000,volume=2.2,alimiter=limit=0.6[a];[a][1]amix=inputs=2:duration=first:weights=1 1[m]",
 "-map","[m]","-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1","radio_english.wav"],check=True)
json.dump(dict(id="radio_english",model="bulbul:v3",speaker="aditya",pace=1.1,noise_seed=2,text=text,request_id=r.json().get("request_id")),open("generation_log.json","w"),indent=1)
print("ok")
