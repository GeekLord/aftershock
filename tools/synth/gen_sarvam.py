"""Synthetic radio-call generator for the Aftershock demo. Scripts are fictional. Output is AI-generated speech
and must stay labelled as such (see missions/michael-sortie/radio/LABEL.md). Run from the folder you want the files in."""
import json,base64,subprocess,sys,requests,os
CALLS=[ # (out, lang_code, model, speaker, text)
 ("R1","en-IN","bulbul:v3","aditya","Base, this is Unit two at Sector C1. Two people trapped under a collapsed house, one of them injured. The street is covered in debris. We need a crane and medics."),
]
log=[]
for out,lc,model,spk,text in CALLS:
  if os.path.exists(f"{out}.wav"): continue
  r=requests.post("https://api.sarvam.ai/text-to-speech",json=dict(text=text,language_code=lc,model=model,speaker=spk,pace=1.1,temperature=0.8,speech_sample_rate=24000,output_audio_codec="wav"),headers={"api-subscription-key":os.environ["SARVAM_API_KEY"]},timeout=120)
  if r.status_code!=200: print(out,r.status_code,r.text[:300]); sys.exit(1)
  raw=f"{out}_raw.wav"; open(raw,"wb").write(b"".join(base64.b64decode(a) for a in r.json()["audios"]))
  clean=f"{out}_clean.wav"
  subprocess.run(["ffmpeg","-y","-loglevel","error","-i",raw,"-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1",clean],check=True); os.remove(raw)
  subprocess.run(["ffmpeg","-y","-loglevel","error","-i",clean,"-f","lavfi","-i","anoisesrc=color=pink:amplitude=0.12:sample_rate=24000:seed=2",
   "-filter_complex","[0]highpass=f=300,lowpass=f=3000,volume=2.2,alimiter=limit=0.6[a];[a][1]amix=inputs=2:duration=first:weights=1 1[m]",
   "-map","[m]","-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1",f"{out}.wav"],check=True)
  log.append(dict(id=out,language_code=lc,model=model,speaker=spk,pace=1.1,temperature=0.8,chars=len(text),text=text,request_id=r.json().get("request_id")))
  print(out,"ok",len(text))
json.dump(log,open("generation_log.json","a" if os.path.exists("generation_log.json") else "w"),indent=1,ensure_ascii=False)
