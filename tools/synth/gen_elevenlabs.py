"""Synthetic radio-call generator for the Aftershock demo. Scripts are fictional. Output is AI-generated speech
and must stay labelled as such (see missions/michael-sortie/radio/LABEL.md). Run from the folder you want the files in."""
import json,subprocess,requests,os,sys
CALLS=[("R2","es","iP95p4xoKVk53GoZ742B","Base, aquí el equipo cinco en el sector A2. Hay cuatro personas atrapadas en una casa derrumbada, dos heridos. Necesitamos una ambulancia."),
       ("R3","ar","pNInz6obpgDQGcFmaJgB","القاعدة، هنا الفريق تسعة. عشرة أشخاص محاصرون في المسجد في شارع النيل، ثلاثة مصابين. نحتاج إلى سيارة إسعاف.")]
MODEL="eleven_multilingual_v2"; log=[]
for out,lang,vid,text in CALLS:
  r=requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=mp3_44100_128",json=dict(text=text,model_id=MODEL,language_code=lang,seed=7,voice_settings=dict(stability=0.5,similarity_boost=0.75)),headers={"xi-api-key":os.environ["ELEVENLABS_API_KEY"]},timeout=120)
  if r.status_code!=200: print(out,r.status_code,r.text[:400]); sys.exit(1)
  open(out+".mp3","wb").write(r.content)
  clean=out+"_clean.wav"
  subprocess.run(["ffmpeg","-y","-loglevel","error","-i",out+".mp3","-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1",clean],check=True)
  subprocess.run(["ffmpeg","-y","-loglevel","error","-i",clean,"-f","lavfi","-i","anoisesrc=color=pink:amplitude=0.12:sample_rate=24000:seed=2",
   "-filter_complex","[0]highpass=f=300,lowpass=f=3000,volume=2.2,alimiter=limit=0.6[a];[a][1]amix=inputs=2:duration=first:weights=1 1[m]",
   "-map","[m]","-ac","1","-ar","24000","-c:a","pcm_s16le","-map_metadata","-1",out+".wav"],check=True)
  log.append(dict(id=out,lang=lang,model=MODEL,voice_id=vid,seed=7,noise_seed=2,chars=len(text),text=text,request_id=r.headers.get("request-id")))
  print(out,"ok")
json.dump(log,open("generation_log.json","w"),indent=1,ensure_ascii=False)
