import json,subprocess,requests,os,concurrent.futures as cf,unicodedata,re
from gtts import gTTS
import jiwer
from common import H,spark
R=[
 ("en","English","Unit four to base. Three people trapped on the roof of the blue house on Maple Street, one elderly man injured, water rising fast, need a boat.", dict(people=3,injured=True,location="Maple Street",need="boat")),
 ("es","Spanish","Base, aquí equipo dos. Hay cinco personas atrapadas en la escuela de la calle Libertad, dos niños heridos, necesitamos un helicóptero.", dict(people=5,injured=True,location="calle Libertad",need="helicopter")),
 ("hi","Hindi","बेस, यह टीम तीन है। गांधी रोड पर मंदिर के पास चार लोग फंसे हैं, कोई घायल नहीं है, हमें नाव चाहिए।", dict(people=4,injured=False,location="Gandhi Road",need="boat")),
 ("tl","Tagalog","Base, ito ang team lima. May anim na tao na naipit sa bubong ng simbahan sa Rizal Street, isang buntis, kailangan namin ng bangka.", dict(people=6,injured=False,location="Rizal Street",need="boat")),
 ("bn","Bengali","বেস, এটা দল ছয়। স্টেশন রোডের হাসপাতালের পাশে দুইজন মানুষ আটকে আছে, একজন আহত, আমাদের অ্যাম্বুলেন্স দরকার।", dict(people=2,injured=True,location="Station Road",need="ambulance")),
 ("tr","Turkish","Merkez, burası ekip yedi. Atatürk Caddesi'ndeki çöken binada yedi kişi enkaz altında, üç yaralı var, vinç lazım.", dict(people=7,injured=True,location="Atatürk Caddesi",need="crane")),
 ("fr","French","Base, ici équipe huit. Huit personnes bloquées dans le gymnase de la rue Victor Hugo, pas de blessés, besoin d'eau potable.", dict(people=8,injured=False,location="rue Victor Hugo",need="water")),
 ("ar","Arabic","القاعدة، هنا الفريق تسعة. عشرة أشخاص محاصرون في المسجد في شارع النيل، ثلاثة مصابين، نحتاج إلى سيارة إسعاف.", dict(people=10,injured=True,location="Nile Street",need="ambulance")),
]
def mk(code,text,tag):
  V={'en':'en-US-GuyNeural','es':'es-MX-JorgeNeural','hi':'hi-IN-MadhurNeural','tl':'fil-PH-AngeloNeural','bn':'bn-BD-PradeepNeural','tr':'tr-TR-AhmetNeural','fr':'fr-FR-HenriNeural','ar':'ar-EG-ShakirNeural'}
  mp=f'voice/{tag}.mp3'
  subprocess.run(['.venv/bin/edge-tts','--voice',V[code],'--text',text,'--write-media',mp],check=True,capture_output=True)
  clean=f'voice/{tag}_clean.wav'; radio=f'voice/{tag}_radio.wav'
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',mp,'-ac','1','-ar','24000','-c:a','pcm_s16le',clean],check=True)
  # radio: 300-3000 Hz band, heavy pink noise ~8 dB SNR, light clipping
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',clean,'-f','lavfi','-i','anoisesrc=color=pink:amplitude=0.12:sample_rate=24000',
    '-filter_complex','[0]highpass=f=300,lowpass=f=3000,volume=2.2,alimiter=limit=0.6[a];[a][1]amix=inputs=2:duration=first:weights=1 1[m]','-map','[m]','-ac','1','-ar','24000','-c:a','pcm_s16le',radio],check=True)
  return clean,radio
def asr(path,lang=None):
  req={"mode":"PUSH_TO_TALK","model":"muse-voice-transcribe-1.0","audioEncoding":"WAV"}
  if lang: req["languageBias"]=[lang]
  for k in range(3):
    r=requests.post('https://api.meta.ai/v1/asr/transcribe',headers=H,files={'request':(None,json.dumps(req),'application/json'),'audio':(os.path.basename(path),open(path,'rb'),'audio/wav')},timeout=120)
    if r.status_code==200: return r.json().get('transcript','')
  return 'ERR '+str(r.status_code)+r.text[:150]
def norm(s): s=unicodedata.normalize('NFC',s.lower()); return re.sub(r"[^\w\s]","",s).strip()
EX=('A rescue radio note (any language) transcript follows. Extract in ENGLISH. Reply ONLY JSON {"people_trapped":int,"anyone_injured":true|false,"location":"street name as said","need":"one word: boat|helicopter|ambulance|crane|water|other","priority":"high|medium|low","english":"translation"}. Transcript: ')
def one(item):
  code,lang,text,truth=item; c,r=mk(code,text,code); out={"lang":lang}
  for cond,p,bias in [("clean",c,None),("radio",r,None),("radio+bias",r,lang)]:
    t=asr(p,bias); cer=jiwer.cer(norm(text),norm(t)) if not t.startswith('ERR') else None
    o,_=spark(EX+t,[]) ; f=dict(people=o.get('people_trapped')==truth['people'],injured=o.get('anyone_injured')==truth['injured'],need=str(o.get('need','')).lower()==truth['need'])
    out[cond]=dict(transcript=t,cer=None if cer is None else round(cer,3),fields=f,loc=o.get('location'),english=o.get('english'))
  print(lang,{k:(v['cer'],sum(v['fields'].values())) for k,v in out.items() if k!='lang'},flush=True); return out
def safe(i):
  try: return one(i)
  except Exception as e: print('skip',i[1],e,flush=True); return None
with cf.ThreadPoolExecutor(4) as ex: O=[o for o in ex.map(safe,R) if o]
json.dump(O,open('t9_results.json','w'),indent=1,ensure_ascii=False)
for cond in ["clean","radio","radio+bias"]:
  cs=[o[cond]['cer'] for o in O if o[cond]['cer'] is not None]; fs=sum(sum(o[cond]['fields'].values()) for o in O)
  print(cond,'mean CER',round(sum(cs)/len(cs),3),'fields correct',fs,'/',3*len(O))
