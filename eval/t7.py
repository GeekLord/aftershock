import json,random,requests,re,concurrent.futures as cf
from PIL import Image
from common import *
d=json.load(open('t5_results.json')); M=json.load(open('crops200_meta.json')); RA=d['RA']
random.seed(7); idx=list(range(200)); random.shuffle(idx); dev,hold=idx[:100],idx[100:]
PA=open('promptA.txt').read(); L=['no_damage','minor','major','destroyed']; sev=lambda g:g in('major','destroyed')
def sc(R,ids):
  t=[M[i]['truth'] for i in ids]; p=[R[k] for k in range(len(ids))]
  tp=sum(sev(a) and sev(b) for a,b in zip(t,p)); fp=sum(not sev(a) and sev(b) for a,b in zip(t,p))
  return dict(acc4=sum(a==b for a,b in zip(t,p)),severe_bin=sum(sev(a)==sev(b) for a,b in zip(t,p)),prec=f"{tp}/{tp+fp}",rec=f"{tp}/{sum(sev(a) for a in t)}")
def run(P,ids):
  def g(i):
    for k in range(3):
      try: o,_=spark(P,[durl(Image.open(M[i]['path']),768)]); return o.get('grade'),o.get('evidence')
      except Exception: pass
    return None,''
  with cf.ThreadPoolExecutor(10) as ex: return list(ex.map(g,ids))
OPT="muse-wonder-1.0"
hist=[]; P=PA; Rdev=[RA[i][0] for i in dev]; Ev=[RA[i][1] for i in dev]
print('base dev',sc(Rdev,dev),'base hold',sc([RA[i][0] for i in hold],hold),flush=True)
for it in range(2):
  errs=[f"truth={M[i]['truth']} predicted={Rdev[k]} " for k,i in enumerate(dev) if Rdev[k]!=M[i]['truth']]
  ask=("You improve a vision prompt for grading hurricane building damage from drone crops (no_damage|minor|major|destroyed). Current prompt:\n<<<\n"+P+"\n>>>\nIts errors on a labelled dev set (labels from RescueNet annotators):\n"+"\n".join(errs)+
       "\nDiagnose the recurring failure patterns, then write an improved prompt that keeps the same JSON output fields grade and evidence. Priority: keep severe (major/destroyed) recall high while cutting false severe calls on minor buildings. Reply ONLY JSON {\"diagnosis\":\"...\",\"prompt\":\"...\"}")
  for k in range(3):
    r=requests.post("https://api.meta.ai/v1/chat/completions",headers=H,json={"model":OPT,"messages":[{"role":"user","content":ask}],"response_format":{"type":"json_object"}},timeout=600).json()
    if 'choices' in r: break
    print('err',str(r)[:300],flush=True)
  o=json.loads(r['choices'][0]['message']['content']); P2=o['prompt']
  R2=run(P2,dev); s2=sc([x[0] for x in R2],dev); print('iter',it,'diagnosis:',o['diagnosis'][:400],'\n dev',s2,flush=True)
  hist.append({"prompt":P2,"diag":o['diagnosis'],"dev":s2})
  if s2['acc4']>=sc(Rdev,dev)['acc4']: P=P2; Rdev=[x[0] for x in R2]; Ev=[x[1] for x in R2]
Rh=run(P,hold); print('FINAL held-out with chosen prompt',sc([x[0] for x in Rh],hold),'chosen_is_base',P==PA,flush=True)
json.dump({"hist":hist,"final_prompt":P,"hold":sc([x[0] for x in Rh],hold)},open('t7_results.json','w'),indent=1)
