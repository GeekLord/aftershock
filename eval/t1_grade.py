import json,concurrent.futures as cf
from PIL import Image
from common import *
P=("You are a FEMA-trained damage assessor. This is a post-hurricane drone photo cropped around ONE building (the building centred in the frame). "
"Grade that building on this scale: no_damage = roof and walls intact; minor = some shingles/roof covering lost, small debris, structure sound; "
"major = significant roof loss exposing decking/rafters, partial wall failure, or structure shifted; destroyed = collapsed, flattened, or only slab/pilings remain. "
"Look carefully before deciding; most standing houses are not destroyed. Reply ONLY JSON: "
'{"grade":"no_damage|minor|major|destroyed","needs_human_review":true|false,"evidence":"<=25 words of visible evidence"}')
M=json.load(open('crops_meta.json')); nm={3:'no_damage',4:'minor',5:'major',6:'destroyed'}
def run(m):
  o,u=spark(P,[durl(Image.open(m['path']),768)]); return {**m,"truth":nm[m['cls']],"pred":o.get('grade'),"review":o.get('needs_human_review'),"evidence":o.get('evidence'),"o":o,"usage":u}
with cf.ThreadPoolExecutor(10) as ex: R=list(ex.map(run,M))
json.dump(R,open('t1_results.json','w'),indent=1)
from collections import Counter
L=['no_damage','minor','major','destroyed']
acc=sum(r['truth']==r['pred'] for r in R); print('4-class',acc,len(R))
cm=Counter((r['truth'],r['pred']) for r in R)
print('truth\\pred',L); [print(t,[cm[(t,p)] for p in L]) for t in L]
sev=lambda g:g in('major','destroyed')
b=sum(sev(r['truth'])==sev(r['pred']) for r in R); print('binary severe vs not',b,len(R))
tp=sum(sev(r['truth']) and sev(r['pred']) for r in R); fp=sum((not sev(r['truth'])) and sev(r['pred']) for r in R); fn=sum(sev(r['truth']) and not sev(r['pred']) for r in R)
print('severe precision',tp,'/',tp+fp,'recall',tp,'/',tp+fn)
off1=sum(abs(L.index(r['truth'])-L.index(r['pred']))<=1 for r in R if r['pred'] in L); print('within one level',off1)
rv=[r for r in R if r['review']]; print('flagged review',len(rv),'errors among flagged',sum(r['truth']!=r['pred'] for r in rv),'errors unflagged',sum(r['truth']!=r['pred'] for r in R if not r['review']))
