import json,glob,random,io,numpy as np,concurrent.futures as cf,pyarrow.parquet as pq
from PIL import Image
from common import *
random.seed(21)
fs=sorted(glob.glob('rn/*test-label*')); random.shuffle(fs)
# T4a: tile scan for destroyed buildings
pos=[];neg=[]
for f in fs:
  iid=f.split('__')[-1].split('_')[0]; a=np.array(Image.open(f))
  for r in range(3):
    for c in range(3):
      t=a[r*1000:(r+1)*1000,c*1333:(c+1)*1333]; d=(t==6).mean(); b=((t>=3)&(t<=5)).mean()
      if d>0.04: pos.append((iid,r,c,True))
      elif d==0 and b>0.04: neg.append((iid,r,c,False))
S=random.sample(pos,30)+random.sample(neg,30)
P=("Post-hurricane drone image tile. Does it contain a DESTROYED building: collapsed, flattened into a debris pile, or only a slab/pilings left where a house stood? "
"Standing houses with roof damage do NOT count. Reply ONLY JSON: {\"destroyed_present\":true|false,\"evidence\":\"<=20 words\"}")
def runA(x):
  iid,r,c,t=x; im=Image.open(f'rn/RescueNet__test__test-org-img__{iid}.jpg').crop((c*1333,r*1000,(c+1)*1333,(r+1)*1000))
  o,_=spark(P,[durl(im,1024)]); return {"x":x,"truth":t,"pred":bool(o.get('destroyed_present')),"o":o}
# T4b: roads with better prompt
T=pq.read_table('rn_test.parquet').to_pylist(); R2=json.load(open('t2_results.json'))
P2=("Post-hurricane drone image. Roads may be BURIED under sand, debris, or wreckage so that the pavement is no longer visible; infer where roads run from driveways, rows of lots and utility lines. "
"Is any road here impassable to a rescue vehicle (buried, covered by debris, trees, or water)? A buried/invisible road counts as blocked. "
'Reply ONLY JSON: {"blocked":true|false,"evidence":"<=20 words"}')
def runB(r):
  o,_=spark(P2,[durl(Image.open(io.BytesIO(T[r['i']]['image']['bytes'])),512)]); return {"truth":r['truth'],"pred":bool(o.get('blocked')),"o":o}
with cf.ThreadPoolExecutor(12) as ex:
  A=list(ex.map(runA,S)); B=list(ex.map(runB,R2))
json.dump({"A":A,"B":B},open('t4_results.json','w'),indent=1)
def rep(n,R):
  tp=sum(r['truth'] and r['pred'] for r in R);fp=sum((not r['truth']) and r['pred'] for r in R);fn=sum(r['truth'] and not r['pred'] for r in R)
  print(n,'acc',sum(r['truth']==r['pred'] for r in R),'/',len(R),'recall',tp,'/',tp+fn,'precision',tp,'/',tp+fp)
rep('tile-destroyed',A); rep('roads-v2',B)
