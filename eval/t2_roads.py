import json,io,random,numpy as np,concurrent.futures as cf, pyarrow.parquet as pq
from PIL import Image
from common import *
T=pq.read_table('rn_test.parquet').to_pylist(); random.seed(3)
blk=[];clr=[]
for i,r in enumerate(T):
  a=np.array(Image.open(io.BytesIO(r['label']['bytes']))); f8=(a==8).mean(); f7=(a==7).mean()
  if f8>0.03: blk.append(i)
  elif f8==0 and f7>0.05: clr.append(i)
S=[(i,True) for i in random.sample(blk,20)]+[(i,False) for i in random.sample(clr,20)]
P=("Post-hurricane drone image. Is any road in this image BLOCKED (impassable to a rescue vehicle) by debris, fallen trees, sand, water, or wreckage? "
'Reply ONLY JSON: {"road_visible":true|false,"blocked":true|false,"evidence":"<=20 words"}')
def run(x):
  i,t=x; o,u=spark(P,[durl(Image.open(io.BytesIO(T[i]['image']['bytes'])),512)]); return {"i":i,"truth":t,"pred":o.get('blocked'),"o":o}
with cf.ThreadPoolExecutor(10) as ex: R=list(ex.map(run,S))
json.dump(R,open('t2_results.json','w'),indent=1)
tp=sum(r['truth'] and r['pred'] for r in R);fp=sum((not r['truth']) and r['pred'] for r in R);fn=sum(r['truth'] and not r['pred'] for r in R)
print('roads acc',sum(r['truth']==bool(r['pred']) for r in R),len(R),'blocked recall',tp,'/',tp+fn,'precision',tp,'/',tp+fp)
print('NOTE label 8 meaning assumed blocked; check',json.dumps([r['o'] for r in R[:3]]))
