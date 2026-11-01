import json,glob,random,numpy as np,concurrent.futures as cf
from PIL import Image
from scipy import ndimage
from common import *
M=json.load(open('crops_meta.json')); imgs=sorted(set(m['img'] for m in M)); random.seed(5); imgs=random.sample(imgs,12)
def run(iid):
  a=np.array(Image.open(f'rn/RescueNet__test__test-label-img__{iid}_lab.png'))
  lab,n=ndimage.label((a>=3)&(a<=6)); gt=[]
  for i,sl in enumerate(ndimage.find_objects(lab),1):
    m=lab[sl]==i
    if m.sum()<60000: continue
    vals,c=np.unique(a[sl][m],return_counts=True); cy,cx=ndimage.center_of_mass(lab==i)
    gt.append((int(vals[c.argmax()]),cx,cy))
  im=Image.open(f'rn/RescueNet__test__test-org-img__{iid}.jpg'); out={}
  for p in ['building','house','damaged building']:
    bx=sam_image(im,p); 
    def hit(cx,cy): return any(x1*4000/w<=cx<=x2*4000/w and y1*3000/(w*3/4)<=cy<=y2*3000/(w*3/4) for (_,x1,y1,x2,y2,w,c) in bx)
    out[p]={"n":len(bx),"hits":[(g[0],hit(g[1],g[2])) for g in gt]}
  return iid,out
with cf.ThreadPoolExecutor(6) as ex: R=dict(ex.map(run,imgs))
json.dump(R,open('t3_results.json','w'),indent=1)
from collections import defaultdict
for p in ['building','house','damaged building']:
  d=defaultdict(lambda:[0,0])
  for v in R.values():
    for cls,h in v[p]['hits']: d[cls][0]+=h; d[cls][1]+=1
  print(p,{k:f"{a}/{b}" for k,(a,b) in sorted(d.items())},'caps16',sum(v[p]['n']>=16 for v in R.values()))
  u=defaultdict(lambda:[0,0])
for iid,v in R.items(): print(iid,{p:(v[p]['n'],sum(h for _,h in v[p]['hits']),len(v[p]['hits'])) for p in v})
