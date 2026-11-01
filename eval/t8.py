import glob,json,random,numpy as np,concurrent.futures as cf
from PIL import Image
from scipy import ndimage
from common import *
random.seed(8)
imgs=sorted(f[:-4] for f in glob.glob('fn/*.jpg'))
print('size',Image.open(imgs[0]+'.jpg').size,flush=True)
def gt(p):
  a=np.array(Image.open(p+'_lab.png'));
  if a.ndim==3: a=a[...,0]
  return a
# --- A: image-level flooded building / flooded road
PI=("Post-hurricane drone photo. Answer two questions about FLOODWATER (standing water where it should not be, not pools, ponds or rivers in their normal banks). "
"1) Is any building standing in or surrounded by floodwater? 2) Is any road covered by floodwater? Reply ONLY JSON "
'{"flooded_building":true|false,"flooded_road":true|false,"evidence":"<=20 words"}')
def imgq(p):
  a=gt(p); o,_=spark(PI,[durl(Image.open(p+'.jpg'),1280)])
  return dict(img=p,gt_b=int((a==1).sum())>20000,gt_r=int((a==3).sum())>20000,pb=o.get('flooded_building'),pr=o.get('flooded_road'))
with cf.ThreadPoolExecutor(10) as ex: A=list(ex.map(imgq,imgs))
def pr(name,t,p):
  tp=sum(a and b for a,b in zip(t,p)); fp=sum((not a) and b for a,b in zip(t,p)); fn=sum(a and not b for a,b in zip(t,p))
  s=f"{name}: acc {sum(a==bool(b) for a,b in zip(t,p))}/{len(t)} recall {tp}/{tp+fn} precision {tp}/{tp+fp}"; print(s,flush=True); return s
out={"A_building":pr('image flooded-building',[x['gt_b'] for x in A],[x['pb'] for x in A]),
     "A_road":pr('image flooded-road',[x['gt_r'] for x in A],[x['pr'] for x in A])}
# --- B: building crops flooded vs not
inst={1:[],2:[]}
for p in imgs:
  a=gt(p); lab,n=ndimage.label((a==1)|(a==2))
  for i,sl in enumerate(ndimage.find_objects(lab),1):
    m=lab[sl]==i
    if m.sum()<40000: continue
    v,c=np.unique(a[sl][m],return_counts=True)
    if c.max()/m.sum()<0.85: continue
    inst[int(v[c.argmax()])].append((p,sl))
print({k:len(v) for k,v in inst.items()},flush=True)
C=[]
for cls in (1,2):
  for p,sl in random.sample(inst[cls],min(40,len(inst[cls]))):
    im=Image.open(p+'.jpg'); W,Hh=im.size; y0,y1,x0,x1=sl[0].start,sl[0].stop,sl[1].start,sl[1].stop; h,w=y1-y0,x1-x0
    C.append((cls,im.crop((max(0,x0-w//2),max(0,y0-h//2),min(W,x1+w//2),min(Hh,y1+h//2)))))
PC=("Post-hurricane drone photo cropped around ONE building in the centre. Is THAT building flooded, i.e. standing in or surrounded by floodwater touching its walls? "
'Reply ONLY JSON {"flooded":true|false,"evidence":"<=20 words"}')
def cq(c): o,_=spark(PC,[durl(c[1],768)]); return c[0]==1,o.get('flooded')
with cf.ThreadPoolExecutor(10) as ex: B=list(ex.map(cq,C))
out["B_crops"]=pr('building crop flooded',[b[0] for b in B],[b[1] for b in B])
# --- C: SAM building recall on flooded vs dry buildings
def sr(p):
  a=gt(p); lab,n=ndimage.label((a==1)|(a==2)); im=Image.open(p+'.jpg'); W,Hh=im.size; bx=sam_image(im,'building'); res=[]
  for i,sl in enumerate(ndimage.find_objects(lab),1):
    m=lab==i
    if m[sl].sum()<40000: continue
    v,c=np.unique(a[sl][m[sl]],return_counts=True); cy,cx=ndimage.center_of_mass(m)
    hit=any(x1*W/w<=cx<=x2*W/w and y1*W/w<=cy<=y2*W/w for (_,x1,y1,x2,y2,w,cf_) in bx)
    res.append((int(v[c.argmax()]),hit))
  return res,len(bx)
sp=random.sample([p for p in imgs if p.split('/')[-1].startswith('Flooded')],12)
with cf.ThreadPoolExecutor(6) as ex: S=list(ex.map(sr,sp))
h1=[h for r,_ in S for c,h in r if c==1]; h2=[h for r,_ in S for c,h in r if c==2]
out["C_sam"]=f"SAM building recall: flooded {sum(h1)}/{len(h1)}, dry {sum(h2)}/{len(h2)}, cap16 hits {sum(n>=16 for _,n in S)}/12"; print(out["C_sam"],flush=True)
json.dump({"out":out,"A":A},open('t8_results.json','w'),indent=1,default=str)
