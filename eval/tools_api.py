# Tool layer exposed to Wonder-written survey scripts
import json,concurrent.futures as cf
from PIL import Image
from common import spark,sam_image,durl
CALLS={"spark":0,"sam":0}
def load_image(path): return Image.open(path).convert('RGB')
def tiles(img,rows=3,cols=3):
  W,H=img.size; out=[]
  for r in range(rows):
    for c in range(cols):
      box=(c*W//cols,r*H//rows,(c+1)*W//cols,(r+1)*H//rows); out.append({"r":r,"c":c,"box":box,"img":img.crop(box)})
  return out
def find_objects(img,noun):
  CALLS["sam"]+=1; return [{"x1":b[1],"y1":b[2],"x2":b[3],"y2":b[4],"conf":b[6],"scale":img.size[0]/b[5]} for b in sam_image(img,noun)]
def ask_vision(img,question,json_schema_hint):
  CALLS["spark"]+=1; o,_=spark(question+" Reply ONLY JSON: "+json_schema_hint,[durl(img,1024)]); return o
def crop(img,box,margin=0.25):
  x1,y1,x2,y2=box; w,h=x2-x1,y2-y1; W,H=img.size
  return img.crop((max(0,int(x1-w*margin)),max(0,int(y1-h*margin)),min(W,int(x2+w*margin)),min(H,int(y2+h*margin))))
def parallel(fn,items,workers=12):
  with cf.ThreadPoolExecutor(workers) as ex: return list(ex.map(fn,items))
