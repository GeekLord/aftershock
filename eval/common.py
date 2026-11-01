import os,json,base64,io,requests,re
from PIL import Image
H={"Authorization":f"Bearer {os.environ['MODEL_API_KEY']}"}
def durl(im,maxs=1280,q=85):
  im=im.convert('RGB'); im.thumbnail((maxs,maxs)); b=io.BytesIO(); im.save(b,'JPEG',quality=q)
  return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
def spark(prompt,imgs,model="muse-spark-1.3"):
  content=[{"type":"text","text":prompt}]+[{"type":"image_url","image_url":{"url":u}} for u in imgs]
  for attempt in range(3):
    r=requests.post("https://api.meta.ai/v1/chat/completions",headers=H,json={"model":model,"messages":[{"role":"user","content":content}],"response_format":{"type":"json_object"}},timeout=90)
    if r.status_code==200:
      t=r.json()['choices'][0]['message']['content']
      try: return json.loads(t[t.find('{'):t.rfind('}')+1]),r.json().get('usage')
      except Exception: return {"raw":t[:300]},None
  return {"err":r.status_code,"t":r.text[:200]},None
def sam_image(im,prompt):
  u=durl(im,1600,88)
  r=requests.post("https://api.meta.ai/v1/responses",headers={**H,"Accept":"text/event-stream"},json={"model":"sam-3.1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":prompt},{"type":"input_image","image_url":u}]}],"stream":True,"metadata":{"include_confidence":"true","mask_encoding":"one_bit"}},stream=True,timeout=90)
  txt=""
  for raw in r.iter_lines():
    l=raw.decode()
    if l.startswith("data: ") and l[6:]!="[DONE]":
      e=json.loads(l[6:])
      if e.get("type")=="response.output_text.delta": txt+=e["delta"]
  return [tuple(map(int,m.groups()[:6]))+(float(m.group(7)),) for m in re.finditer(r'(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=\d+;c=([\d.]+)',txt)]
