import os,sys,json,base64,requests,time
clip,prompt,out=sys.argv[1],sys.argv[2],sys.argv[3]
b=base64.b64encode(open(clip,'rb').read()).decode()
t=time.time()
r=requests.post("https://api.meta.ai/v1/responses",headers={"Authorization":f"Bearer {os.environ['MODEL_API_KEY']}","Content-Type":"application/json","Accept":"text/event-stream"},
 json={"model":"sam-3.1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":prompt},{"type":"input_video","video_url":"data:video/mp4;base64,"+b}]}],"stream":True,"metadata":{"include_confidence":"true","mask_encoding":"one_bit"}},stream=True,timeout=600)
if r.status_code!=200: print(r.status_code,r.text[:800]); sys.exit(1)
lines=[];events=[]
for raw in r.iter_lines():
  if not raw: continue
  l=raw.decode()
  if not l.startswith("data: "): continue
  d=l[6:]
  if d=="[DONE]": break
  e=json.loads(d); events.append(e)
  if e.get("type")=="response.output_text.delta": lines.append(e["delta"])
  if e.get("type") in("response.failed","error"): print(e)
json.dump({"lines":lines,"usage":[e.get("response",{}).get("usage") for e in events if e.get("type")=="response.completed"]},open(out,'w'))
print(prompt,"frames:",len(lines),"secs:",round(time.time()-t,1))
