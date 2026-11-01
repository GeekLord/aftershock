import json,re,time,traceback,io,contextlib,sys
from common import spark
API=open('tools_api.py').read()
TASK=("You are the planner of Aftershock, a disaster triage agent. Write ONE self-contained Python 3 program (no prose) that surveys a single post-hurricane drone image at path IMAGE_PATH (4000x3000) "
"and prints ONE JSON object: {\"tiles\":[{\"r\":..,\"c\":..,\"destroyed_present\":bool,\"road_blocked\":bool,\"buildings\":[{\"box\":[x1,y1,x2,y2],\"grade\":..}]}],\"priority\":[list of tiles ranked most urgent first with a one-line reason]}. "
"Use ONLY these tools, already importable via `from tools_api import *` (source below). Rules: find_objects (SAM) misses destroyed buildings, so detect destroyed areas with ask_vision on each tile; "
"roads may be buried under sand/debris and that counts as blocked; grade each SAM building crop with ask_vision into no_damage|minor|major|destroyed; SAM boxes are in the coordinates of the image you passed scaled by 'scale'. "
"Run independent calls concurrently with parallel(). Cap at 40 building crops total. Do not print anything except the final JSON.\n\nTOOLS SOURCE:\n"+API)
t=time.time()
from common import H
import requests
r=requests.post("https://api.meta.ai/v1/chat/completions",headers=H,json={"model":"muse-wonder-1.0","messages":[{"role":"user","content":TASK}]},timeout=600)
j=r.json(); code=j['choices'][0]['message']['content']; print('wonder secs',round(time.time()-t,1),j.get('usage'),flush=True)
m=re.search(r"```(?:python)?\n(.*?)```",code,re.S); code=m.group(1) if m else code
open('t6_program.py','w').write(code)
results=[]
for iid in ['13639','11121','10844']:
  src=code.replace('IMAGE_PATH',repr(f'rn/RescueNet__test__test-org-img__{iid}.jpg'))
  import tools_api; tools_api.CALLS.update(spark=0,sam=0)
  buf=io.StringIO(); t=time.time(); err=None
  try:
    with contextlib.redirect_stdout(buf): exec(compile(src,'prog','exec'),{"__name__":"__main__"})
  except Exception as e: err=traceback.format_exc()[-800:]
  outp=buf.getvalue()
  try: parsed=json.loads(outp[outp.find('{'):outp.rfind('}')+1]); ok=True
  except Exception: parsed=None; ok=False
  results.append({"img":iid,"secs":round(time.time()-t,1),"calls":dict(tools_api.CALLS),"json_ok":ok,"err":err,"out":parsed})
  print(iid,round(time.time()-t,1),dict(tools_api.CALLS),'json_ok',ok,'err',(err or '')[-300:],flush=True)
json.dump(results,open('t6_results.json','w'),indent=1)
