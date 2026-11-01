import json,os,subprocess,time,sys
res=[]
for iid in ['13639','11121','10844']:
  env=dict(os.environ,IMAGE_PATH=f'rn/RescueNet__test__test-org-img__{iid}.jpg',PYTHONPATH='.')
  # count calls via wrapper
  wrapper=("import tools_api,runpy,sys,json\nsys.argv=['prog']\n"
   "try:\n runpy.run_path('t6_program.py',run_name='__main__')\nfinally:\n print('\\nCALLS'+json.dumps(tools_api.CALLS),file=sys.stderr)\n")
  t=time.time(); p=subprocess.run([sys.executable,'-c',wrapper],env=env,capture_output=True,text=True,timeout=900)
  out=p.stdout; calls=[l for l in p.stderr.splitlines() if l.startswith('CALLS')]
  try: parsed=json.loads(out[out.find('{'):out.rfind('}')+1]); ok=True
  except Exception: parsed=None; ok=False
  res.append({"img":iid,"secs":round(time.time()-t,1),"calls":calls,"json_ok":ok,"rc":p.returncode,"err":p.stderr[-600:] if p.returncode else "","out":parsed})
  print(iid,res[-1]['secs'],calls,'ok',ok,'rc',p.returncode,res[-1]['err'][-300:],flush=True)
json.dump(res,open('t6_results.json','w'),indent=1)
