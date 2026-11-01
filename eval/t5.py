import glob,numpy as np,random,json,os,concurrent.futures as cf
from PIL import Image
from scipy import ndimage
from collections import Counter
from common import *
random.seed(101)
NM={3:'no_damage',4:'minor',5:'major',6:'destroyed'}; L=list(NM.values())
M=json.load(open('crops200_meta.json'))
PA=open('promptA.txt').read(); PB=open('promptB.txt').read()
def grade(m,P,model):
  for k in range(3):
    try:
      o,u=spark(P,[durl(Image.open(m['path']),768)],model); return o.get('grade'),o.get('evidence'),u
    except Exception as e: err=e
  return None,str(err)[:80],None
def run_all(P,model,MS):
  with cf.ThreadPoolExecutor(10) as ex: return list(ex.map(lambda m:grade(m,P,model),MS))
def score(name,MS,R):
  t=[m['truth'] for m in MS]; p=[r[0] for r in R]; sev=lambda g:g in('major','destroyed')
  acc=sum(a==b for a,b in zip(t,p)); bi=sum(sev(a)==sev(b) for a,b in zip(t,p))
  tp=sum(sev(a) and sev(b) for a,b in zip(t,p)); fp=sum(not sev(a) and sev(b) for a,b in zip(t,p)); fn=sum(sev(a) and not sev(b) for a,b in zip(t,p))
  ow=sum(b in L and abs(L.index(a)-L.index(b))<=1 for a,b in zip(t,p))
  toks=[r[2] for r in R if r[2]]; ct=sum(x['completion_tokens'] for x in toks); pt=sum(x['prompt_tokens'] for x in toks)
  s=dict(name=name,n=len(MS),acc4=acc,within1=ow,severe_bin=bi,sev_prec=f"{tp}/{tp+fp}",sev_rec=f"{tp}/{tp+fn}",prompt_toks=pt,compl_toks=ct)
  print(s,flush=True); return s
out={}
RA=run_all(PA,'muse-spark-1.3',M); out['sparkA']=score('spark-1.3 A',M,RA)
RB=run_all(PB,'muse-spark-1.3',M); out['sparkB']=score('spark-1.3 B',M,RB)
RW=run_all(PA,'muse-wonder-1.0',M); out['wonderA']=score('wonder A',M,RW)
# disagreement as uncertainty
def dis(rs):
  agree=[len(set(r[i][0] for r in rs))==1 for i in range(len(M))]
  err=[RA[i][0]!=M[i]['truth'] for i in range(len(M))]
  a=sum(agree); ea=sum(e for g,e in zip(agree,err) if g); ed=sum(e for g,e in zip(agree,err) if not g)
  return dict(agree=a,err_when_agree=f"{ea}/{a}",err_when_disagree=f"{ed}/{len(M)-a}")
out['disagree_AB']=dis([RA,RB]); out['disagree_AW']=dis([RA,RW]); out['disagree_ABW']=dis([RA,RB,RW])
# majority vote
from collections import Counter as C
vote=[(C([RA[i][0],RB[i][0],RW[i][0]]).most_common(1)[0][0],None,None) for i in range(len(M))]
out['vote']=score('vote A+B+W',M,vote)
print(json.dumps({k:v for k,v in out.items() if k.startswith('dis')}),flush=True)
json.dump({"out":out,"RA":RA,"RB":RB,"RW":RW},open('t5_results.json','w'))
