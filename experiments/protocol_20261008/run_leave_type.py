import os,sys,json,random
import numpy as np,pandas as pd,torch
from torch import nn
sys.path.insert(0,os.path.dirname(__file__))
from run_phase_ab import flags,make_model,forward,COND,SPAT,TABLE,seed_all
ROOT=os.path.join(os.path.dirname(__file__),'leave_type_out')
def enc(x,means,std):
 cont=COND[:5]+SPAT; c=torch.tensor(np.c_[((x[COND[:5]]-means[COND[:5]])/std[COND[:5]]).values,x[COND[5:]].values],dtype=torch.float32); s=torch.tensor(((x[SPAT]-means[SPAT])/std[SPAT]).values,dtype=torch.float32); ia=torch.tensor(x.is_asphalt_raw.values,dtype=torch.float32); y=torch.tensor(x.response_ue.values,dtype=torch.float32)[:,None]; return c,s,ia,y
def main():
 seed_all(20261008); d=flags(pd.read_csv(TABLE)); d=d[(d.relative_time_s<=280)&np.isfinite(d.response_ue)].copy(); d['dev_fold']=d.dev_fold.fillna(0).astype(int); d=d[d.dev_fold>0].copy(); d[COND+SPAT]=d[COND+SPAT].replace([np.inf,-np.inf],np.nan).fillna(0.); os.makedirs(ROOT,exist_ok=True)
 for kind in ['B0','B3']:
  out=os.path.join(ROOT,kind); os.makedirs(out,exist_ok=True)
  for hold in sorted(d.load_type.unique()):
   tr=d[(d.load_type!=hold)&(d.relative_time_s%5==0)].copy(); ev=d[d.load_type==hold].copy(); cases=tr.case_id.drop_duplicates().tolist(); ivcase=cases[0]; fit=tr[tr.case_id!=ivcase].copy(); iv=tr[tr.case_id==ivcase].copy(); cont=COND[:5]+SPAT; means=fit[cont].mean(); std=fit[cont].std().replace(0,1).fillna(1.); tc,ts,ti,ty=enc(fit,means,std); vc,vs,vi,vy=enc(iv,means,std); ec,es,ei,ey=enc(ev,means,std); model=make_model(kind,len(COND),len(SPAT)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5); best=1e99; state=None; bad=0
   for ep in range(100):
    model.train(); order=torch.randperm(len(ty))
    for st in range(0,len(order),1024):
     ix=order[st:st+1024]; opt.zero_grad(); loss=((forward(model,kind,tc[ix],ts[ix],ti[ix])-ty[ix])**2).mean(); loss.backward(); opt.step()
    model.eval();
    with torch.no_grad(): vl=float(((forward(model,kind,vc,vs,vi)-vy)**2).mean().sqrt())
    if vl<best-1e-4: best=vl; state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
    else: bad+=1
    if ep>=50 and bad>=20: break
   model.load_state_dict(state); model.eval();
   with torch.no_grad(): yp=forward(model,kind,ec,es,ei).numpy().ravel()
   z=ev[['case_id','sensor_id','layer','relative_time_s','quality_weight','response_ue']].copy(); z['predicted_ue']=yp; z['fold']=hold; z=z.rename(columns={'response_ue':'actual_ue'}); z.to_csv(os.path.join(out,'leave_'+hold+'_predictions.csv'),index=False); print(kind,hold,len(z))
if __name__=='__main__': main()
