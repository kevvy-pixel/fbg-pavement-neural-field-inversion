import os, json, argparse, random, hashlib
import numpy as np, pandas as pd, torch
from torch import nn

TABLE=r'C:\Users\27356\Documents\Codex\2026-10-07\c-users-27356-documents-codex-2026\outputs\reorganized_dataset_excel_mapping\film_dynamic_training_table.csv'
ROOT=os.path.dirname(__file__)
SEED=20261008
COND=['load_kg','p1_x','p1_y','p2_x','p2_y','is_single','is_dual_symmetric','is_dual_asymmetric','is_longitudinal_line','sensor_direction']
SPAT=['sensor_x','sensor_y','sensor_z','relative_time_s']

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def flags(d):
    t=d.load_type.fillna('').astype(str)
    d['is_single']=t.str.startswith('Single').astype(float)
    d['is_dual_symmetric']=t.eq('DualSymmetric').astype(float)
    d['is_dual_asymmetric']=t.eq('DualAsymmetric').astype(float)
    d['is_longitudinal_line']=t.eq('LongitudinalLine').astype(float)
    d['sensor_direction']=d.direction.map({'Longitudinal':0.0,'Transverse':1.0}).fillna(0.0)
    d['sensor_x']=d.x_cm; d['sensor_y']=d.y_cm; d['sensor_z']=d.z_cm
    d['is_asphalt_raw']=(d.layer.astype(str)=='Asphalt').astype(np.int64)
    return d
class MLP(nn.Module):
    def __init__(self,in_dim,h=64,layers=2,out_dim=1):
        super().__init__(); a=[]; z=in_dim
        for _ in range(layers): a += [nn.Linear(z,h),nn.SiLU()]; z=h
        a += [nn.Linear(z,out_dim)]; self.net=nn.Sequential(*a)
    def forward(self,x): return self.net(x)
class FiLM(nn.Module):
    def __init__(self,cond_dim,spat_dim,h=64,layered=False,layers=2,drop=0):
        super().__init__(); self.layered=layered; self.h=h
        self.c=MLP(cond_dim,h,layers,h); self.s=nn.Linear(spat_dim,h)
        self.g=nn.ModuleList([nn.Linear(h,h) for _ in range(layers)])
        self.b=nn.ModuleList([nn.Linear(h,h) for _ in range(layers)])
        self.blocks=nn.ModuleList([nn.Linear(h,h) for _ in range(layers)])
        self.act=nn.SiLU(); self.drop=nn.Dropout(drop)
        self.head_a=nn.Linear(h,1); self.head_s=nn.Linear(h,1) if layered else self.head_a
    def forward(self,c,s,ia):
        cc=self.c(c); z=self.s(s)
        for g,b,l in zip(self.g,self.b,self.blocks): z=self.act(l(z)* (1+g(cc)) + b(cc)); z=self.drop(z)
        ya=self.head_a(z); ys=self.head_s(z)
        return torch.where(ia[:,None]>0.5,ya,ys)
def make_model(kind,cd,sd):
    if kind=='B0': return MLP(cd+sd,64,2,1)
    if kind=='B1':
        class L(nn.Module):
            def __init__(self):
                super().__init__(); self.net=MLP(cd+sd,64,2,64); self.a=nn.Linear(64,1); self.s=nn.Linear(64,1)
            def forward(self,x,ia):
                z=self.net(x); return torch.where(ia[:,None]>0.5,self.a(z),self.s(z))
        return L()
    if kind=='B2': return FiLM(cd,sd,64,False,2,0)
    if kind=='B3': return FiLM(cd,sd,64,True,2,0)
    if kind=='B4': return FiLM(cd,sd,128,True,3,0)
    if kind=='B5': return FiLM(cd,sd,64,True,2,.10)
    raise ValueError(kind)
def forward(m,kind,c,s,ia):
    if kind=='B0': return m(torch.cat([c,s],1))
    if kind=='B1': return m(torch.cat([c,s],1),ia)
    return m(c,s,ia)
def save_json(p,x):
    with open(p,'w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',default='B0,B1,B2,B3,B4,B5'); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--epochs',type=int,default=100); args=ap.parse_args()
    seed_all(args.seed); d=flags(pd.read_csv(TABLE)); d=d[d.relative_time_s<=280].copy(); d['dev_fold']=d.dev_fold.fillna(0).astype(int)
    d=d[np.isfinite(d.response_ue)].copy()
    d[COND+SPAT]=d[COND+SPAT].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    # deterministic outer folds: the five fixed-test cases have dev_fold 0 and are never evaluated here
    dev=d[d.dev_fold>0].copy(); folds=sorted(dev.dev_fold.unique().tolist())
    cont=COND[:5]+SPAT; cat=COND[5:]
    outroot=os.path.join(ROOT,'A_B_models'); os.makedirs(outroot,exist_ok=True)
    for kind in args.models.split(','):
      mroot=os.path.join(outroot,kind,f'seed_{args.seed}'); os.makedirs(mroot,exist_ok=True)
      allpred=[]; logs=[]
      save_json(os.path.join(mroot,'config.json'),{'model':kind,'seed':args.seed,'epochs_max':args.epochs,'train_stride_s':5,'eval_stride_s':1,'fixed_test_opened':False,'table':TABLE})
      for fold in folds:
        trcase=dev.loc[dev.dev_fold!=fold,'case_id'].drop_duplicates().tolist(); vcase=dev.loc[dev.dev_fold==fold,'case_id'].drop_duplicates().tolist()
        # one deterministic internal validation case, still case-level
        ivcase=trcase[0]; tcase=[x for x in trcase if x!=ivcase]
        train=dev[dev.case_id.isin(tcase)&(dev.relative_time_s%5==0)].copy(); iv=dev[dev.case_id==ivcase].copy(); ev=dev[dev.case_id.isin(vcase)].copy()
        means=train[cont].mean(); std=train[cont].std().replace(0,1).fillna(1)
        def enc(x):
            c=torch.tensor(np.c_[((x[COND[:5]]-means[COND[:5]])/std[COND[:5]]).values,x[cat].values],dtype=torch.float32)
            s=torch.tensor(((x[SPAT]-means[SPAT])/std[SPAT]).values,dtype=torch.float32)
            ia=torch.tensor(x.is_asphalt_raw.values,dtype=torch.float32); y=torch.tensor(x.response_ue.values,dtype=torch.float32)[:,None]
            return c,s,ia,y
        tc,ts,ti,ty=enc(train); vc,vs,vi,vy=enc(iv); ec,es,ei,ey=enc(ev)
        model=make_model(kind,len(COND),len(SPAT)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4 if kind=='B5' else 1e-5)
        best=float('inf'); beststate=None; bad=0
        for ep in range(1,args.epochs+1):
            model.train(); order=torch.randperm(len(ty)); sq=0.0
            for start in range(0,len(order),1024):
                ix=order[start:start+1024]; opt.zero_grad(); pred=forward(model,kind,tc[ix],ts[ix],ti[ix]); loss=((pred-ty[ix])**2).mean(); loss.backward(); opt.step(); sq += float(loss.detach())*len(ix)
            model.eval();
            with torch.no_grad(): vl=float(((forward(model,kind,vc,vs,vi)-vy)**2).mean().sqrt())
            logs.append({'model':kind,'seed':args.seed,'fold':fold,'epoch':ep,'train_rmse':float(np.sqrt(sq/len(ty))),'internal_val_rmse':vl})
            if vl<best-1e-4: best=vl; beststate={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
            else: bad+=1
            if ep>=50 and bad>=20: break
        model.load_state_dict(beststate); model.eval()
        with torch.no_grad(): yp=forward(model,kind,ec,es,ei).numpy().ravel()
        z=ev[['case_id','sensor_id','layer','relative_time_s','quality_weight','response_ue']].copy(); z['predicted_ue']=yp; z['fold']=fold; z=z.rename(columns={'response_ue':'actual_ue'}); allpred.append(z); z.to_csv(os.path.join(mroot,f'fold_{fold}_predictions.csv'),index=False)
      pd.DataFrame(logs).to_csv(os.path.join(mroot,'training_log.csv'),index=False)
      pd.concat(allpred,ignore_index=True).to_csv(os.path.join(mroot,'oof_predictions.csv'),index=False)
      print(kind, 'done', len(pd.concat(allpred)), 'rows')
if __name__=='__main__': main()
