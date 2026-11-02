from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

DATA=Path(r"C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset\reorganized_dataset")

def prep(df,fitmask):
    d=df.copy(); p2ok=d.p2_x.notna().to_numpy(np.float32); p1x=d.p1_x.to_numpy(float); p1y=d.p1_y.to_numpy(float); p2x=d.p2_x.fillna(0).to_numpy(float); p2y=d.p2_y.fillna(0).to_numpy(float)
    fm=np.asarray(fitmask); pts=np.r_[p1x[fm],p2x[fm][p2ok[fm]>0]]; ptsy=np.r_[p1y[fm],p2y[fm][p2ok[fm]>0]]; cm=np.array([pts.mean(),ptsy.mean()]); cs=np.array([pts.std(),ptsy.std()]); cs[cs<1e-6]=1
    midx=np.where(p2ok>0,(p1x+p2x)/2,p1x); midy=np.where(p2ok>0,(p1y+p2y)/2,p1y); pair=np.where(p2ok>0,np.hypot(p1x-p2x,p1y-p2y),0); npt=1+p2ok
    raw_inv=np.c_[d.load_kg.to_numpy(float),npt,midx,midy,pair,d.direction.eq("Longitudinal").to_numpy(float)]; im=raw_inv[fm].mean(0); isd=raw_inv[fm].std(0); isd[isd<1e-6]=1
    raw_sp=np.c_[d.x_cm,d.y_cm,d.z_cm,d.relative_time_s]; sm=raw_sp[fm].mean(0); ss=raw_sp[fm].std(0); ss[ss<1e-6]=1
    inv=(raw_inv-im)/isd; sp=(raw_sp-sm)/ss
    x=np.c_[inv[:,0],inv[:,1],(p1x-cm[0])/cs[0],(p1y-cm[1])/cs[1],np.ones(len(d)),(p2x-cm[0])/cs[0],(p2y-cm[1])/cs[1],p2ok,inv[:,2:6],d.layer.eq("Asphalt").to_numpy(float),sp].astype(np.float32)
    norm={"coord_mean":cm.tolist(),"coord_std":cs.tolist(),"inv_mean":im.tolist(),"inv_std":isd.tolist(),"spatial_mean":sm.tolist(),"spatial_std":ss.tolist()}; return x,norm

class Core(nn.Module):
    def __init__(self,invariant=False,residual=False,h=64):
        super().__init__(); self.invariant=invariant; self.residual=residual
        if invariant:
            self.point=nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,h)); self.inv=nn.Sequential(nn.Linear(6,h),nn.Tanh(),nn.Linear(h,h))
        else: self.cond=nn.Sequential(nn.Linear(12,h),nn.Tanh(),nn.Linear(h,h))
        self.base=nn.Linear(4,h); self.g1=nn.Linear(h,h); self.b1=nn.Linear(h,h); self.g2=nn.Linear(h,h); self.b2=nn.Linear(h,h)
        if residual:
            self.shared=nn.Linear(h,1); self.ra=nn.Linear(h,1); self.rs=nn.Linear(h,1)
        else: self.oa=nn.Linear(h,1); self.os=nn.Linear(h,1)
    def forward(self,x):
        layer=x[:,12]
        if self.invariant:
            p1=x[:,2:5]; p2=x[:,5:8]; hp=self.point(p1)*p1[:,2:3]+self.point(p2)*p2[:,2:3]; iv=torch.cat([x[:,0:2],x[:,8:12]],1); c=torch.tanh(hp+self.inv(iv))
        else: c=self.cond(x[:,:12])
        z=torch.tanh(self.base(x[:,13:17])); z=torch.tanh(self.g1(c)*z+self.b1(c)); z=torch.tanh(self.g2(c)*z+self.b2(c))
        if self.residual:
            base=self.shared(z).squeeze(1); rr=torch.where(layer>=.5,self.ra(z).squeeze(1),self.rs(z).squeeze(1)); return base+0.1*rr
        return torch.where(layer>=.5,self.oa(z).squeeze(1),self.os(z).squeeze(1))

def train(model,x,y,fit,val,seed,max_epochs):
    torch.manual_seed(seed); np.random.seed(seed); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); ds=TensorDataset(torch.from_numpy(x[fit]),torch.from_numpy(y[fit].astype(np.float32))); dl=DataLoader(ds,batch_size=16384,shuffle=True); best=None; bestv=np.inf; stale=0; hist=[]; bestep=0
    xv=torch.from_numpy(x[val]); yv=torch.from_numpy(y[val].astype(np.float32))
    for ep in range(max_epochs):
        model.train(); losses=[]
        for xb,yb in dl:
            loss=((model(xb)-yb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval();
        with torch.no_grad(): vl=float(((model(xv)-yv)**2).mean())
        hist.append({"epoch":ep+1,"train_loss":float(np.mean(losses)),"val_loss":vl})
        if vl<bestv: bestv=vl;bestep=ep+1;stale=0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if ep+1>=50 and stale>=20: break
    model.load_state_dict(best); return model,pd.DataFrame(hist),bestep

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA/"film_dynamic_training_table.csv"); df=df[df.split_role.eq("development")].copy().reset_index(drop=True); y=df.response_ue.to_numpy(np.float32); seeds=[20261008,20261009,20261010]
    tasks=[]
    for fold in sorted(df.dev_fold.unique()): tasks.append(("F1",f"fold{int(fold)}",(df.dev_fold!=fold).to_numpy(),(df.dev_fold==fold).to_numpy()))
    for held in ["DualAsymmetric","DualSymmetric"]: tasks.append((held,held,(df.load_type!=held).to_numpy(),(df.load_type==held).to_numpy()))
    if args.task:
        tasks=[t for t in tasks if t[0] in args.task.split(',')]
    configs={"G2":(True,False),"G3":(False,True),"G2G3":(True,True)}
    if args.only:
        configs={k:v for k,v in configs.items() if k in args.only.split(',')}
    metrics=[]
    for name,(invariant,residual) in configs.items():
        md=out/name; (md/"predictions").mkdir(parents=True,exist_ok=True); (md/"history").mkdir(exist_ok=True); (md/"checkpoints").mkdir(exist_ok=True)
        for task,label,trainmask,testmask in tasks:
            folds=sorted(df.loc[trainmask,"dev_fold"].unique()); valmask=trainmask&(df.dev_fold==folds[0]).to_numpy(); fitmask=trainmask&~valmask&(df.relative_time_s.mod(5).eq(0).to_numpy()); x,norm=prep(df,fitmask)
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                model=Core(invariant,residual); model,hist,bestep=train(model,x,y,fitmask,valmask,seed,args.epochs); model.eval();
                with torch.no_grad(): pred=model(torch.from_numpy(x[testmask])).numpy()
                q=df[testmask].copy(); q["actual_ue"]=y[testmask]; q["predicted_ue"]=pred; q["model"]=name; q["seed"]=seed; q["task"]=task; q.to_csv(md/"predictions"/f"{task}_{label}_{seed}.csv",index=False,encoding="utf-8-sig"); hist.to_csv(md/"history"/f"{task}_{label}_{seed}.csv",index=False); torch.save({"state_dict":model.state_dict(),"normalizer":norm,"best_epoch":bestep,"invariant":invariant,"residual":residual},md/"checkpoints"/f"{task}_{label}_{seed}.pt")
                e=y[testmask]-pred; t=df.loc[testmask,"relative_time_s"].to_numpy(); lay=df.loc[testmask,"layer"].to_numpy(); ss=df.loc[testmask,"sensor_string"].to_numpy()
                for ph,m in [("overall",np.ones(len(e),bool)),("early",t<=90),("late",t>90),("late_asphalt",(t>90)&(lay=="Asphalt")),("late_soil",(t>90)&(lay=="Soil")),("SAL2",ss=="SAL2"),("SAT1",ss=="SAT1"),("SAL1",ss=="SAL1"),("SAT2",ss=="SAT2")]:
                    if m.any(): metrics.append({"model":name,"task":task,"fold":label,"seed":seed,"phase":ph,"sse":float(np.sum(e[m]**2)),"n":int(m.sum()),"rmse":float(np.sqrt(np.mean(e[m]**2))),"best_epoch":bestep})
                # exact swap test on dual tasks
                if task in ["DualAsymmetric","DualSymmetric"]:
                    xs=x[testmask].copy(); xs[:,2:5],xs[:,5:8]=x[testmask][:,5:8].copy(),x[testmask][:,2:5].copy()
                    with torch.no_grad(): ps=model(torch.from_numpy(xs)).numpy()
                    metrics.append({"model":name,"task":task,"fold":label,"seed":seed,"phase":"swap_prediction","sse":float(np.sum((pred-ps)**2)),"n":len(pred),"rmse":float(np.sqrt(np.mean((pred-ps)**2))),"best_epoch":bestep,"swap_median_abs":float(np.median(np.abs(pred-ps))),"swap_p95_abs":float(np.quantile(np.abs(pred-ps),.95))})
    m=pd.DataFrame(metrics); m.to_csv(out/"metrics_file_level.csv",index=False,encoding="utf-8-sig"); (out/"RUN_MANIFEST.json").write_text(json.dumps({"models":configs,"seeds":seeds,"epochs":args.epochs,"min_epochs":50,"patience":20,"loss":"ordinary MSE","fixed_test_opened":False,"tasks":["F1","DualAsymmetric","DualSymmetric"]},ensure_ascii=False,indent=2),encoding="utf-8")
    print(m.groupby(["model","task","phase"]).rmse.mean().to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=150); ap.add_argument("--only",default=""); ap.add_argument("--task",default=""); run(ap.parse_args())
