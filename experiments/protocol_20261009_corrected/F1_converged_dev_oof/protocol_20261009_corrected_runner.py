from pathlib import Path
import argparse, json, os, time
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset\reorganized_dataset")

def add_features(df, version="G1"):
    d = df.copy()
    for c in ["p1_x","p1_y","p2_x","p2_y"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["p2_x"] = d["p2_x"].fillna(d["p1_x"])
    d["p2_y"] = d["p2_y"].fillna(d["p1_y"])
    d["is_asphalt"] = d.layer.eq("Asphalt").astype(float)
    d["is_long"] = d.direction.eq("Longitudinal").astype(float)
    d["n_points"] = d.load_type.str.contains("Dual").astype(float) + 1.0
    d["mid_x"] = (d.p1_x+d.p2_x)/2; d["mid_y"] = (d.p1_y+d.p2_y)/2
    d["pair_dist"] = np.hypot(d.p1_x-d.p2_x, d.p1_y-d.p2_y)
    d["d1"] = np.hypot(d.x_cm-d.p1_x, d.y_cm-d.p1_y)
    d["d2"] = np.hypot(d.x_cm-d.p2_x, d.y_cm-d.p2_y)
    d["dmin"] = np.minimum(d.d1,d.d2)
    # In the source event table, p1/p2 are the only available geometric endpoints
    # for longitudinal line events; expose them explicitly as line endpoints.
    if version == "G0":
        d["is_single"] = d.load_type.str.contains("Single").astype(float)
        d["is_dual"] = d.load_type.str.contains("Dual").astype(float)
        d["is_line"] = d.load_type.str.contains("LongitudinalLine").astype(float)
        cols = ["load_kg","p1_x","p1_y","p2_x","p2_y","is_long","is_single","is_dual","is_line","is_asphalt","x_cm","y_cm","z_cm","relative_time_s"]
    else:
        cols = ["load_kg","n_points","p1_x","p1_y","p2_x","p2_y","mid_x","mid_y","pair_dist","d1","d2","dmin","is_long","is_asphalt","x_cm","y_cm","z_cm","relative_time_s"]
    return d, cols

def metrics(pred, truth, d):
    out=[]; e=truth-pred
    groups=[("overall",np.ones(len(d),bool)),("early",d.relative_time_s.to_numpy()<=90),("late",d.relative_time_s.to_numpy()>90),
            ("late_asphalt",(d.layer=="Asphalt").to_numpy()&(d.relative_time_s.to_numpy()>90)),
            ("late_soil",(d.layer=="Soil").to_numpy()&(d.relative_time_s.to_numpy()>90))]
    for name,m in groups:
        if m.any(): out.append({"phase":name,"rmse_ue":float(np.sqrt(np.mean(e[m]**2))),"n":int(m.sum())})
    c=[]
    for case,g in d.assign(_e2=e*e).groupby("case_id"):
        c.append({"case_id":case,"rmse_ue":float(np.sqrt(g._e2.mean())),"n":int(len(g))})
    return out, pd.DataFrame(c)

def ridge_predict(xtr,ytr,xte,alpha=1e-2):
    mu=xtr.mean(0); sd=xtr.std(0); sd[sd<1e-6]=1
    a=(xtr-mu)/sd; b=(xte-mu)/sd
    xt=np.concatenate([np.ones((len(a),1)),a],1); xv=np.concatenate([np.ones((len(b),1)),b],1)
    reg=alpha*np.eye(xt.shape[1]); reg[0,0]=0
    w=np.linalg.solve(xt.T@xt+reg,xt.T@ytr)
    return xv@w

def torch_models(input_dim, cond_dim, spatial_dim, layered=True):
    import torch
    from torch import nn
    class MLP(nn.Module):
        def __init__(self):
            super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        def forward(self,x): return self.net(x).squeeze(1)
    class FiLM(nn.Module):
        def __init__(self):
            super().__init__(); h=64; self.cond=nn.Sequential(nn.Linear(cond_dim,h),nn.Tanh(),nn.Linear(h,h)); self.base=nn.Linear(spatial_dim,h); self.g1=nn.Linear(h,h); self.b1=nn.Linear(h,h); self.g2=nn.Linear(h,h); self.b2=nn.Linear(h,h); self.a=nn.Linear(h,1); self.s=nn.Linear(h,1)
        def forward(self,x):
            c=x[:,:cond_dim]; z=torch.tanh(self.base(x[:,cond_dim:])); h=self.cond(c); z=torch.tanh(self.g1(h)*z+self.b1(h)); z=torch.tanh(self.g2(h)*z+self.b2(h)); return torch.where(x[:,cond_dim-1]>=0,self.a(z).squeeze(1),self.s(z).squeeze(1))
    return MLP, FiLM

def train_nn(model_name, xtr,ytr,wtr,xval,yval,seed,epochs=150):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    MLP,FiLM=torch_models(xtr.shape[1], xtr.shape[1]-4, 4)
    model=MLP() if model_name=="B0" else FiLM()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    ds=TensorDataset(torch.from_numpy(xtr.astype(np.float32)),torch.from_numpy(ytr.astype(np.float32)),torch.from_numpy(wtr.astype(np.float32)))
    loader=DataLoader(ds,batch_size=16384,shuffle=True)
    best=None; bestv=np.inf; history=[]; best_epoch=0; stale=0
    for ep in range(epochs):
        model.train()
        epoch_losses=[]
        for xb,yb,wb in loader:
            loss=((model(xb)-yb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): v=float(torch.mean((model(torch.from_numpy(xval.astype(np.float32)))-torch.from_numpy(yval.astype(np.float32)))**2))
        history.append({"epoch":ep+1,"train_loss":float(np.mean(epoch_losses)),"val_loss":v})
        if v<bestv: bestv=v; best_epoch=ep+1; stale=0; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if ep+1>=50 and stale>=20: break
    if best: model.load_state_dict(best)
    model.eval()
    return model, pd.DataFrame(history), best_epoch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--stage",choices=["A","C1","D1","both"],default="A"); ap.add_argument("--version",choices=["G0","G1"],default="G1"); ap.add_argument("--epochs",type=int,default=35); ap.add_argument("--seeds",default="20261008,20261009,20261010"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/"film_dynamic_training_table.csv"); df=df[df.split_role.eq("development")].copy(); df,cols=add_features(df,args.version)
    df[cols]=df[cols].apply(pd.to_numeric,errors="coerce").fillna(0)
    split=[]
    for case,g in df.groupby("case_id",sort=False):
        lt=g.load_type.iloc[0]; fold=int(g.dev_fold.iloc[0])
        split.append({"case_id":case,"load_type":lt,"outer_role":"development","held_out_type":"","development_or_fixed_test":"development","dev_fold":fold})
    splitdf=pd.DataFrame(split)
    (out/"splits").mkdir(exist_ok=True)
    # frozen C1 split file
    rows=[]
    for held in sorted(df.load_type.unique()):
        for _,r in splitdf.iterrows():
            rows.append({**r.to_dict(),"outer_role":"test" if r.load_type==held else "train_pool","held_out_type":held})
    pd.DataFrame(rows).to_csv(out/"splits"/"leave_type_split_v2.csv",index=False,encoding="utf-8-sig")
    # frozen D1 sensor split file
    sm=df[["sensor_id","sensor_string","layer"]].drop_duplicates().copy(); sm["spatial_fold"]=sm.sensor_string; sm["role"]="development"; sm.to_csv(out/"splits"/"sensor_spatial_split_v1.csv",index=False,encoding="utf-8-sig")
    seeds=[int(x) for x in args.seeds.split(",")]
    import torch
    def run_one(model, trainmask, testmask, seed, tag, held):
        tr=df[trainmask].copy(); te=df[testmask].copy();
        xte=df.loc[testmask,cols].to_numpy(np.float32); yte=df.loc[testmask,"response_ue"].to_numpy(np.float32)
        dest=out/("phase_A_baseline" if tag=="A" else ("phase_C_leave_type" if tag=="C1" else "phase_D_sensor_holdout"))/model; dest.mkdir(parents=True,exist_ok=True)
        if model=="A0-3":
            fitmask=trainmask & ((df.relative_time_s.to_numpy()%5)==0)
            xtr=df.loc[fitmask,cols].to_numpy(np.float32); ytr=df.loc[fitmask,"response_ue"].to_numpy(np.float32)
            pred=ridge_predict(xtr,ytr,xte)
        else:
            # complete-case inner validation by case/fold, never by rows
            validmask=trainmask & (df.dev_fold==int(sorted(df.loc[trainmask,"dev_fold"].unique())[0]))
            fitmask=trainmask & ~validmask & ((df.relative_time_s.to_numpy()%5)==0)
            xtr=df.loc[fitmask,cols].to_numpy(np.float32); xva=df.loc[validmask,cols].to_numpy(np.float32)
            ytr=df.loc[fitmask,"response_ue"].to_numpy(np.float32); yva=df.loc[validmask,"response_ue"].to_numpy(np.float32); qw=df.loc[fitmask,"quality_weight"].to_numpy(np.float32)
            normmu=xtr.mean(0); normsd=xtr.std(0); normsd[normsd<1e-6]=1
            m,hist,best_epoch=train_nn(model,((xtr-normmu)/normsd),ytr,qw,((xva-normmu)/normsd),yva,seed,args.epochs)
            hist.to_csv(dest/f"history_{held}_{seed}.csv",index=False,encoding="utf-8-sig")
            (dest/f"best_epoch_{held}_{seed}.txt").write_text(str(best_epoch),encoding="utf-8")
            torch.save({"model":model,"state_dict":m.state_dict(),"input_mean":normmu,"input_std":normsd,"best_epoch":best_epoch,"seed":seed,"held_out":held},dest/f"checkpoint_{held}_{seed}.pt")
            with torch.no_grad(): pred=m(torch.from_numpy(((xte-normmu)/normsd).astype(np.float32))).numpy()
        mm,cases=metrics(pred,yte,te.reset_index(drop=True)); base=te.reset_index(drop=True); p=base[["case_id","sensor_id","sensor_string","layer","relative_time_s","load_type","load_kg","p1_x","p1_y","p2_x","p2_y"]].copy(); p["actual_ue"]=yte; p["predicted_ue"]=pred; p["model"]=model; p["seed"]=seed; p["held_out_type"]=held; p["split"]=tag
        p.to_csv(dest/f"pred_{held}_{seed}.csv",index=False,encoding="utf-8-sig")
        for z in mm: z.update({"model":model,"seed":seed,"held_out":held,"split":tag})
        for _,z in cases.iterrows():
            z=z.to_dict(); z.update({"model":model,"seed":seed,"held_out":held,"split":tag}); mm.append(z)
        return mm
    allm=[]
    if args.stage in ("A",):
        for fold in sorted(df.dev_fold.dropna().unique()):
            train=(df.dev_fold!=fold).to_numpy(); test=(df.dev_fold==fold).to_numpy()
            allm += sum((run_one(model,train,test,seed,"A",f"fold{int(fold)}") for model in ["A0-3","B0","B3"] for seed in ([20261008] if model=="A0-3" else seeds)),[])
    if args.stage in ("C1","both"):
        for held in sorted(df.load_type.unique()):
            train=(df.load_type!=held).to_numpy(); test=(df.load_type==held).to_numpy()
            allm += sum((run_one(model,train,test,seed,"C1",held) for model in ["A0-3","B0","B3"] for seed in ([20261008] if model=="A0-3" else seeds)),[])
    if args.stage in ("D1","both"):
        for held in sorted(df.sensor_string.unique()):
            train=(df.sensor_string!=held).to_numpy(); test=(df.sensor_string==held).to_numpy()
            allm += sum((run_one(model,train,test,seed,"D1",held) for model in ["A0-3","B0","B3"] for seed in ([20261008] if model=="A0-3" else seeds)),[])
    m=pd.DataFrame(allm); m.to_csv(out/"metrics_long.csv",index=False,encoding="utf-8-sig")
    if len(m):
        m.groupby(["split","model","held_out","phase"],as_index=False).rmse_ue.mean().to_csv(out/"metrics_summary.csv",index=False,encoding="utf-8-sig")
    (out/"RUN_MANIFEST.json").write_text(json.dumps({"protocol":"2026-10-09","version":args.version,"stage":args.stage,"epochs":args.epochs,"seeds":seeds,"fixed_test_opened":False,"root":str(ROOT)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(m.groupby(["split","model","phase"]).rmse_ue.mean().to_string())
if __name__=="__main__": main()
