from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F1_converged_dev_oof")
import protocol_20261009_corrected_runner as runner

BASE=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected")
F2=BASE/"F2_leave_type_out"; OUT=BASE/"F2_5_F3_5_audit"; CUR=OUT/"curves"; CUR.mkdir(parents=True,exist_ok=True)
DATA=runner.ROOT
df=pd.read_csv(DATA/"film_dynamic_training_table.csv"); df=df[df.split_role.eq("development")].copy(); df,cols=runner.add_features(df,"G1"); df[cols]=df[cols].apply(pd.to_numeric,errors="coerce").fillna(0)
mapd=pd.read_csv(DATA/"Reorganized_Sensor_Mapping.csv")[["sensor_id","x_cm","y_cm","z_cm","layer"]]

def read(split,model,held):
    root=F2 if split=="C1" else BASE/"F3_sensor_string_holdout"; sub=root/("phase_C_leave_type" if split=="C1" else "phase_D_sensor_holdout")/model
    parts=[]
    for f in sub.glob(f"pred_{held}_*.csv"):
        seed=int(f.stem.rsplit("_",1)[1]); z=pd.read_csv(f); z["seed"]=seed; parts.append(z)
    return pd.concat(parts,ignore_index=True)

def signed_curve(name,d):
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520"><rect width="100%" height="100%" fill="white"/><text x="500" y="24" text-anchor="middle" font-size="16">{name}</text>']
    W,H,ml,mt,mr,mb=1000,520,70,35,20,55; t=d.relative_time_s.to_numpy(); ymin=min(d.actual_ue.min(),d.pred_b0.min(),d.pred_b3.min()); ymax=max(d.actual_ue.max(),d.pred_b0.max(),d.pred_b3.max()); ymax=max(ymax,ymin+1)
    px=lambda x:ml+(x/280)*(W-ml-mr); py=lambda y:mt+(ymax-y)/(ymax-ymin)*(H-mt-mb)
    for col,color,label in [("actual_ue","#111","actual"),("pred_b0","#1f77b4","B0"),("pred_b3","#d62728","B3")]:
        q=d.sort_values("relative_time_s"); pts=" ".join(f"{px(x):.1f},{py(y):.1f}" for x,y in zip(q.relative_time_s,q[col])); lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="820" y="{55+20*(("actual_ue","pred_b0","pred_b3").index(col))}" fill="{color}" font-size="12">{label}</text>')
    lines.append(f'<line x1="{px(90)}" y1="{mt}" x2="{px(90)}" y2="{H-mb}" stroke="#000" stroke-dasharray="5,5"/><line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="#222"/><text x="500" y="510" text-anchor="middle">relative_time_s</text></svg>'); Path(name).write_text("".join(lines),encoding="utf-8")

def compare_type(held):
    b0=read("C1","B0",held); b3=read("C1","B3",held); keys=["case_id","sensor_id","relative_time_s","seed"]
    z=b0.merge(b3,on=keys,suffixes=("_b0","_b3"),how="inner"); z=z.merge(mapd,on="sensor_id",suffixes=("","_map")); z["e_b0"]=z.actual_ue_b0-z.predicted_ue_b0; z["e_b3"]=z.actual_ue_b3-z.predicted_ue_b3; z["delta_rmse_point"]=np.abs(z.e_b3)-np.abs(z.e_b0); z["signed_bias_b0"]=z.predicted_ue_b0-z.actual_ue_b0; z["signed_bias_b3"]=z.predicted_ue_b3-z.actual_ue_b3; z["d1"]=np.hypot(z.x_cm-z.p1_x_b0,z.y_cm-z.p1_y_b0); z["d2"]=np.hypot(z.x_cm-z.p2_x_b0.fillna(z.p1_x_b0),z.y_cm-z.p2_y_b0.fillna(z.p1_y_b0)); z["dmin"]=np.minimum(z.d1,z.d2); z["amp_bin"]=pd.cut(np.abs(z.actual_ue_b0),[-np.inf,5,20,50,100,200,np.inf],labels=["<5","5-20","20-50","50-100","100-200",">200"]); z["time_bin"]=pd.cut(z.relative_time_s,[-1,30,60,90,150,210,280],labels=["0-30","31-60","61-90","91-150","151-210","211-280"]); z.to_csv(OUT/f"{held}_pointwise.csv",index=False,encoding="utf-8-sig")
    def agg(group): return pd.Series({"n":len(group),"rmse_b0":np.sqrt(np.mean(group.e_b0**2)),"rmse_b3":np.sqrt(np.mean(group.e_b3**2)),"delta_b3_minus_b0":np.sqrt(np.mean(group.e_b3**2))-np.sqrt(np.mean(group.e_b0**2)),"bias_b0":group.signed_bias_b0.mean(),"bias_b3":group.signed_bias_b3.mean(),"peak_abs_error_b0":np.max(np.abs(group.e_b0)),"peak_abs_error_b3":np.max(np.abs(group.e_b3))})
    for name,grp in [("layer",z.groupby("layer")),("sensor_string",z.groupby("sensor_string_b0")),("seed",z.groupby("seed")),("time_bin",z.groupby("time_bin",observed=True)),("amp_bin",z.groupby("amp_bin",observed=True))]: aggout=grp.apply(agg).reset_index(); aggout.to_csv(OUT/f"{held}_{name}_decomposition.csv",index=False,encoding="utf-8-sig")
    z["dmin_bin"]=pd.cut(z.dmin,[-np.inf,5,10,20,40,np.inf],labels=["<5","5-10","10-20","20-40",">40"]); z.groupby("dmin_bin",observed=True).apply(agg).reset_index().to_csv(OUT/f"{held}_distance_decomposition.csv",index=False,encoding="utf-8-sig")
    no_sal=z[~z.sensor_string_b0.eq("SAL2")]; pd.DataFrame([{"held_out":held,"all_rmse_b0":np.sqrt(np.mean(z.e_b0**2)),"all_rmse_b3":np.sqrt(np.mean(z.e_b3**2)),"all_delta":np.sqrt(np.mean(z.e_b3**2))-np.sqrt(np.mean(z.e_b0**2)),"without_sal2_rmse_b0":np.sqrt(np.mean(no_sal.e_b0**2)),"without_sal2_rmse_b3":np.sqrt(np.mean(no_sal.e_b3**2)),"without_sal2_delta":np.sqrt(np.mean(no_sal.e_b3**2))-np.sqrt(np.mean(no_sal.e_b0**2))}]).to_csv(OUT/f"{held}_sal2_diagnostic.csv",index=False,encoding="utf-8-sig")
    for sid in ["SAL2","SAT2"]:
        q=z[z.sensor_string_b0.eq(sid)].groupby("relative_time_s",as_index=False).agg(actual_ue=("actual_ue_b0","mean"),pred_b0=("predicted_ue_b0","mean"),pred_b3=("predicted_ue_b3","mean")); signed_curve(str(CUR/f"{held}_{sid}.svg"),q)
    return z

zas=compare_type("DualAsymmetric"); zsy=compare_type("DualSymmetric")

# Swap-order sensitivity using saved B3/B0 checkpoints and the same case-bound features.
import torch
swap=[]
for held in ["DualAsymmetric","DualSymmetric"]:
    for model in ["B0","B3"]:
        sub=BASE/"F2_leave_type_out"/"phase_C_leave_type"/model
        for seed in [20261008,20261009,20261010]:
            ck=next(sub.glob(f"checkpoint_{held}_{seed}.pt"),None)
            if ck is None: continue
            state=torch.load(ck,map_location="cpu",weights_only=False); MLP,FiLM=runner.torch_models(len(cols),len(cols)-4,4); net=(MLP() if model=="B0" else FiLM()); net.load_state_dict(state["state_dict"]); net.eval()
            q=df[df.load_type.eq(held)].copy(); x=q[cols].to_numpy(np.float32); x2=q.copy(); x2["p1_x"],x2["p2_x"]=q.p2_x.to_numpy(),q.p1_x.to_numpy(); x2["p1_y"],x2["p2_y"]=q.p2_y.to_numpy(),q.p1_y.to_numpy(); x2,_=runner.add_features(x2,"G1"); x2[cols]=x2[cols].apply(pd.to_numeric,errors="coerce").fillna(0); x2=x2[cols].to_numpy(np.float32); mu=np.asarray(state["input_mean"]); sd=np.asarray(state["input_std"]); y=q.response_ue.to_numpy(float)
            with torch.no_grad(): p=net(torch.from_numpy(((x-mu)/sd).astype(np.float32))).numpy(); ps=net(torch.from_numpy(((x2-mu)/sd).astype(np.float32))).numpy()
            swap.append({"held_out":held,"model":model,"seed":seed,"rmse_original":float(np.sqrt(np.mean((y-p)**2))),"rmse_swapped":float(np.sqrt(np.mean((y-ps)**2))),"swap_prediction_rmse":float(np.sqrt(np.mean((p-ps)**2))),"swap_prediction_max_abs":float(np.max(np.abs(p-ps)))})
pd.DataFrame(swap).to_csv(OUT/"swap_sensitivity.csv",index=False,encoding="utf-8-sig")
manifest={"stages":["F2.5","F3.5"],"held_types":["DualAsymmetric","DualSymmetric"],"models":["B0","B3"],"sal2_retained_in_main_metrics":True,"swap_test":"checkpoint inference only; no retraining"}; (OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
root_manifest=BASE/"RUN_MANIFEST.json"
if root_manifest.exists():
    allm=json.loads(root_manifest.read_text(encoding="utf-8")); allm.setdefault("stages",{}).update({"F1.5":{"status":"completed","report":"F1_5_protocol_consistency/F1_5_REPORT.md"},"F2.5":{"status":"completed","report":"F2_5_F3_5_audit/f25_f35_report.md"},"F3.5":{"status":"completed","report":"F2_5_F3_5_audit/f25_f35_report.md"}}); root_manifest.write_text(json.dumps(allm,ensure_ascii=False,indent=2),encoding="utf-8")
print(pd.DataFrame(swap).to_string(index=False))
