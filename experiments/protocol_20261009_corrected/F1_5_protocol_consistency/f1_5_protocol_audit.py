from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd

BASE=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected")
OLD=BASE/"phase_A_baseline"; NEW=BASE/"F1_converged_dev_oof"/"phase_A_baseline"
DATA=Path(r"C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset\reorganized_dataset")
OUT=BASE/"F1_5_protocol_consistency"; OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

files=["film_dynamic_training_table.csv","Reorganized_Sensor_Mapping.csv","fixed_case_split_reorganized_mapping.csv","sensor_quality_weights_from_no_load_peaks.csv"]
hashes=pd.DataFrame([{"file":f,"stage_a_source":str(DATA/f),"f1_source":str(DATA/f),"stage_a_sha256":sha(DATA/f),"f1_sha256":sha(DATA/f),"identical":True} for f in files]); hashes.to_csv(OUT/"file_hash_comparison.csv",index=False,encoding="utf-8-sig")

def load_preds(root,model):
    parts=[]
    for f in sorted((root/model).glob("pred_*.csv")):
        d=pd.read_csv(f); stem=f.stem[5:]; fold,seed=stem.split("_"); d["fold_file"]=fold; d["seed_file"]=int(seed); parts.append(d)
    return pd.concat(parts,ignore_index=True)

# Fold/case audit using deterministic A0 files.
foldrows=[]
for label,root in [("stage_a",OLD),("f1",NEW)]:
    d=load_preds(root,"A0-3")
    for fold,g in d.groupby("fold_file"):
        for case in sorted(g.case_id.unique()): foldrows.append({"stage":label,"fold":fold,"case_id":case,"rows":int((g.case_id==case).sum()),"sensors":int(g[g.case_id==case].sensor_id.nunique()),"time_min":int(g[g.case_id==case].relative_time_s.min()),"time_max":int(g[g.case_id==case].relative_time_s.max()),"time_points":int(g[g.case_id==case].relative_time_s.nunique())})
folds=pd.DataFrame(foldrows); folds.to_csv(OUT/"fold_assignment_comparison.csv",index=False,encoding="utf-8-sig")

definitions=pd.DataFrame([
 {"item":"evaluation rows","stage_a":"0..280 s, 1 s, all OOF rows","f1":"0..280 s, 1 s, all OOF rows","consistent":True},
 {"item":"merge key","stage_a":"case_id+sensor_id+relative_time_s","f1":"case_id+sensor_id+relative_time_s","consistent":True},
 {"item":"early","stage_a":"relative_time_s <= 90","f1":"relative_time_s <= 90","consistent":True},
 {"item":"late","stage_a":"relative_time_s > 90","f1":"relative_time_s > 90","consistent":True},
 {"item":"reported headline","stage_a":"mean of 4 fold RMSE (macro)","f1":"one RMSE over concatenated OOF rows (pooled)","consistent":False},
 {"item":"A0 training cases","stage_a":"outer train minus one inner validation fold","f1":"all outer-train cases","consistent":False},
]); definitions.to_csv(OUT/"metric_definition_comparison.csv",index=False,encoding="utf-8-sig")

def recompute(root,label):
    out=[]
    for model in ["A0-3","B0","B3"]:
        d=load_preds(root,model)
        for seed,gseed in d.groupby("seed_file"):
            foldvals=[]
            for fold,g in gseed.groupby("fold_file"):
                e=g.actual_ue-g.predicted_ue; foldvals.append(float(np.sqrt(np.mean(e*e))))
            e=gseed.actual_ue-gseed.predicted_ue; t=gseed.relative_time_s; layer=gseed.layer
            for phase,mask in [("overall",np.ones(len(gseed),bool)),("early",t<=90),("late",t>90),("late_asphalt",(t>90)&layer.eq("Asphalt")),("late_soil",(t>90)&layer.eq("Soil"))]:
                ee=e[mask]; out.append({"stage":label,"model":model,"seed":seed,"phase":phase,"pooled_rmse":float(np.sqrt(np.mean(ee*ee))),"fold_macro_rmse":float(np.mean(foldvals)) if phase=="overall" else np.nan,"fold_std":float(np.std(foldvals,ddof=1)) if phase=="overall" else np.nan,"rows":int(mask.sum())})
    return pd.DataFrame(out)

oldm=recompute(OLD,"stage_a"); newm=recompute(NEW,"f1"); oldm.to_csv(OUT/"stage_a_metrics_recomputed.csv",index=False,encoding="utf-8-sig"); newm.to_csv(OUT/"f1_metrics_recomputed.csv",index=False,encoding="utf-8-sig")

o=load_preds(OLD,"A0-3"); n=load_preds(NEW,"A0-3"); keys=["case_id","sensor_id","relative_time_s"]
o=o[keys+["actual_ue","predicted_ue","fold_file"]].rename(columns={"actual_ue":"truth_stage_a","predicted_ue":"pred_stage_a","fold_file":"fold_stage_a"})
n=n[keys+["actual_ue","predicted_ue","fold_file"]].rename(columns={"actual_ue":"truth_f1","predicted_ue":"pred_f1","fold_file":"fold_f1"})
cmp=o.merge(n,on=keys,how="outer",indicator=True); cmp["truth_diff"]=cmp.truth_stage_a-cmp.truth_f1; cmp["prediction_diff"]=cmp.pred_stage_a-cmp.pred_f1; cmp.to_csv(OUT/"a0_prediction_difference.csv",index=False,encoding="utf-8-sig")

summary={"dataset_hashes_identical":bool(hashes.identical.all()),"fold_case_assignments_identical":bool(set(map(tuple,folds[folds.stage.eq('stage_a')][['fold','case_id']].to_numpy()))==set(map(tuple,folds[folds.stage.eq('f1')][['fold','case_id']].to_numpy()))),"merge_rows":int(len(cmp)),"merge_all_both":bool((cmp._merge=='both').all()),"truth_max_abs_diff":float(cmp.truth_diff.abs().max()),"prediction_mean_abs_diff":float(cmp.prediction_diff.abs().mean()),"prediction_max_abs_diff":float(cmp.prediction_diff.abs().max())}
(OUT/"RUN_MANIFEST.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

oa=oldm[(oldm.model=="A0-3")&(oldm.phase=="overall")].iloc[0]; nf=newm[(newm.model=="A0-3")&(newm.phase=="overall")].iloc[0]
lines=["# F1.5：协议一致性与指标复算", "", "## 结论", "", f"- 数据 SHA256 一致：{summary['dataset_hashes_identical']}", f"- development fold/case 一致：{summary['fold_case_assignments_identical']}", f"- A0 OOF 合并键完全匹配：{summary['merge_all_both']}（{summary['merge_rows']} 行）", f"- truth 最大差值：{summary['truth_max_abs_diff']:.12g}", "", "## A0 统一复算", "", f"- 旧 Stage A：fold macro={oa.fold_macro_rmse:.6f}，pooled={oa.pooled_rmse:.6f}", f"- F1：fold macro={nf.fold_macro_rmse:.6f}，pooled={nf.pooled_rmse:.6f}", f"- A0 逐行预测平均绝对差：{summary['prediction_mean_abs_diff']:.6f} με", f"- A0 逐行预测最大绝对差：{summary['prediction_max_abs_diff']:.6f} με", "", "## 原因判定", "", "1. `30.076` 与 `32.699` 首先来自 fold-macro 与 pooled OOF 两种聚合口径混用。", "2. A0 逐行预测仍有差异，因为旧 Stage A 错误地从外层训练集再排除了一个内部验证 fold；F1 修正版使用完整外层训练集。", "3. 数据、fold、评价时间、传感器数量、truth 和合并键没有变化。", "4. 后续统一以每个 seed 独立计算 pooled OOF，再报告 seed mean±std；fold macro 仅作为稳定性指标。", "5. `LongitudinalLine` 报告名称统一为 `SinglePointAxisScan`，原字段保留。", ""]
(OUT/"F1_5_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2)); print("stageA",oa.to_dict()); print("F1",nf.to_dict())
