from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected")
def files(split): return OUT/("F2_leave_type_out" if split=="C1" else "F3_sensor_string_holdout")/"phase_C_leave_type" if split=="C1" else OUT/"F3_sensor_string_holdout"/"phase_D_sensor_holdout"
rows=[]; details=[]
for split,root in [("C1",OUT/"F2_leave_type_out"/"phase_C_leave_type"),("D1",OUT/"F3_sensor_string_holdout"/"phase_D_sensor_holdout")]:
    for md in root.iterdir():
        if not md.is_dir(): continue
        model=md.name
        for f in md.glob("pred_*.csv"):
            held,seed=f.stem[5:].rsplit("_",1); seed=int(seed); d=pd.read_csv(f); e=d.actual_ue.to_numpy()-d.predicted_ue.to_numpy(); t=d.relative_time_s.to_numpy(); l=d.layer.to_numpy()
            for ph,m in [("overall",np.ones(len(d),bool)),("early",t<=90),("late",t>90),("late_asphalt",(l=="Asphalt")&(t>90)),("late_soil",(l=="Soil")&(t>90))]:
                rows.append({"split":split,"model":model,"held_out":held,"seed":seed,"phase":ph,"sse":float(np.sum(e[m]**2)),"n":int(m.sum()),"rmse":float(np.sqrt(np.mean(e[m]**2)))})
            z=d.assign(e2=e*e)
            for sid,g in z.groupby("sensor_string"):
                details.append({"split":split,"model":model,"held_out":held,"seed":seed,"sensor_string":sid,"layer":g.layer.iloc[0],"n":len(g),"rmse":float(np.sqrt(g.e2.mean()))})
r=pd.DataFrame(rows); det=pd.DataFrame(details); r.to_csv(OUT/"F2_F3_file_metrics.csv",index=False,encoding="utf-8-sig"); det.to_csv(OUT/"F2_F3_sensor_metrics.csv",index=False,encoding="utf-8-sig")
pooled=r.groupby(["split","model","held_out","phase"],as_index=False)[["sse","n"]].sum(); pooled["pooled_rmse"]=np.sqrt(pooled.sse/pooled.n); pooled.to_csv(OUT/"F2_F3_pooled_metrics.csv",index=False,encoding="utf-8-sig")
seed=r.groupby(["split","model","held_out","seed","phase"],as_index=False).apply(lambda x: pd.Series({"pooled_rmse":np.sqrt(x.sse.sum()/x.n.sum())}),include_groups=False).reset_index(drop=True); seed.to_csv(OUT/"F2_F3_seed_metrics.csv",index=False,encoding="utf-8-sig")
macro=pooled[pooled.phase.eq("overall")].groupby(["split","model"],as_index=False).pooled_rmse.agg(["mean","std"]).reset_index(); macro.to_csv(OUT/"F2_F3_model_summary.csv",index=False,encoding="utf-8-sig")
d1=det[det.split.eq("D1")].groupby(["model","held_out"],as_index=False).agg(rmse_mean=("rmse","mean"),n=("n","sum")); d1.to_csv(OUT/"F3_sensor_breakdown.csv",index=False,encoding="utf-8-sig")
lines=["# F2/F3：正式荷载类型与传感器串外推", "", "- 普通 MSE；充分收敛设置；fixed-test closed。", "- A0-3 deterministic；B0/B3 三个 seed 分开计算。", "", "## 模型平均", "", macro.to_string(index=False), "", "## F2 各留出荷载类型 overall pooled RMSE", "", pooled[(pooled.split=="C1")&(pooled.phase=="overall")].to_string(index=False), "", "## F3 各传感器串 breakdown", "", d1.to_string(index=False), "", "## 当前判断", "", "1. B3 的 F2 整体外推约 30.923，低于 A0-3 但高于 B0；因此不能选为无条件主模型。", "2. B3 的 F3 整体空间外推约 20.784，优于 B0 和 A0-3；主要收益来自土层，沥青层需要单独看。", "3. SAL2 仍是最差传感器串之一，主指标没有删除 SAL2。", "4. 阶段 E 双重外推暂不启动，先完成 F2/F3 的 DualAsymmetric、SAL2/SAT2 专项分解和协议冻结。", ""]
(OUT/"F2_F3_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
manifest={"protocol":"2026-10-09","dataset":"guochuang_fbg_pavement_dataset/reorganized_dataset","loss":"ordinary MSE","fixed_test_opened":False,"stages":{"F0":{"status":"completed","sal2_sat2_curves":10},"F1":{"status":"completed","max_epochs":150,"min_epochs":50,"patience":20,"seeds":[20261008,20261009,20261010]},"F2":{"status":"completed","prediction_files":35},"F3":{"status":"completed","prediction_files":49},"F4":{"status":"not_started","reason":"F2/F3 reports retained before double extrapolation"},"F5":{"status":"not_started"},"F6":{"status":"not_started"},"F7":{"status":"closed_until_freeze"}}}
(OUT/"RUN_MANIFEST.json").write_text(__import__('json').dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(macro.to_string(index=False))
