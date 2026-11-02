from pathlib import Path
import numpy as np,pandas as pd,json
OUT=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F1_converged_dev_oof")
rows=[]; cases=[]; sensors=[]; pred_store={}
for modeldir in (OUT/"phase_A_baseline").iterdir():
    if not modeldir.is_dir(): continue
    model=modeldir.name
    for f in modeldir.glob("pred_*.csv"):
        stem=f.stem.replace("pred_",""); fold,seed=stem.split("_"); seed=int(seed)
        d=pd.read_csv(f); e=d.actual_ue.to_numpy()-d.predicted_ue.to_numpy(); t=d.relative_time_s.to_numpy(); layer=d.layer.to_numpy()
        pred_store.setdefault((model,seed),[]).append(d)
        for ph,m in [("overall",np.ones(len(d),bool)),("early",t<=90),("late",t>90),("late_asphalt",(layer=="Asphalt")&(t>90)),("late_soil",(layer=="Soil")&(t>90))]:
            rows.append({"model":model,"fold":fold,"seed":seed,"phase":ph,"rmse_ue":float(np.sqrt(np.mean(e[m]**2)))})
        z=d.assign(e2=e*e)
        for key,g in z.groupby("case_id"):
            cases.append({"model":model,"fold":fold,"seed":seed,"case_id":key,"rmse_ue":float(np.sqrt(g.e2.mean()))})
        for key,g in z.groupby("sensor_string"):
            sensors.append({"model":model,"fold":fold,"seed":seed,"sensor_string":key,"layer":g.layer.iloc[0],"n_sensors":int(g.sensor_id.nunique()),"rmse_ue":float(np.sqrt(g.e2.mean()))})
m=pd.DataFrame(rows); c=pd.DataFrame(cases); s=pd.DataFrame(sensors)
m.to_csv(OUT/"metrics_by_fold_seed.csv",index=False,encoding="utf-8-sig"); c.to_csv(OUT/"metrics_by_case.csv",index=False,encoding="utf-8-sig"); s.to_csv(OUT/"metrics_by_sensor_string.csv",index=False,encoding="utf-8-sig")
seed_rows=[]
for (model,seed),parts in pred_store.items():
    d=pd.concat(parts,ignore_index=True); e=d.actual_ue.to_numpy()-d.predicted_ue.to_numpy(); t=d.relative_time_s.to_numpy(); layer=d.layer.to_numpy()
    for ph,mask in [("overall",np.ones(len(d),bool)),("early",t<=90),("late",t>90),("late_asphalt",(layer=="Asphalt")&(t>90)),("late_soil",(layer=="Soil")&(t>90))]:
        seed_rows.append({"model":model,"seed":seed,"phase":ph,"pooled_rmse_ue":float(np.sqrt(np.mean(e[mask]**2)))})
seed=pd.DataFrame(seed_rows); fold=m.groupby(["model","fold","phase"],as_index=False).rmse_ue.mean()
seed_summary=seed.groupby(["model","phase"],as_index=False).pooled_rmse_ue.agg(["mean","std"]).reset_index(); fold_summary=fold.groupby(["model","phase"],as_index=False).rmse_ue.agg(["mean","std"]).reset_index()
seed.to_csv(OUT/"pooled_oof_by_seed.csv",index=False,encoding="utf-8-sig")
seed_summary.to_csv(OUT/"seed_summary.csv",index=False,encoding="utf-8-sig"); fold_summary.to_csv(OUT/"fold_summary.csv",index=False,encoding="utf-8-sig")
worst_case=c.groupby(["model","case_id"],as_index=False).rmse_ue.mean().sort_values("rmse_ue",ascending=False).groupby("model",as_index=False).first(); worst_sensor=s.groupby(["model","sensor_string","layer"],as_index=False).rmse_ue.mean().sort_values("rmse_ue",ascending=False).groupby("model",as_index=False).first(); worst_case.to_csv(OUT/"worst_case.csv",index=False,encoding="utf-8-sig"); worst_sensor.to_csv(OUT/"worst_sensor_string.csv",index=False,encoding="utf-8-sig")
sal=s[s.sensor_string.eq("SAL2")].groupby("model",as_index=False).rmse_ue.mean(); sal.to_csv(OUT/"SAL2_metrics.csv",index=False,encoding="utf-8-sig")
overall=seed[seed.phase.eq("overall")].groupby("model").pooled_rmse_ue.agg(["mean","std"])
lines=["# F1：A0/B0/B3 充分收敛 development OOF", "", "- 数据：新数据包 `guochuang_fbg_pavement_dataset/reorganized_dataset`", "- 训练采样：5 s；评价采样：1 s；loss：ordinary MSE", "- 训练上限：150 epochs；至少 50 epochs；patience=20", "- fixed-test：closed", "", "## Seed mean ± std（每个 seed 先独立计算 pooled OOF，再汇总）", "", "| model | overall mean | std |", "|---|---:|---:|"]
for model,r in overall.iterrows(): lines.append(f"| {model} | {r['mean']:.3f} | {r['std']:.3f} |")
lines += ["", "## 最差案例和传感器串", "", "```text", worst_case.to_string(index=False), "", worst_sensor.to_string(index=False), "", "SAL2:", sal.to_string(index=False), "```", "", "## 解释边界", "", "1. 所有传感器仍保留在主指标；SAL2 仅作专项诊断。", "2. B3 的土层改善不能自动等同于全结构最优。", "3. F2/F3 正式外推应使用本轮充分收敛 checkpoint 和同一评价脚本。"]
(OUT/"F1_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(overall)
