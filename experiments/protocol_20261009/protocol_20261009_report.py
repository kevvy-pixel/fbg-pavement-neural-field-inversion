from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009")

def summarize_file(f, split, model, held, seed):
    d=pd.read_csv(f); e=d.actual_ue.to_numpy()-d.predicted_ue.to_numpy(); t=d.relative_time_s.to_numpy(); layer=d.layer.to_numpy()
    rows=[]
    for phase,m in [("overall",np.ones(len(d),bool)),("early",t<=90),("late",t>90),("late_asphalt",(layer=="Asphalt")&(t>90)),("late_soil",(layer=="Soil")&(t>90))]:
        if m.any(): rows.append({"split":split,"model":model,"held_out":held,"seed":seed,"phase":phase,"rmse_ue":float(np.sqrt(np.mean(e[m]**2)))})
    c=d.assign(e2=e*e).groupby("case_id").e2.mean().pow(.5).reset_index(name="rmse_ue"); c["split"]=split;c["model"]=model;c["held_out"]=held;c["seed"]=seed
    return rows,c

allrows=[]; cases=[]
pooled=[]
for split,sub in [("C1",OUT/"phase_C_leave_type"),("D1",OUT/"phase_D_sensor_holdout")]:
    for f in sub.glob("*/pred_*.csv"):
        model=f.parent.name; stem=f.stem[5:]; held,seed=stem.rsplit("_",1)
        r,c=summarize_file(f,split,model,held,int(seed)); allrows += r; cases.append(c)
        raw=pd.read_csv(f); err=raw.actual_ue.to_numpy()-raw.predicted_ue.to_numpy();
        for phase,mask in [("overall",np.ones(len(raw),bool)),("early",raw.relative_time_s.to_numpy()<=90),("late",raw.relative_time_s.to_numpy()>90)]:
            pooled.append({"split":split,"model":model,"held_out":held,"seed":int(seed),"phase":phase,"sse":float(np.sum(err[mask]**2)),"n":int(mask.sum())})
m=pd.DataFrame(allrows); c=pd.concat(cases,ignore_index=True)
m.to_csv(OUT/"metrics_file_level.csv",index=False,encoding="utf-8-sig")
c.to_csv(OUT/"case_metrics.csv",index=False,encoding="utf-8-sig")
agg=m.groupby(["split","model","held_out","phase"],as_index=False).rmse_ue.agg(["mean","std","count"]).reset_index(); agg.to_csv(OUT/"metrics_by_holdout.csv",index=False,encoding="utf-8-sig")
overall=m[m.phase=="overall"].groupby(["split","model","held_out"],as_index=False).rmse_ue.mean()
pp=pd.DataFrame(pooled).groupby(["split","model","held_out","phase"],as_index=False)[["sse","n"]].sum(); pp["pooled_rmse_ue"]=np.sqrt(pp.sse/pp.n); pp.to_csv(OUT/"pooled_metrics.csv",index=False,encoding="utf-8-sig")
worst=c.groupby(["split","model","held_out","case_id"],as_index=False).rmse_ue.mean().sort_values("rmse_ue",ascending=False).groupby(["split","model","held_out"],as_index=False).first()
worst.to_csv(OUT/"worst_case_metrics.csv",index=False,encoding="utf-8-sig")

lines=["# Protocol 2026-10-09：阶段 C+D 报告", "", "## 执行边界", "", "- 输入版本：G1 几何输入；荷载类型标签未作为主输入。", "- development-only；fixed-test 5 个工况始终未打开。", "- C1：5 类荷载类型留出；D1：7 条天然传感器串留出。", "- 训练轮数为计算预算版：C1=5 epochs，D1=3 epochs；因此本报告用于协议筛选，不替代最终充分收敛训练。", "- 所有预测文件均保留，按完整案例/传感器串留出，未按时间点随机拆分。", "- `pooled_metrics.csv` 是跨种子拼接后的 pooled 指标；`metrics_by_holdout.csv` 是文件级 seed 统计。", ""]
for split in ["C1","D1"]:
    lines += [f"## {split} 汇总", "", "| 模型 | 留出对象 | overall RMSE 均值 | std | 最差 case |", "|---|---|---:|---:|---:|"]
    z=overall[overall.split==split]
    for _,r in z.groupby(["model","held_out"]).rmse_ue.agg(["mean","std"]).reset_index().iterrows():
        w=worst[(worst.split==split)&(worst.model==r.model)&(worst.held_out==r.held_out)]
        wc=float(w.rmse_ue.iloc[0]) if len(w) else float("nan")
        lines.append(f"| {r.model} | {r.held_out} | {r['mean']:.3f} | {r['std'] if pd.notna(r['std']) else 0:.3f} | {wc:.3f} |")
    lines += ["", "### 按阶段/层位均值", "", "```text", agg[agg.split==split].to_string(index=False), "```", ""]
lines += ["## 当前判断", "", "1. C1 和 D1 的冻结划分及完整预测已生成。", "2. G1 版本下，DualAsymmetric 和 SAL2 等对象仍是高风险区域。", "3. D1 中 B3 在多数土层传感器串上较好，但 SAL2 的沥青层外推仍困难。", "4. 本轮使用短训练预算，下一步应对 C1 中排名靠前的模型/荷载类型补充充分收敛训练和多种子复核。", "5. 在 C1、D1 充分收敛结果确认前，不进入阶段 E/F，也不打开 fixed-test。", ""]
(OUT/"PHASE_C_D_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
manifest={"protocol":"2026-10-09","input_version":"G1","stages":{"C1":{"completed":True,"epochs":5,"prediction_files":35},"D1":{"completed":True,"epochs":3,"prediction_files":49}},"fixed_test_opened":False,"models":["A0-3","B0","B3"]}
(OUT/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(overall.groupby(["split","model"]).rmse_ue.mean().to_string())
