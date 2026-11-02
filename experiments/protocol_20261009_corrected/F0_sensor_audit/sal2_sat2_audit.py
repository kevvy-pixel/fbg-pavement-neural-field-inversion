from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(r"C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset")
DATA=ROOT/"reorganized_dataset"
OUT=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F0_sensor_audit")
CUR=OUT/"SAL2_SAT2_curves"; CUR.mkdir(parents=True,exist_ok=True)

mapping=pd.read_csv(DATA/"Reorganized_Sensor_Mapping.csv")
df=pd.read_csv(DATA/"film_dynamic_training_table.csv")
weights=pd.read_csv(DATA/"sensor_quality_weights_from_no_load_peaks.csv")
target=mapping[mapping.sensor_string.isin(["SAL2","SAT2"])].copy()
rows=[]
for _,r in target.iterrows():
    same=mapping[(mapping.sensor_string==r.sensor_string)&(mapping.sensor_id!=r.sensor_id)].copy(); same["dist"]=np.hypot(same.x_cm-r.x_cm,same.y_cm-r.y_cm); same=same.sort_values(["dist","sensor_index"]).head(2)
    near=mapping[(mapping.layer==r.layer)&(mapping.sensor_id!=r.sensor_id)].copy(); near["dist"]=np.hypot(near.x_cm-r.x_cm,near.y_cm-r.y_cm); near=near.sort_values(["dist","sensor_id"]).head(3)
    rows.append({"sensor_id":r.sensor_id,"channel":r.channel,"sensor_string":r.sensor_string,"layer":r.layer,"direction":r.direction,"x_cm":r.x_cm,"y_cm":r.y_cm,"z_cm":r.z_cm,"unit":"cm","same_string_neighbors":";".join(same.sensor_id),"nearest_same_layer":";".join(near.sensor_id),"nearest_dist_cm":";".join(f"{x:.2f}" for x in near.dist)})
map_a=pd.DataFrame(rows); map_a.to_csv(OUT/"sensor_mapping_audit.csv",index=False,encoding="utf-8-sig")

td=df[df.sensor_string.isin(["SAL2","SAT2"])].copy(); td["phase"]=np.where(td.relative_time_s<=90,"early","late")
audit=[]
for (case,sid),g in td.groupby(["case_id","sensor_id"],sort=False):
    gg=g.sort_values("relative_time_s"); y=gg.response_ue.to_numpy(); t=gg.relative_time_s.to_numpy(); early=y[t<=90]; late=y[t>90]
    row=gg.iloc[0]; audit.append({"case_id":case,"load_type":row.load_type,"load_kg":row.load_kg,"sensor_id":sid,"sensor_string":row.sensor_string,"layer":row.layer,"x_cm":row.x_cm,"y_cm":row.y_cm,"z_cm":row.z_cm,"overall_mean_ue":float(y.mean()),"overall_std_ue":float(y.std()),"min_ue":float(y.min()),"max_ue":float(y.max()),"max_abs_ue":float(np.max(np.abs(y))),"max_abs_time_s":float(t[np.argmax(np.abs(y))]),"early_mean_ue":float(early.mean()),"late_mean_ue":float(late.mean()),"early_abs_mean_ue":float(np.abs(early).mean()),"late_abs_mean_ue":float(np.abs(late).mean()),"sign_change_count":int(np.sum(np.sign(y[1:])*np.sign(y[:-1])<0))})
resp_a=pd.DataFrame(audit); resp_a.to_csv(OUT/"sensor_response_audit.csv",index=False,encoding="utf-8-sig")

def write_svg(glt, ss, path):
    ids=list(mapping.loc[mapping.sensor_string.eq(ss),"sensor_id"]); W,H=1000,500; ml,mt,mr,mb=70,35,20,55
    series=[]; xmin,xmax=0,280; ymin,ymax=np.inf,-np.inf
    for sid in ids:
        q=glt[glt.sensor_id.eq(sid)].sort_values("relative_time_s")
        if len(q):
            x=q.relative_time_s.to_numpy(float); y=q.response_ue.to_numpy(float); series.append((sid,x,y)); ymin=min(ymin,y.min()); ymax=max(ymax,y.max())
    if not series: return
    if ymax<=ymin: ymax=ymin+1
    def px(x): return ml+(x-xmin)/(xmax-xmin)*(W-ml-mr)
    def py(y): return mt+(ymax-y)/(ymax-ymin)*(H-mt-mb)
    colors=["#1f77b4","#d62728","#2ca02c","#9467bd","#ff7f0e","#17becf","#8c564b","#e377c2"]
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>',f'<text x="{W/2}" y="22" text-anchor="middle" font-size="16">{lt} — {ss}</text>',f'<line x1="{ml}" y1="{py(0)}" x2="{W-mr}" y2="{py(0)}" stroke="#888" stroke-dasharray="4,4"/>',f'<line x1="{px(90)}" y1="{mt}" x2="{px(90)}" y2="{H-mb}" stroke="#000" stroke-dasharray="5,5"/><text x="{px(90)+4}" y="{mt+14}" font-size="11">early/late 90 s</text>']
    for i,(sid,x,y) in enumerate(series):
        pts=" ".join(f"{px(a):.1f},{py(b):.1f}" for a,b in zip(x,y)); yy=mt+18+(i%8)*15; s += [f'<polyline points="{pts}" fill="none" stroke="{colors[i%len(colors)]}" stroke-width="1.2"/><text x="{W-170+(i//8)*85}" y="{yy}" font-size="10" fill="{colors[i%len(colors)]}">{sid}</text>']
    s += [f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{H-mb}" stroke="#222"/><line x1="{ml}" y1="{H-mb}" x2="{W-mr}" y2="{H-mb}" stroke="#222"/><text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="12">relative_time_s</text><text x="18" y="{H/2}" transform="rotate(-90 18,{H/2})" text-anchor="middle" font-size="12">response_ue</text></svg>']
    Path(path).write_text("".join(s),encoding="utf-8")
for lt,glt in df.groupby("load_type"):
    for ss in ["SAL2","SAT2"]:
        ids=list(mapping.loc[mapping.sensor_string.eq(ss),"sensor_id"])
        if not ids: continue
        write_svg(glt,ss,CUR/f"{lt}_{ss}.svg")

# Quality/peak audit for D2-CH1 and SAL2/SAT2
wcols=list(weights.columns); wsel=weights[weights.astype(str).apply(lambda c:c.str.contains("D2_CH1|SAL2|SAT2|D2-CH1",case=False,regex=True)).any(axis=1)]
summary={"mapping_rows":int(len(map_a)),"response_rows":int(len(resp_a)),"cases":int(td.case_id.nunique()),"load_types":sorted(td.load_type.unique()),"quality_columns":wcols,"quality_matching_rows":int(len(wsel)),"sal2_sensor_count":int((mapping.sensor_string=="SAL2").sum()),"sat2_sensor_count":int((mapping.sensor_string=="SAT2").sum())}
(OUT/"F0_MANIFEST.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

by_type=resp_a.groupby(["load_type","sensor_string"],as_index=False).agg(max_abs_ue_mean=("max_abs_ue","mean"),late_abs_mean_ue=("late_abs_mean_ue","mean"),max_abs_ue_max=("max_abs_ue","max"),sign_change_mean=("sign_change_count","mean"))
by_type.to_csv(OUT/"response_summary_by_type.csv",index=False,encoding="utf-8-sig")
lines=["# SAL2/SAT2 专项审计（F0）","", "## 审计范围", "", "- 主数据：新数据包 `reorganized_dataset/film_dynamic_training_table.csv`。", "- 当前动态训练表覆盖 0–280 s；原始 Demod 文本已保留在 `raw_data/`，但其仪器导出格式不直接作为模型输入。", "- 主指标继续保留全部传感器；本报告只提供 SAL2/SAT2 专项诊断，不删除异常传感器。", "", "## 映射和空间关系", "", map_a.to_string(index=False), "", "## 响应异常摘要", "", by_type.to_string(index=False), "", "## D2-CH1/质量权重", "", f"质量权重文件列：{', '.join(wcols)}", f"匹配到 SAL2/SAT2/D2_CH1 的行数：{len(wsel)}", "", "## 当前结论", "", "1. SAL2 位于 D2_CH1，8 个点，沥青层，纵向方向，坐标 x=37–72 cm、y=18 cm。", "2. SAT2 位于 D2_CH2，6 个点，沥青层，横向方向，坐标 x=42.5 cm、y=12.5–37.5 cm。", "3. 曲线按荷载类型分别保存，虚线 90 s 用于 early/late 分段。", "4. SAL2/SAT2 不因误差较大而从主指标删除；后续同时报告全传感器与专项诊断。", "5. 需要结合质量权重原始峰文件进一步确认 D2_CH1 峰身份；在此之前 raw weighted RMSE 仅作诊断。", ""]
(OUT/"SAL2_SAT2_AUDIT_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(summary); print(by_type.to_string(index=False))
