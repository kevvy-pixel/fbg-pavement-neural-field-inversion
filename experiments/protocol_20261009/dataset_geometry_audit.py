from pathlib import Path
import json,re
import numpy as np,pandas as pd

ROOT=Path(r"C:\Users\27356\Documents\Codex\2026-10-09\c-users-27356-documents-codex-2026\outputs\guochuang_fbg_pavement_dataset")
DATA=ROOT/"reorganized_dataset"; SP=ROOT/"spatial_model"
OUT=Path(r"D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009\data_audit")
OUT.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(DATA/"film_dynamic_training_table.csv")
events=df.groupby("case_id",sort=False).first().reset_index()
sp=pd.read_csv(SP/"load_points_coordinates.csv")
sp["X_cm"]=sp.X_m*100; sp["Y_cm"]=sp.Y_m*100; sp["Z_cm"]=sp.Z_m*100

def pts(rows): return sorted([(round(float(x),4),round(float(y),4)) for x,y in rows])
def spatial_expected(r):
    lt=str(r.load_type); cid=str(r.case_id)
    if lt=="SingleCenter": return [(60,25)],"SinglePoint"
    if lt=="SingleEccentric": return [(60,15)],"SinglePoint"
    if lt=="DualSymmetric": return [(60,15),(60,35)],"DualPoint"
    if lt=="DualAsymmetric": return [(40,15),(80,35)],"DualPoint"
    if lt=="LongitudinalLine":
        m=re.search(r"(\d+(?:\.\d+)?)cm",cid)
        x=float(m.group(1)) if m else float(r.p1_x)
        return [(x,25)],"SinglePointAxisScan"
    return [],"Unknown"

rows=[]
for _,r in events.iterrows():
    tr=[]
    if pd.notna(r.p1_x): tr.append((r.p1_x,r.p1_y))
    if pd.notna(r.p2_x): tr.append((r.p2_x,r.p2_y))
    exp,sem=spatial_expected(r)
    status="match" if pts(tr)==pts(exp) else "mismatch"
    note="training table retained as authoritative case-bound geometry"
    if r.load_type=="LongitudinalLine": note += "; semantic correction: SinglePointAxisScan, not line load"
    if status=="mismatch": note += "; spatial_model coordinates differ and require source audit"
    rows.append({"case_id":r.case_id,"load_type":r.load_type,"training_p1_x_cm":r.p1_x,"training_p1_y_cm":r.p1_y,"training_p2_x_cm":r.p2_x,"training_p2_y_cm":r.p2_y,"spatial_p1_x_cm":exp[0][0] if exp else np.nan,"spatial_p1_y_cm":exp[0][1] if exp else np.nan,"spatial_p2_x_cm":exp[1][0] if len(exp)>1 else np.nan,"spatial_p2_y_cm":exp[1][1] if len(exp)>1 else np.nan,"geometry_status":status,"chosen_source":"film_dynamic_training_table.csv","geometry_semantics":sem,"notes":note})
rec=pd.DataFrame(rows); rec.to_csv(OUT/"load_geometry_reconciliation.csv",index=False,encoding="utf-8-sig")

sm=pd.read_csv(DATA/"Reorganized_Sensor_Mapping.csv"); af=pd.read_csv(SP/"all_fbg_points_coordinates.csv")
sensor_audit={"reorganized_sensor_rows":int(len(sm)),"reorganized_structural_sensors":int((sm.layer!="Temperature").sum()),"spatial_point_rows":int(len(af)),"layer_counts":sm.groupby("layer").size().to_dict()}
integrity={"training_rows":int(len(df)),"unique_cases":int(df.case_id.nunique()),"unique_structural_sensors":int(df.sensor_id.nunique()),"time_min":int(df.relative_time_s.min()),"time_max":int(df.relative_time_s.max()),"time_points_per_case_sensor":int(df.groupby(["case_id","sensor_id"]).relative_time_s.nunique().median()),"missing_response":int(df.response_ue.isna().sum()),"temperature_rows":int((df.layer=="Temperature").sum()),"development_cases":int((df.groupby("case_id").split_role.first()=="development").sum()),"fixed_test_cases":int((df.groupby("case_id").split_role.first()=="fixed_test").sum())}
manifest={"dataset_root":str(ROOT),"training_table":"reorganized_dataset/film_dynamic_training_table.csv","geometry_authority":"training_table_case_bound_p1_p2","longitudinal_semantics":"SinglePointAxisScan","coordinate_unit":"cm","fixed_test_opened":False,"integrity":integrity,"sensor_audit":sensor_audit,"geometry_counts":rec.geometry_status.value_counts().to_dict()}
(OUT/"dataset_audit_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
lines=["# 数据阶段 0 审计（Protocol 2026-10-09 修正版）","",f"统一数据根目录：`{ROOT}`","", "## 完整性", "", "```json", json.dumps(integrity,ensure_ascii=False,indent=2), "```", "", "## 荷载几何核对", "", "```text", rec[["case_id","load_type","training_p1_x_cm","training_p1_y_cm","training_p2_x_cm","training_p2_y_cm","spatial_p1_x_cm","spatial_p1_y_cm","spatial_p2_x_cm","spatial_p2_y_cm","geometry_status","chosen_source","geometry_semantics"]].to_string(index=False), "```", "", "## 结论", "", "- 当前正式模型输入继续采用 `reorganized_dataset/film_dynamic_training_table.csv` 中与 case_id 逐行绑定的 p1/p2。", "- `LongitudinalLine` 统一解释为 `SinglePointAxisScan`，不构造 line_start/line_end/line_length/line_angle。", "- `spatial_model/load_points_coordinates.csv` 仅作为空间示意和几何审计来源。", "- 所有 mismatch 必须在物理坐标含义确认后再解释；当前不把 mismatch 自动归因于模型错误。", "- fixed-test 仍保持关闭。"]
(OUT/"DATA_AUDIT_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(rec[["case_id","load_type","geometry_status","geometry_semantics"]].to_string(index=False)); print(json.dumps(manifest,ensure_ascii=False,indent=2))
