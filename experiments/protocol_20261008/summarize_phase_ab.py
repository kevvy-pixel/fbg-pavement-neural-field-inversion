import os,json,pandas as pd,numpy as np
ROOT=os.path.join(os.path.dirname(__file__),'A_B_models')
def rm(a,b): return float(np.sqrt(np.mean((a-b)**2)))
rows=[]
for m in sorted(os.listdir(ROOT)):
  p=os.path.join(ROOT,m,'seed_20261008','oof_predictions.csv')
  if not os.path.isfile(p): continue
  d=pd.read_csv(p); d['phase']=np.where(d.relative_time_s<=90,'early','late')
  one=lambda g: rm(g.actual_ue,g.predicted_ue)
  r={'model':m,'seed':20261008,'pooled_overall_rmse':one(d),'pooled_early_rmse':one(d[d.phase=='early']),'pooled_late_rmse':one(d[d.phase=='late']),'pooled_late_asphalt_rmse':one(d[(d.phase=='late')&(d.layer=='Asphalt')]),'pooled_late_soil_rmse':one(d[(d.phase=='late')&(d.layer=='Soil')])}
  s=json.load(open(os.path.join(ROOT,m,'seed_20261008','pooled_summary.json')))
  r['pooled_raw_weighted_rmse']=s['pooled']['raw_weighted_rmse']; r['pooled_clipped_weighted_rmse']=s['pooled']['clipped_weighted_rmse']; r['fold_macro_mean_rmse']=s['fold_macro']['overall']['overall_rmse']['mean']; r['fold_macro_std_rmse']=s['fold_macro']['overall']['overall_rmse']['std']; r['fold_min_rmse']=s['fold_macro']['overall']['overall_rmse']['min']; r['fold_max_rmse']=s['fold_macro']['overall']['overall_rmse']['max']; rows.append(r)
out=os.path.join(os.path.dirname(__file__),'phase_ab_summary.csv'); pd.DataFrame(rows).to_csv(out,index=False)
print(pd.DataFrame(rows).to_string(index=False))
