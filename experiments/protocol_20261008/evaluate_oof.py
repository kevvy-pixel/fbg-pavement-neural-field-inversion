import argparse, json, glob, os
import numpy as np, pandas as pd

def rmse(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2))) if len(a) else float('nan')
def wrmse(a,b,w):
    a=np.asarray(a,float); b=np.asarray(b,float); w=np.asarray(w,float)
    return float(np.sqrt(np.sum(w*(a-b)**2)/np.sum(w))) if len(a) and np.sum(w)>0 else float('nan')
def metric_frame(df):
    out={
      'n': int(len(df)),
      'overall_rmse': rmse(df.actual_ue,df.predicted_ue),
      'raw_weighted_rmse': wrmse(df.actual_ue,df.predicted_ue,df.quality_weight),
      'clipped_weighted_rmse': wrmse(df.actual_ue,df.predicted_ue,np.maximum(df.quality_weight,.05)),
      'exclude_d2_ch1_rmse': rmse(df.loc[df.sensor_id!='D2-CH1','actual_ue'],df.loc[df.sensor_id!='D2-CH1','predicted_ue'])
    }
    return out
def add_stats(rows, key):
    vals=pd.Series([r[key] for r in rows],dtype=float)
    return {'mean':float(vals.mean()),'std':float(vals.std(ddof=1)) if len(vals)>1 else 0.0,'min':float(vals.min()),'max':float(vals.max())}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(args.predictions,'fold_*_predictions.csv')))
    if not files and os.path.isfile(args.predictions): files=[args.predictions]
    if not files: raise FileNotFoundError(args.predictions)
    parts=[]
    for f in files:
        d=pd.read_csv(f)
        if 'response_ue' in d and 'actual_ue' not in d: d=d.rename(columns={'response_ue':'actual_ue'})
        if 'case_id' not in d: d['case_id']='unknown'
        if 'sensor_id' not in d: d['sensor_id']='unknown'
        if 'relative_time_s' not in d and 'time_s' in d: d['relative_time_s']=d['time_s']
        if 'quality_weight' not in d: d['quality_weight']=1.0
        if 'fold' not in d: d['fold']=os.path.basename(f).split('_')[1]
        d['phase']=np.where(d.relative_time_s<=90,'early','late')
        parts.append(d)
    allp=pd.concat(parts,ignore_index=True)
    os.makedirs(args.output,exist_ok=True)
    allp.to_csv(os.path.join(args.output,'oof_predictions.csv'),index=False)
    fold_rows=[]
    for fold,g in allp.groupby('fold',sort=True):
        r={'fold':fold,**metric_frame(g)}; fold_rows.append(r)
        for ph,gg in g.groupby('phase'):
            fold_rows.append({'fold':fold,'phase':ph,**metric_frame(gg)})
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.output,'fold_metrics.csv'),index=False)
    case_rows=[]
    for (case,ph),g in allp.groupby(['case_id','phase'],sort=True): case_rows.append({'case_id':case,'phase':ph,**metric_frame(g)})
    pd.DataFrame(case_rows).to_csv(os.path.join(args.output,'case_metrics.csv'),index=False)
    sensor_rows=[]
    for (sensor,ph),g in allp.groupby(['sensor_id','phase'],sort=True): sensor_rows.append({'sensor_id':sensor,'phase':ph,**metric_frame(g)})
    pd.DataFrame(sensor_rows).to_csv(os.path.join(args.output,'sensor_metrics.csv'),index=False)
    summary={'pooled':metric_frame(allp),'phases':{ph:metric_frame(g) for ph,g in allp.groupby('phase')},'fold_macro':{}}
    for ph in ['overall','early','late']:
        gg=allp if ph=='overall' else allp[allp.phase==ph]
        rows=[metric_frame(g) for _,g in gg.groupby('fold')]
        summary['fold_macro'][ph]={k:add_stats(rows,k) for k in ['overall_rmse','raw_weighted_rmse','clipped_weighted_rmse','exclude_d2_ch1_rmse']}
    with open(os.path.join(args.output,'pooled_summary.json'),'w',encoding='utf-8') as f: json.dump(summary,f,indent=2)
if __name__=='__main__': main()
