import os,json,argparse
import numpy as np,pandas as pd
TABLE=r'C:\Users\27356\Documents\Codex\2026-10-07\c-users-27356-documents-codex-2026\outputs\reorganized_dataset_excel_mapping\film_dynamic_training_table.csv'
ROOT=os.path.join(os.path.dirname(__file__),'A_B_models','B0','seed_20261008')
STATIC=['load_kg','p1_x','p1_y','p2_x','p2_y','sensor_x','sensor_y','sensor_z','is_asphalt','is_single','is_dual_symmetric','is_dual_asymmetric','is_longitudinal_line','sensor_direction']
def add_flags(d):
    t=d.load_type.fillna('').astype(str); d['is_single']=t.str.startswith('Single').astype(float); d['is_dual_symmetric']=t.eq('DualSymmetric').astype(float); d['is_dual_asymmetric']=t.eq('DualAsymmetric').astype(float); d['is_longitudinal_line']=t.eq('LongitudinalLine').astype(float); d['sensor_direction']=d.direction.map({'Longitudinal':0.,'Transverse':1.}).fillna(0.); d['is_asphalt']=(d.layer=='Asphalt').astype(float); d['sensor_x']=d.x_cm; d['sensor_y']=d.y_cm; d['sensor_z']=d.z_cm; return d
def feat(d,means,std,taus,drift):
    x=d[STATIC].copy(); cont=['load_kg','p1_x','p1_y','p2_x','p2_y','sensor_x','sensor_y','sensor_z']; x[cont]=(x[cont]-means[cont])/std[cont]; x=x.fillna(0.).values; t=d.relative_time_s.values.astype(float); bs=[np.ones(len(t))]+[np.exp(-t/z) for z in taus]
    if drift: bs.append(t/280.)
    B=np.stack(bs,1); return np.concatenate([x[:,j:j+1]*B for j in range(x.shape[1])],1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ridge',type=float,default=1e-3); args=ap.parse_args(); os.makedirs(ROOT,exist_ok=True)
    d=add_flags(pd.read_csv(TABLE)); d=d[(d.relative_time_s<=280)&np.isfinite(d.response_ue)].copy(); d[STATIC]=d[STATIC].replace([np.inf,-np.inf],np.nan).fillna(0.); d['dev_fold']=d.dev_fold.fillna(0).astype(int); dev=d[d.dev_fold>0].copy()
    configs=[('A0-1',[40.],False,0.),('A0-2',[40.,160.],False,0.),('A0-3',[40.,160.],True,0.),('A0-4',[40.,160.],True,1e-3),('A0-5',[40.,160.],True,1e-3)]
    for name,taus,drift,graph in configs:
        out=ROOT; preds=[]
        for fold in sorted(dev.dev_fold.unique()):
            tr=dev[(dev.dev_fold!=fold)&(dev.relative_time_s%5==0)].copy(); ev=dev[dev.dev_fold==fold].copy(); cont=['load_kg','p1_x','p1_y','p2_x','p2_y','sensor_x','sensor_y','sensor_z']; means=tr[cont].mean(); std=tr[cont].std().replace(0,1).fillna(1.)
            X=feat(tr,means,std,taus,drift); y=tr.response_ue.values; A=X.T@X+args.ridge*np.eye(X.shape[1]); b=X.T@y
            if graph>0:
                groups=tr.groupby(['case_id','relative_time_s','sensor_string'],sort=False)
                for (_,_,_),g in groups:
                    g=g.sort_values('sensor_index'); rows=g.index.tolist()
                    for i in range(len(rows)-1):
                        r1,r2=rows[i],rows[i+1]
                        if tr.loc[r1,'layer']!=tr.loc[r2,'layer']: continue
                        if name=='A0-5' and tr.loc[r1,'layer']!='Soil': continue
                        x1=feat(tr.loc[[r1]],means,std,taus,drift)[0]; x2=feat(tr.loc[[r2]],means,std,taus,drift)[0]; xd=x1-x2; A += graph*np.outer(xd,xd)
            w=np.linalg.solve(A,b); yp=feat(ev,means,std,taus,drift)@w; z=ev[['case_id','sensor_id','layer','relative_time_s','quality_weight','response_ue']].copy(); z['predicted_ue']=yp; z['fold']=fold; z=z.rename(columns={'response_ue':'actual_ue'}); z.to_csv(os.path.join(out,f'{name}_fold_{fold}_predictions.csv'),index=False); preds.append(z)
        allp=pd.concat(preds,ignore_index=True); allp.to_csv(os.path.join(out,f'{name}_oof_predictions.csv'),index=False); json.dump({'model':name,'taus_s':taus,'linear_drift':drift,'graph_penalty':graph,'ridge':args.ridge,'fixed_test_opened':False},open(os.path.join(out,f'{name}_config.json'),'w'),indent=2); print(name,'done',len(allp))
if __name__=='__main__': main()
