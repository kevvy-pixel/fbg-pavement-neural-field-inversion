from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
sys.path.insert(0,r'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F2_leave_type_out')
import protocol_20261009_corrected_runner as base
sys.path.insert(0,r'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\G_ablation')
from g_ablation_runner import Core,prep,train,DATA
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def main(out,epochs=150):
 out=Path(out); [ (out/x).mkdir(parents=True,exist_ok=True) for x in ['predictions','checkpoints','history'] ]
 raw=pd.read_csv(DATA/'film_dynamic_training_table.csv'); df=raw.copy().reset_index(drop=True); dev=df.split_role.eq('development').to_numpy(); fixed=df.split_role.isin(['fixed-test','fixed_test']).to_numpy(); y=df.response_ue.to_numpy(np.float32); seeds=[20261008,20261009,20261010]; rows=[]
 d,cols=base.add_features(df,'G1'); d[cols]=d[cols].apply(pd.to_numeric,errors='coerce').fillna(0); fold=sorted(df.loc[dev,'dev_fold'].dropna().unique())[0]; val=dev&(df.dev_fold==fold).to_numpy(); fit=dev&~val&(df.relative_time_s.mod(5).eq(0).to_numpy()); xg,normg=prep(df,fit)
 for model in ['A0-3','B0','G2']:
  for seed in ([20261008] if model=='A0-3' else seeds):
   if model=='G2':
    torch.manual_seed(seed); np.random.seed(seed); m=Core(True,False); m,h,b=train(m,xg,y,fit,val,seed,epochs); m.eval();
    with torch.no_grad(): pred=m(torch.from_numpy(xg[fixed])).numpy()
    h.to_csv(out/'history'/f'{model}_{seed}.csv',index=False); torch.save({'state_dict':m.state_dict(),'normalizer':normg,'best_epoch':b},out/'checkpoints'/f'{model}_{seed}.pt')
   elif model=='A0-3':
    pred=base.ridge_predict(d.loc[fit,cols].to_numpy(np.float32),y[fit],d.loc[fixed,cols].to_numpy(np.float32)); b=0
   else:
    xtr=d.loc[fit,cols].to_numpy(np.float32); xv=d.loc[val,cols].to_numpy(np.float32); xt=d.loc[fixed,cols].to_numpy(np.float32); mu=xtr.mean(0); sd=xtr.std(0); sd[sd<1e-6]=1; m,h,b=base.train_nn(model,(xtr-mu)/sd,y[fit],np.ones(int(fit.sum())),(xv-mu)/sd,y[val],seed,epochs); h.to_csv(out/'history'/f'{model}_{seed}.csv',index=False); torch.save({'model':model,'state_dict':m.state_dict(),'input_mean':mu,'input_std':sd,'best_epoch':b},out/'checkpoints'/f'{model}_{seed}.pt'); pred=m(torch.from_numpy(((xt-mu)/sd).astype(np.float32))).detach().numpy()
   q=df[fixed].copy(); q['actual_ue']=y[fixed]; q['predicted_ue']=pred; q['model']=model; q['seed']=seed; q.to_csv(out/'predictions'/f'{model}_{seed}.csv',index=False,encoding='utf-8-sig'); e=y[fixed]-pred; t=q.relative_time_s.to_numpy(); lay=q.layer.to_numpy(); ss=q.sensor_string.to_numpy()
   for ph,mm in [('overall',np.ones(len(e),bool)),('early',t<=90),('late',t>90),('late_asphalt',(t>90)&(lay=='Asphalt')),('late_soil',(t>90)&(lay=='Soil'))]+[(s,ss==s) for s in ['SAL1','SAL2','SAT1','SAT2','SSL1','SSL2','SST']]:
    if mm.any(): rows.append({'model':model,'seed':seed,'phase':ph,'rmse':float(np.sqrt(np.mean(e[mm]**2))),'n':int(mm.sum()),'best_epoch':b})
 pd.DataFrame(rows).to_csv(out/'metrics_by_file.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(rows).groupby(['model','phase']).rmse.agg(['mean','std','count']).reset_index().to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig'); manifest={'stage':'F7_fixed_test','training_role':'development','evaluation_role':'fixed-test','models':['A0-3','B0','G2'],'seeds':seeds,'epochs':epochs,'fixed_cases':sorted(df.loc[fixed,'case_id'].unique().tolist()),'fixed_test_opened_once':True,'no_model_selection_after_fixed_test':True,'data_sha256':sha(DATA/'film_dynamic_training_table.csv')}; (out/'FIXED_TEST_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'FIXED_TEST_REPORT.md').write_text('# F7 fixed-test\n\n固定测试只执行一次，结果不用于调参。\n',encoding='utf-8')
if __name__=='__main__':
 import argparse; p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=150); a=p.parse_args(); main(a.out,a.epochs)
