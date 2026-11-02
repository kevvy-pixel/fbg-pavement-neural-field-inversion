from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
sys.path.insert(0,r'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\F2_leave_type_out')
import protocol_20261009_corrected_runner as base
sys.path.insert(0,r'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\protocol_20261009_corrected\G_ablation')
from g_ablation_runner import Core,prep,train,DATA

COMB=[('DualAsymmetric','SAL2'),('DualAsymmetric','SAT2'),('DualAsymmetric','SSL1'),('DualSymmetric','SAL2'),('DualSymmetric','SAT2'),('SingleEccentric','SAL2'),('LongitudinalLine','SAL2'),('SingleCenter','SAL2')]
SEEDS=[20261008,20261009,20261010]
def main(out,epochs=150):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); raw=pd.read_csv(DATA/'film_dynamic_training_table.csv'); df=raw[raw.split_role.eq('development')].copy().reset_index(drop=True); dfb,cols=base.add_features(df,'G1'); dfb[cols]=dfb[cols].apply(pd.to_numeric,errors='coerce').fillna(0); y=df.response_ue.to_numpy(np.float32); rows=[]
 for held,ss in COMB:
  task=f'{held}__{ss}'; td=out/task; (td/'predictions').mkdir(parents=True,exist_ok=True); (td/'checkpoints').mkdir(exist_ok=True); (td/'history').mkdir(exist_ok=True)
  trainmask=((df.load_type!=held)&(df.sensor_string!=ss)).to_numpy(); testmask=((df.load_type==held)&(df.sensor_string==ss)).to_numpy(); folds=sorted(df.loc[trainmask,'dev_fold'].unique()); valmask=trainmask&(df.dev_fold==folds[0]).to_numpy(); fitmask=trainmask&~valmask&(df.relative_time_s.mod(5).eq(0).to_numpy())
  xg,normg=prep(df,fitmask)
  for model in ['A0-3','B0','G2']:
   seeds=[SEEDS[0]] if model=='A0-3' else SEEDS
   for seed in seeds:
    pf=td/'predictions'/f'{model}_{seed}.csv'
    if pf.exists(): continue
    if model=='G2':
     torch.manual_seed(seed); np.random.seed(seed); m=Core(True,False); m,h,b=train(m,xg,y,fitmask,valmask,seed,epochs); m.eval();
     with torch.no_grad(): pred=m(torch.from_numpy(xg[testmask])).numpy()
     h.to_csv(td/'history'/f'{model}_{seed}.csv',index=False); torch.save({'state_dict':m.state_dict(),'normalizer':normg,'best_epoch':b},td/'checkpoints'/f'{model}_{seed}.pt')
    else:
     xtr=dfb.loc[fitmask,cols].to_numpy(np.float32); xva=dfb.loc[valmask,cols].to_numpy(np.float32); xte=dfb.loc[testmask,cols].to_numpy(np.float32); ytr=y[fitmask]; yva=y[valmask]
     if model=='A0-3': pred=base.ridge_predict(xtr,ytr,xte); b=0
     else:
      mu=xtr.mean(0); sd=xtr.std(0); sd[sd<1e-6]=1; mm,h,b=base.train_nn(model,(xtr-mu)/sd,ytr,np.ones(len(xtr)),(xva-mu)/sd,yva,seed,epochs); h.to_csv(td/'history'/f'{model}_{seed}.csv',index=False); torch.save({'model':model,'state_dict':mm.state_dict(),'input_mean':mu,'input_std':sd,'best_epoch':b},td/'checkpoints'/f'{model}_{seed}.pt'); pred=mm(torch.from_numpy(((xte-mu)/sd).astype(np.float32))).detach().numpy()
    q=df[testmask].copy(); q['actual_ue']=y[testmask]; q['predicted_ue']=pred; q['model']=model; q['seed']=seed; q['held_type']=held; q['held_sensor_string']=ss; q.to_csv(pf,index=False,encoding='utf-8-sig'); e=y[testmask]-pred; t=q.relative_time_s.to_numpy(); lay=q.layer.to_numpy()
    for ph,mask in [('overall',np.ones(len(e),bool)),('early',t<=90),('late',t>90),('late_asphalt',(t>90)&(lay=='Asphalt')),('late_soil',(t>90)&(lay=='Soil'))]:
     if mask.any(): rows.append({'task':task,'held_type':held,'held_sensor_string':ss,'model':model,'seed':seed,'phase':ph,'rmse':float(np.sqrt(np.mean(e[mask]**2))),'n':int(mask.sum()),'best_epoch':b})
 pd.DataFrame(rows).to_csv(out/'metrics_file_level.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(rows).groupby(['task','model','phase']).rmse.agg(['mean','std','count']).reset_index().to_csv(out/'seed_summary.csv',index=False,encoding='utf-8-sig'); (out/'RUN_MANIFEST.json').write_text(json.dumps({'stage':'F4','combinations':COMB,'models':['A0-3','B0','G2'],'seeds':SEEDS,'epochs':epochs,'fixed_test_opened':False},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
 import argparse; p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=150); a=p.parse_args(); main(a.out,a.epochs)
