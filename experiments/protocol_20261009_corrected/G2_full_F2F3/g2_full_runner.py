from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,torch
from g_ablation_runner import Core,prep,DATA,train

def main(a):
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA/'film_dynamic_training_table.csv'); df=df[df.split_role.eq('development')].copy().reset_index(drop=True); y=df.response_ue.to_numpy(np.float32); seeds=[20261008,20261009,20261010]; rows=[]
 tasks=[]
 for held in ['DualAsymmetric','DualSymmetric','LongitudinalLine','SingleCenter','SingleEccentric']:
  tasks.append(('F2_'+held,'F2',held,(df.load_type!=held).to_numpy(),(df.load_type==held).to_numpy()))
 for s in ['SAL1','SAL2','SAT1','SAT2','SSL1','SSL2','SST']:
  tasks.append(('F3_'+s,'F3',s,(df.sensor_string!=s).to_numpy(),(df.sensor_string==s).to_numpy()))
 for task,stage,label,trainmask,testmask in tasks:
  td=out/stage/task; (td/'predictions').mkdir(parents=True,exist_ok=True); (td/'history').mkdir(exist_ok=True); (td/'checkpoints').mkdir(exist_ok=True)
  folds=sorted(df.loc[trainmask,'dev_fold'].unique()); valmask=trainmask&(df.dev_fold==folds[0]).to_numpy(); fitmask=trainmask&~valmask&(df.relative_time_s.mod(5).eq(0).to_numpy()); x,norm=prep(df,fitmask)
  for seed in seeds:
   predfile=td/'predictions'/f'{label}_{seed}.csv'
   if predfile.exists() and predfile.stat().st_size>1000:
    continue
   torch.manual_seed(seed); np.random.seed(seed); m=Core(True,False); m,h,b=train(m,x,y,fitmask,valmask,seed,a.epochs); m.eval()
   with torch.no_grad(): pred=m(torch.from_numpy(x[testmask])).numpy()
   q=df[testmask].copy(); q['actual_ue']=y[testmask]; q['predicted_ue']=pred; q['model']='G2'; q['seed']=seed; q['stage']=stage; q['task']=task; q.to_csv(predfile,index=False,encoding='utf-8-sig'); h.to_csv(td/'history'/f'{label}_{seed}.csv',index=False); torch.save({'state_dict':m.state_dict(),'normalizer':norm,'best_epoch':b},td/'checkpoints'/f'{label}_{seed}.pt')
   e=y[testmask]-pred; t=df.loc[testmask,'relative_time_s'].to_numpy(); lay=df.loc[testmask,'layer'].to_numpy(); ss=df.loc[testmask,'sensor_string'].to_numpy()
   for ph,mm in [('overall',np.ones(len(e),bool)),('early',t<=90),('late',t>90),('late_asphalt',(t>90)&(lay=='Asphalt')),('late_soil',(t>90)&(lay=='Soil')),('SAL2',ss=='SAL2'),('SAT2',ss=='SAT2'),('SAL1',ss=='SAL1'),('SAT1',ss=='SAT1'),('SSL1',ss=='SSL1'),('SSL2',ss=='SSL2'),('SST',ss=='SST')]:
    if mm.any(): rows.append({'stage':stage,'task':task,'label':label,'seed':seed,'phase':ph,'rmse':float(np.sqrt(np.mean(e[mm]**2))),'n':int(mm.sum()),'best_epoch':b})
 pd.DataFrame(rows).to_csv(out/'metrics_file_level.csv',index=False,encoding='utf-8-sig'); s=pd.DataFrame(rows).groupby(['stage','task','phase']).rmse.agg(['mean','std','count']).reset_index(); s.to_csv(out/'seed_summary.csv',index=False,encoding='utf-8-sig'); (out/'RUN_MANIFEST.json').write_text(json.dumps({'model':'G2','stages':['F2','F3'],'seeds':seeds,'epochs':a.epochs,'min_epochs':50,'patience':20,'fixed_test_opened':False},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--epochs',type=int,default=150); main(p.parse_args())
