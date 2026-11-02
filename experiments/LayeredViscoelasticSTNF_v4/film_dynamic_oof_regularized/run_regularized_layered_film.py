from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(r'C:\Users\27356\Documents\Codex\2026-10-07\c-users-27356-documents-codex-2026\outputs\reorganized_dataset_excel_mapping')
OUT=Path(r'D:\matlab\guochuang\fbg-pavement-neural-field-inversion\experiments\LayeredViscoelasticSTNF_v4\film_dynamic_oof_regularized')
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'film_dynamic_training_table.csv'); df=df[df.split_role.eq('development')].copy()
df['p2_x']=df.p2_x.fillna(df.p1_x); df['p2_y']=df.p2_y.fillna(df.p1_y)
df['is_long']=df.direction.eq('Longitudinal').astype(float); df['is_asphalt']=df.layer.eq('Asphalt').astype(float)
df['is_single']=df.load_type.astype(str).str.contains('SingleCenter|SingleEccentric',regex=True).astype(float); df['is_dual']=df.load_type.astype(str).str.contains('Dual',regex=True).astype(float); df['is_line']=df.load_type.astype(str).str.contains('LongitudinalLine',regex=True).astype(float)
cols=['load_kg','p1_x','p1_y','p2_x','p2_y','is_long','is_single','is_dual','is_line','is_asphalt','x_cm','y_cm','z_cm','relative_time_s']
X=df[cols].to_numpy(np.float32); y=df.response_ue.to_numpy(np.float32); q=df.quality_weight.to_numpy(np.float32); folds=df.dev_fold.to_numpy(int); phase=df.relative_time_s.to_numpy(int)
class FiLM(nn.Module):
    def __init__(self,h=64):
        super().__init__(); self.cond=nn.Sequential(nn.Linear(9,h),nn.Tanh(),nn.Dropout(.10),nn.Linear(h,h)); self.base=nn.Linear(4,h); self.g1=nn.Linear(h,h); self.b1=nn.Linear(h,h); self.g2=nn.Linear(h,h); self.b2=nn.Linear(h,h); self.drop=nn.Dropout(.10); self.out=nn.Linear(h,1)
    def forward(self,x):
        c=x[:,:9]; s=x[:,10:]; z=torch.tanh(self.base(s)); h=self.cond(c); z=torch.tanh(self.g1(h)*z+self.b1(h)); z=self.drop(z); z=torch.tanh(self.g2(h)*z+self.b2(h)); return self.out(z).squeeze(1)
class LayeredFiLM(FiLM):
    def __init__(self,h=64):
        super().__init__(h); self.out_asphalt=nn.Linear(h,1); self.out_soil=nn.Linear(h,1)
    def forward(self,x):
        c=x[:,:9]; s=x[:,10:]; z=torch.tanh(self.base(s)); h=self.cond(c); z=torch.tanh(self.g1(h)*z+self.b1(h)); z=self.drop(z); z=torch.tanh(self.g2(h)*z+self.b2(h)); a=self.out_asphalt(z).squeeze(1); b=self.out_soil(z).squeeze(1); return torch.where(x[:,9]>=0,a,b)
metrics=[]
for fold in sorted(np.unique(folds)):
    train=(folds!=fold)&((phase%5)==0); valid=folds==fold; mu=X[train].mean(0); sd=X[train].std(0); sd[sd<1e-6]=1; xt=(X[train]-mu)/sd; xv=(X[valid]-mu)/sd
    model=LayeredFiLM(); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(y[train]),torch.from_numpy(q[train])),batch_size=1024,shuffle=True)
    for _ in range(100):
        for xb,yb,qb in loader:
            loss=(torch.clamp(qb,min=.05)*(model(xb)-yb)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred=model(torch.from_numpy(xv)).numpy()
    truth=y[valid]; layer=df.loc[valid,'layer'].to_numpy(); qw=q[valid]
    for name,mask in [('overall',np.ones(valid.sum(),bool)),('early',phase[valid]<=90),('late',phase[valid]>90)]:
        e=truth[mask]-pred[mask]; metrics.append({'fold':fold,'phase':name,'rmse_ue':float(np.sqrt(np.mean(e*e))),'weighted_rmse_ue':float(np.sqrt(np.sum(qw[mask]*e*e)/np.sum(qw[mask])))})
    for lay in ['Asphalt','Soil']:
        mask=(layer==lay)&(phase[valid]>90); e=truth[mask]-pred[mask]; metrics.append({'fold':fold,'phase':'late_'+lay.lower(),'rmse_ue':float(np.sqrt(np.mean(e*e))),'weighted_rmse_ue':float(np.sqrt(np.sum(qw[mask]*e*e)/np.sum(qw[mask])))})
s=pd.DataFrame(metrics); s.to_csv(OUT/'layered_film_metrics.csv',index=False,encoding='utf-8-sig'); payload={'model':'layered FiLM dynamic microstrain field','epochs':80,'fit_time_stride_s':5,'development_cases':17,'fixed_test_opened':False,'metrics':s.groupby('phase')[['rmse_ue','weighted_rmse_ue']].mean().to_dict('index')}; (OUT/'layered_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(s.groupby('phase')[['rmse_ue','weighted_rmse_ue']].mean().to_string())
