"""ICRAF v1.3 validation, analytical exact-budget random baseline."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[v]='2'
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from soil_stage3_reassessment import prep,predict,summarize
import argparse
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--same-coord-cohort',action='store_true')
 p.add_argument('--country',action='store_true')
 a=p.parse_args()
 root=Path('icraf_data')
 stem='coords' if a.same_coord_cohort else 'all'
 co='id.layer_local_c'
 C='loc.country_src_txt'
 SOC='oc_usda.c1059_w.pct'
 U='id.layer_uuid_txt'
 prof='profile_plotcode'
 ll=['longitude.point_wgs84_dd','latitude.point_wgs84_dd']
 site=pd.read_parquet(root/'ossl_soilsite_v1.3.parquet')
 lab=pd.read_parquet(root/'ossl_soillab_v1.3.parquet')
 src={k:pd.read_parquet(root/f'ossl_{k}_v1.3.parquet') for k in ['visnir','mir']}
 cw=pd.read_csv('ICRAF_ISRIC_v1.3_profile_crosswalk.tsv',sep='\t')
 for k,d in dict(site=site,lab=lab,**src).items():
  print(k,len(d),d[co].nunique(),d[co].isna().sum(),d[co].duplicated().sum(),flush=True)
  assert d[co].is_unique
 df=site[[co,U,C,*ll]].merge(cw,on=[co,U],how='left',validate='one_to_one').merge(lab[[co,SOC]],on=co,validate='one_to_one')
 assert df[prof].notna().all() and df.groupby(prof)[C].nunique().max()==1
 df=df.merge(src['visnir'],on=co,validate='one_to_one').merge(src['mir'],on=co,validate='one_to_one')
 N=[c for c in df.columns if c.startswith('scan_visnir.') and 1350<=int(c.split('.')[1].split('_')[0])<=2500]
 M=[c for c in df.columns if c.startswith('scan_mir.')]
 good=np.isfinite(df[[SOC,*N,*M]].to_numpy(dtype=float)).all(1)&(df[SOC]>=0)&df[C].notna()
 df=df.loc[good].copy()
 print('finite paired cohort',len(df),df[prof].nunique(),df[C].nunique(),'missing coordinate',df[ll].isna().any(axis=1).sum(),flush=True)
 if a.same_coord_cohort:df=df.dropna(subset=ll)
 df=df.sort_values(U).reset_index(drop=True)
 x,z=prep(df[N].to_numpy()),prep(df[M].to_numpy())
 y,g,keys,country=df[SOC].to_numpy(),df[prof].to_numpy(),df[U].to_numpy(),df[C].to_numpy()
 print('analysis cohort',len(df),len(set(g)),len(set(country)),flush=True)
 splits=list((str(i+1),tr,te) for i,(tr,te) in enumerate(GroupKFold(5).split(x,groups=g)))
 if a.country:
  stem+='_country'
  splits=[(c,np.flatnonzero(country!=c),np.flatnonzero(country==c)) for c,n in df[C].value_counts().items() if n>=30]
 rows=[]
 for f,tr,te in splits:
  assert not set(g[tr])&set(g[te])
  p,k=predict(x,y,g,tr,te);q,l=predict(z,y,g,tr,te)
  print(f,len(tr),len(te),k,l,flush=True)
  rows.append(pd.DataFrame(dict(fold=f,key=keys[te],group=g[te],country=country[te],y=y[te],nir=p,aug=q,k_nir=k,k_aug=l,n_train=len(tr))))
  pd.concat(rows,ignore_index=True).to_csv(f'icraf_pred_{stem}.csv',index=False)
 d=pd.concat(rows,ignore_index=True); assert d.key.is_unique
 print(stem,summarize(d),flush=True)
 pd.DataFrame([summarize(d,frac=b) for b in [0,.1,.2,.3,.4,.5,.6,1]]).to_csv(f'icraf_budget_{stem}.csv',index=False)
 if a.country:
  pd.DataFrame([dict(country=c,**summarize(r.reset_index(drop=True))) for c,r in d.groupby('country')]).to_csv(f'icraf_by_country_{stem}.csv',index=False)
if __name__=='__main__':main()
