import pandas as pd, numpy as np
from soil_stage3_reassessment import policy,ci
for tag in ('all','coords','all_country'):
 d=pd.read_csv(f'icraf_pred_{tag}.csv')
 fused=d.assign(aug=(d.nir+d.aug)/2)
 rows=[]
 for b in (.2,.4,.5,.6,1):
  a=policy(d,b); f=policy(fused,b)
  rows.append(dict(cohort=tag,budget=b,n=len(d),selected=a.selected.sum(),only_first=a.base.mean(),only_MIR=a.full.mean(),all_pair_equal_fusion=f.full.mean(),selected_then_replace=a.high.mean(),random_then_replace=a.random.mean(),selected_then_fuse=f.high.mean(),random_then_fuse=f.random.mean()))
 print(pd.DataFrame(rows).to_string(index=False))
 pd.DataFrame(rows).to_csv(f'icraf_fixed_fusion_{tag}.csv',index=False)
