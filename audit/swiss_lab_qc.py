import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='2'
import pandas as pd, numpy as np
from soil_stage3_reassessment import prep, predict, summarize
d='agro_data/'
n=pd.read_csv(d+'03_spectraA_neo.csv'); m=pd.read_csv(d+'02_spectraA_psr.csv'); l=pd.read_csv(d+'01_laboratory_analyses.csv')
ids={v:set(z.km_id) for v,z in n.groupby('scantype')}
print('field counts', {v:len(z) for v,z in ids.items()}, 'a∩b',len(ids['a']&ids['b']), 'a,b diffs',sorted(ids['a']^ids['b']),flush=True)
for tag,valid in [('all',ids['l']),('a_proxy',ids['a']),('b_proxy',ids['b'])]:
 keys=np.array(sorted(map(str,valid)))
 N=n[n.scantype=='l'].set_index(n[n.scantype=='l'].km_id.astype(str)); M=m[m.scantype=='l'].set_index(m[m.scantype=='l'].km_id.astype(str))
 y=l.set_index(l.km_id.astype(str)).loc[keys,'OC_pc'].to_numpy()
 g=M.loc[keys,'field'].to_numpy()
 x=prep(N.loc[keys,N.columns.str.isdigit()]);z=prep(M.loc[keys,M.columns.str.isdigit()])
 rows=[]
 for f in sorted(set(g)):
  tr=np.flatnonzero(g!=f);te=np.flatnonzero(g==f)
  p,k=predict(x,y,g,tr,te);q,j=predict(z,y,g,tr,te)
  rows.append(pd.DataFrame(dict(fold=f,key=keys[te],group=g[te],y=y[te],nir=p,aug=q)))
 D=pd.concat(rows,ignore_index=True)
 print(tag,summarize(D),flush=True)
 D.to_csv(f'swiss_lab_{tag}.csv',index=False)
