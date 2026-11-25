"""Reassessment of selective spectroscopy. Python 3, NumPy/pandas/SciPy/sklearn.
Run: python soil_stage3_reassessment.py --root <original project> --out <new directory>
Original files are read-only. No result-dependent policy tuning.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[v]='2'
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

BUDGETS = [0,.1,.2,.3,.4,.5,1]
def prep(x):
    x=np.asarray(x,dtype=float)
    return savgol_filter((x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-12),11,2,deriv=1,axis=1)

def predict(x,y,g,tr,te):
    # All components and hyperparameters chosen in training only.
    ng=len(np.unique(g[tr])); splits=list(GroupKFold(min(3,ng)).split(x[tr],y[tr],g[tr])) if ng>=2 else []
    cap=min(20,x.shape[1],min((len(a)-1 for a,b in splits),default=len(tr)-1))
    grid=[k for k in [2,4,8,12,16,20] if k<=cap] or [1]
    err=np.zeros(len(grid))
    for a,b in splits:
        u,v=tr[a],tr[b]
        for j,k in enumerate(grid):
            m=PLSRegression(n_components=k).fit(x[u],y[u])
            err[j]+=np.mean((m.predict(x[v]).ravel()-y[v])**2)
    k=grid[int(np.argmin(err))] if splits else min(4,cap)
    m=PLSRegression(n_components=k).fit(x[tr],y[tr])
    return np.maximum(0,m.predict(x[te]).ravel()),k

def make_splits(g,domain=None):
    if domain is None:
        return [(str(i+1),a,b) for i,(a,b) in enumerate(GroupKFold(5).split(g,groups=g))]
    result=[]
    for d in sorted(set(domain)):
        te=np.flatnonzero(domain==d)
        tr=np.flatnonzero((domain!=d)&~np.isin(g,g[te]))
        result.append((str(d),tr,te))
    return result

def run(name,x,z,y,g,keys,splits):
    rows=[]
    for f,tr,te in splits:
        assert not set(g[tr])&set(g[te])
        p,k=predict(x,y,g,tr,te); q,l=predict(z,y,g,tr,te)
        rows.append(pd.DataFrame(dict(scenario=name,fold=f,key=keys[te],group=g[te],y=y[te],nir=p,aug=q,k_nir=k,k_aug=l,n_train=len(tr))))
        print(name,f,len(tr),len(te),k,l,flush=True)
    d=pd.concat(rows,ignore_index=True)
    assert d.key.is_unique and len(d)==len(y)
    return d

def policy(d,frac,log=False):
    y,p,q=d.y.to_numpy(),d.nir.to_numpy(),d.aug.to_numpy()
    if log: y,p,q=map(np.log1p,(y,p,q))
    en,em=np.abs(y-p),np.abs(y-q); g=en-em
    high=np.zeros(len(d),bool); oracle=high.copy(); random=np.zeros(len(d))
    for _,ix in d.groupby('fold').indices.items():
        k=int(np.floor(len(ix)*frac+1e-9))
        h=ix[np.lexsort((d.key.to_numpy()[ix],-d.nir.to_numpy()[ix]))[:k]]
        o=ix[np.lexsort((d.key.to_numpy()[ix],-g[ix]))[:k]]
        high[h]=True;oracle[o]=True;random[ix]=k/len(ix)
    return pd.DataFrame(dict(base=en,full=em,high=en-high*g,random=en-random*g,oracle=en-oracle*g,selected=high,gain=g),index=d.index)

def ci(x,groups):
    a=pd.DataFrame({'x':x,'group':groups}).groupby('group').x.agg(['sum','count']).to_numpy()
    if len(a)<2:
        return [float('nan'),float('nan')]
    rng=np.random.default_rng(7301); out=[]
    for _ in range(3000):
        b=a[rng.integers(len(a),size=len(a))];out.append(b[:,0].sum()/b[:,1].sum())
    return np.quantile(out,[.025,.975]).tolist()

def summarize(d,frac=.2,log=False):
    p=policy(d,frac,log); v=p[['base','full','high','random','oracle']].mean().to_dict()
    v.update(n=len(d),groups=d.group.nunique(),selected=int(p.selected.sum()),budget=frac,loss='log1p' if log else 'absolute',high_minus_random=v['high']-v['random'],improvement_vs_base=1-v['high']/v['base'],improvement_vs_random=1-v['high']/v['random'])
    v['conditional_ci_low'],v['conditional_ci_high']=ci(p.high-p.random,d.group)
    for k in ['nir','aug']:
        e=d[k]-d.y; v[k+'_r2']=1-float((e**2).sum())/float(((d.y-d.y.mean())**2).sum());v[k+'_rmse']=float(np.sqrt((e**2).mean()))
    return v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    s=np.load(a.root/'stage2/results/single_device_data.npz',allow_pickle=False)
    y,g,keys=s['y'].astype(float),s['pedon'],s['keys'];x,z=prep(s['nir']),prep(s['mir'])
    assert len(set(keys))==len(keys) and np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.all(y>=0)
    outputs=[]
    for name,domain in [('A_pedon',None),('A_device',s['scanner']),('A_lab',s['lab'])]:
        outputs.append(run(name,x,z,y,g,keys,make_splits(g,domain)))
    # Different instrument pairing: replication of the allocation concept, not same modality external transfer.
    spec=importlib.util.spec_from_file_location('source_b',a.root/'scripts/02b_datasetB.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    ad=a.root/'data/agroscope';lab=pd.read_csv(ad/'01_laboratory_analyses.csv',encoding='utf-8-sig');lab=lab.set_index('km_id').OC_pc
    psr,_=b.load_scan(ad/'02_spectraA_psr.csv','l');neo,_=b.load_scan(ad/'03_spectraA_neo.csv','l')
    fld=pd.read_csv(ad/'02_spectraA_psr.csv',usecols=['km_id','field','scantype']);fld=fld[fld.scantype=='l'].drop_duplicates(['km_id','field']);assert fld.km_id.is_unique
    fld=fld.set_index('km_id').field;keys=np.array(sorted(set(lab.index.astype(str))&set(psr)&set(neo)&set(fld.index.astype(str))))
    lab.index=lab.index.astype(str);fld.index=fld.index.astype(str);y=lab.loc[keys].to_numpy(float);g=fld.loc[keys].astype(str).to_numpy();x=prep(np.array([neo[k] for k in keys]));z=prep(np.array([psr[k] for k in keys]));print('B fields',pd.Series(g).value_counts().to_dict(),flush=True)
    outputs.append(run('B_field',x,z,y,g,keys,make_splits(g,g)))
    d=pd.concat(outputs,ignore_index=True);d.to_csv(a.out/'predictions.csv',index=False)
    summary=[];details=[];curves=[];heter=[];diag=[]
    for scenario,sub in d.groupby('scenario',sort=False):
        print('summarizing',scenario,flush=True)
        sub=sub.reset_index(drop=True)
        for log in [False,True]:summary.append(dict(scenario=scenario,**summarize(sub,log=log)))
        for f,t in sub.groupby('fold'):
            details.append(dict(scenario=scenario,fold=f,**summarize(t.reset_index(drop=True))))
        for frac in BUDGETS:
            p=policy(sub,frac);curves.append(dict(scenario=scenario,budget=frac,n=len(sub),selected=int(p.selected.sum()),**p[['base','full','high','random','oracle']].mean().to_dict()))
        p=policy(sub,.2);sub=pd.concat([sub,p],axis=1)
        for variable in ['nir','y']:
            sub['bin']=pd.qcut(sub[variable],5,duplicates='drop')
            for k,t in sub.groupby('bin',observed=True):
                heter.append(dict(scenario=scenario,stratification=variable,bin=str(k),n=len(t),soc_mean=t.y.mean(),base=t.base.mean(),aug=t.full.mean(),gain=t.gain.mean(),positive=(t.gain>0).mean(),selected=t.selected.mean(),high=t.high.mean(),random=t.random.mean()))
        gain=sub.gain.to_numpy();top=np.argsort(-gain)[:max(1,len(gain)//20)];pos=np.maximum(gain,0)
        cutoff=sub.y.quantile(.95);keep=sub.y<=cutoff
        diag.append(dict(scenario=scenario,positive_gain_share_top5=pos[top].sum()/pos.sum(),signed_gain_share_top5=gain[top].sum()/gain.sum() if gain.sum()!=0 else np.nan,positive_gain_fraction=(gain>0).mean(),rho_pred_gain=spearmanr(sub.nir,gain).statistic,rho_true_absN=spearmanr(sub.y,sub.base).statistic,rho_pred_absN=spearmanr(sub.nir,sub.base).statistic,high_minus_random_without_true_top5_descriptive=(sub.high-sub.random)[keep].mean(),high_minus_random_true_top5=(sub.high-sub.random)[~keep].mean()))
    for name,rows in [('summary',summary),('folds',details),('budget_curve',curves),('heterogeneity',heter),('diagnostics',diag)]:pd.DataFrame(rows).to_csv(a.out/(name+'.csv'),index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
