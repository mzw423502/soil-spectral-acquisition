"""R10 supplementary computation. Read-only R9 inputs; independent output tree.

The PLS1 component path reuses a single sklearn fit at the maximal component
count, reconstructing each prefix from x_weights_, x_loadings_, y_loadings_.
The script validates that reconstruction against separately fitted models.
No held-out outcomes enter selection; all bootstrap intervals are conditional
on fitted outer predictions, matching R9's cluster bootstrap.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'
import argparse, hashlib, json, time, sys, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import pyarrow.parquet as pq

CANDS=(2,4,8,12,16,20)
WEIGHTS=np.round(np.arange(21)*.05,2)
BUDGETS=WEIGHTS.copy()
SEED=20261125
B=3000
NAMES=('neospectra','icraf','lucas')
DAT={}

def jd(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf-8'); tmp.replace(path)

def prep(x):
    x=np.asarray(x,dtype=np.float64)
    sd=x.std(1,keepdims=True);sd[sd==0]=1
    return savgol_filter((x-x.mean(1,keepdims=True))/sd,11,2,deriv=1,axis=1)

def init(root):
    global DAT
    root=Path(root);m=root/'stage2_final/manifests';p=root/'stage2_batch2/predictions'
    outs=pd.read_csv(m/'outer_split_manifest.csv',dtype={'sample_id':str,'group':str})
    old=pd.read_csv(p/'all_corrected_predictions.csv',dtype={'sample_id':str,'group':str})
    fixed=pd.read_csv(p/'m5_fixed.csv',dtype={'sample_id':str})
    for name in NAMES:
        d=np.load(m/f'data_{name}.npz',allow_pickle=True);ids=d['ids'].astype(str);y=d['y'].astype(float);g=d['groups'].astype(str)
        assert len(set(ids))==len(ids)
        mf=outs[outs.dataset==name].set_index('sample_id').reindex(ids)
        a=old[old.dataset==name].set_index('sample_id').reindex(ids);f=fixed[fixed.dataset==name].set_index('sample_id').reindex(ids)
        assert len(a)==len(y) and np.allclose(a.y,y) and np.allclose(f.y,y) and np.allclose(mf.y,y)
        assert np.array_equal(mf.group.astype(str),g) and np.array_equal(a.group.astype(str),g)
        assert np.array_equal(mf.outer_fold,a.outer_fold) and np.array_equal(mf.outer_fold,f.outer_fold)
        for fold in sorted(mf.outer_fold.unique()):
            test=mf.outer_fold.to_numpy()==fold;assert not set(g[test])&set(g[~test])
        a=a.reset_index().rename(columns={'index':'sample_id'});a['pred_M5_original']=a.pred_M5;a['pred_M5']=f.pred_M5_fixed.to_numpy()
        country=None
        if name=='icraf':
            site=pq.read_table(root/'data/icraf/ossl_soilsite_v1.3.parquet',columns=['id.layer_local_c','loc.country_src_txt']).to_pandas()
            assert site['id.layer_local_c'].is_unique
            c=site.set_index('id.layer_local_c').reindex(ids)['loc.country_src_txt'];assert c.notna().all();country=c.astype(str).to_numpy()
            assert pd.DataFrame({'g':g,'c':country}).groupby('g').c.nunique().max()==1
            v=pd.Series(country).value_counts();assert len(v[v>=30])==37 and v[v>=30].sum()==3370
        DAT[name]={'Xn':prep(d['X_nir']),'Xm':prep(d['X_mir']),'y':y,'g':g,'ids':ids,'fold':mf.outer_fold.to_numpy(int),'old':a,'country':country}

def gsplit(idx,groups):
    k=min(3,len(set(groups[idx])));assert k>=2
    for t,v in GroupKFold(k).split(np.zeros(len(idx)),groups=groups[idx]):
        tr,va=idx[t],idx[v];assert not set(groups[tr])&set(groups[va]);yield tr,va

def path_fit(A, y, Bx, candidates=CANDS, scale=True):
    cs=[k for k in candidates if k<=min(A.shape[1],len(A)-1)];assert cs
    m=PLSRegression(n_components=max(cs),scale=scale).fit(A,y)
    # Prefix coefficients use precisely the corresponding latent-component subspace.
    # Do not use rotations from the maximal model blindly; recover the prefix.
    coef=[]
    for k in cs:
        W=m.x_weights_[:,:k];P=m.x_loadings_[:,:k];Q=m.y_loadings_[:,:k]
        rotations=W@np.linalg.pinv(P.T@W)
        beta=(rotations@Q.T).ravel()*float(m._y_std[0])/m._x_std
        coef.append(beta)
    pred=(Bx-m._x_mean)@np.asarray(coef).T+float(m._y_mean[0])
    return dict(zip(cs,pred.T))

def scaler(d,tr):
    sn=float(np.std(d['Xn'][tr]));sm=float(np.std(d['Xm'][tr]));return sn or 1.,sm or 1.

def mat(d,idx,mode,sc=None):
    if mode=='n':return d['Xn'][idx]
    if mode=='m':return d['Xm'][idx]
    assert sc is not None
    return np.hstack((d['Xn'][idx]/sc[0],d['Xm'][idx]/sc[1]))

def tune(d,tr,groups,mode,log,context):
    scores={k:[] for k in CANDS}
    for i,(it,iv) in enumerate(gsplit(tr,groups),1):
        sc=scaler(d,it) if mode=='joint' else None
        A,V=mat(d,it,mode,sc),mat(d,iv,mode,sc)
        pp=path_fit(A,d['y'][it],V,scale=mode!='joint')
        row={'context':context,'mode':mode,'inner_fold':i,'n_train':len(it),'n_val':len(iv),'sn':sc[0] if sc else None,'sm':sc[1] if sc else None}
        for k,p in pp.items():
            mse=float(np.mean((d['y'][iv]-p)**2));scores[k].append(mse);row[f'mse_{k}']=mse
        log.append(row)
    v={k:float(np.mean(z)) for k,z in scores.items() if len(z)==min(3,len(set(groups[tr])))}
    nc=min(v,key=v.get)
    log.append({'context':context,'mode':mode,'inner_fold':0,'n_train':len(tr),'selected':nc,**{f'mse_{k}':s for k,s in v.items()}})
    return nc

def fit(d,tr,te,mode,nc):
    sc=scaler(d,tr) if mode=='joint' else None
    A,V=mat(d,tr,mode,sc),mat(d,te,mode,sc)
    # A final model is independently fitted using the selected number.
    model=PLSRegression(n_components=nc,scale=mode!='joint').fit(A,d['y'][tr])
    return np.clip(model.predict(V).ravel(),0,None),sc

def strict_oof(d,idx,groups,mode,logger,context,extra=None):
    o=np.full(len(d['y']),np.nan);extra_preds=[];choices=[]
    for i,(tr,va) in enumerate(gsplit(idx,groups),1):
        nc=tune(d,tr,groups,mode,logger,f'{context}/inner{i}')
        if extra is None:
            o[va],_=fit(d,tr,va,mode,nc)
        else:
            pr,_=fit(d,tr,np.concatenate([va,extra]),mode,nc)
            o[va]=pr[:len(va)];extra_preds.append(pr[len(va):])
        choices.append(nc)
    assert np.isfinite(o[idx]).all()
    return o[idx],extra_preds,choices

def validate_path(out):
    records=[]
    for name in NAMES:
        d=DAT[name];idx=np.arange(len(d['y']));tr,te=next(gsplit(idx,d['g']))
        # Actual spectral rows and groups, independent of all reported testing.
        tr=tr[:min(len(tr),350)];te=te[:min(len(te),100)]
        for mode in ('n','m','joint'):
            sc=scaler(d,tr) if mode=='joint' else None;A,V=mat(d,tr,mode,sc),mat(d,te,mode,sc)
            p=path_fit(A,d['y'][tr],V,scale=mode!='joint')
            for k,pr in p.items():
                ref=PLSRegression(n_components=k,scale=mode!='joint').fit(A,d['y'][tr]).predict(V).ravel()
                diff=float(np.max(np.abs(pr-ref)));records.append({'dataset':name,'mode':mode,'ncomp':k,'max_abs_error':diff})
                assert np.allclose(pr,ref,rtol=1e-7,atol=1e-8),(name,mode,k,diff)
    pd.DataFrame(records).to_csv(out/'pls_path_equivalence.csv',index=False)

def outer_job(name,fold,out):
    d=DAT[name];g=d['g'];te=np.flatnonzero(d['fold']==fold);tr=np.flatnonzero(d['fold']!=fold);logs=[]
    # M5: no validation rows participate in an inner block scale.
    nc=tune(d,tr,g,'joint',logs,'M5_outer_selection');p5,sc=fit(d,tr,te,'joint',nc)
    o5,_,nc5=strict_oof(d,tr,g,'joint',logs,'M5_oof')
    # U1 reuses the same disjoint 3-fold fitting protocol as B00 M1.
    o1,sub,ncs=strict_oof(d,tr,g,'n',logs,'U1',extra=te)
    u1=np.std(np.asarray(sub),axis=0,ddof=0)
    nc1=tune(d,tr,g,'n',logs,'M1_reproduction');p1,_=fit(d,tr,te,'n',nc1)
    old=d['old'].iloc[te];delta=float(np.max(np.abs(p1-old.pred_M1)))
    # U2 fixed 20 nonzero principal axes, fit only outer training spectra.
    A,V=d['Xn'][tr],d['Xn'][te];pca=PCA(n_components=min(20,A.shape[1],len(tr)-1),svd_solver='full').fit(A)
    usable=pca.explained_variance_>np.finfo(float).eps*max(pca.explained_variance_)
    u2=np.sqrt(np.sum(pca.transform(V)[:,usable]**2/pca.explained_variance_[usable],axis=1))
    pred=old.copy();pred['pred_M5_r9']=pred.pred_M5;pred['pred_M5']=p5;pred['U1']=u1;pred['U2']=u2;pred['M1_reproduced']=p1
    pred['U1_subfold_1']=sub[0];pred['U1_subfold_2']=sub[1];pred['U1_subfold_3']=sub[2]
    pred.to_csv(out/f'outer_{name}_{fold}.csv',index=False)
    pd.DataFrame(logs).to_csv(out/f'outer_{name}_{fold}_cv.csv',index=False)
    result={'dataset':name,'outer_fold':fold,'n_test':len(te),'M5_ncomp':nc,'M5_inner_ncomp':nc5,'M5_sn':sc[0],'M5_sm':sc[1],'M5_inner_oof_mae':float(np.mean(np.abs(d['y'][tr]-o5))),'M1_ncomp':nc1,'U1_ncomp':ncs,'M1_max_diff':delta,'U2_components':int(usable.sum()),'M5_mae':float(np.mean(np.abs(d['y'][te]-p5)))}
    jd(out/f'outer_{name}_{fold}.json',result)
    return result

def loco_job(country,out):
    d=DAT['icraf'];c=d['country'];te=np.flatnonzero(c==country);tr=np.flatnonzero(c!=country);logs=[]
    assert len(te)>=30 and country not in c[tr] and not set(d['g'][tr])&set(d['g'][te])
    o1,_,n1=strict_oof(d,tr,c,'n',logs,'LOCO_NIR');o2,_,n2=strict_oof(d,tr,c,'m',logs,'LOCO_MIR')
    ws={float(w):float(np.mean(np.abs(d['y'][tr]-(w*o1+(1-w)*o2)))) for w in WEIGHTS}
    w=min(ws,key=ws.get);nc1=tune(d,tr,c,'n',logs,'LOCO_outer_NIR');nc2=tune(d,tr,c,'m',logs,'LOCO_outer_MIR')
    p1,_=fit(d,tr,te,'n',nc1);p2,_=fit(d,tr,te,'m',nc2);p4=w*p1+(1-w)*p2
    sy=d['y'][te];e1=np.abs(sy-p1);e4=np.abs(sy-p4);k=int(np.floor(.2*len(te)+1e-10));q=k/len(te)
    order=np.lexsort((d['ids'][te],-p1));chosen=np.zeros(len(te),bool);chosen[order[:k]]=True
    es=np.where(chosen,e4,e1);er=(1-q)*e1+q*e4
    t=pd.DataFrame({'country':country,'sample_id':d['ids'][te],'group':d['g'][te],'y':sy,'pred_M1':p1,'pred_M2':p2,'pred_M4':p4,'selected':chosen,'error_priority':es,'error_random_expectation':er,'error_M4':e4})
    slug=hashlib.sha256(country.encode()).hexdigest()[:12]
    t.to_csv(out/f'loco_{slug}.csv',index=False);pd.DataFrame(logs).to_csv(out/f'loco_{slug}_cv.csv',index=False)
    r={'country':country,'n':len(te),'n_train':len(tr),'train_countries':len(set(c[tr])),'selected':k,'w_M4':w,'nc_NIR':nc1,'nc_MIR':nc2,'inner_nc_NIR':n1,'inner_nc_MIR':n2,'weight_curve':ws,'mae_M1':float(e1.mean()),'mae_M2':float(np.abs(sy-p2).mean()),'mae_M4':float(e4.mean()),'mae_priority':float(es.mean()),'mae_random':float(er.mean()),'priority_minus_random':float((es-er).mean())}
    jd(out/f'loco_{slug}.json',r);return r

def boot_idx(groups):
    labs,ix=np.unique(np.asarray(groups).astype(str),return_inverse=True)
    counts=np.bincount(ix);rng=np.random.default_rng(SEED);sample=rng.integers(0,len(labs),size=(B,len(labs)))
    den=counts[sample].sum(1)
    def ci(values):
        sums=np.bincount(ix,weights=np.asarray(values));v=sums[sample].sum(1)/den
        return np.quantile(v,[.025,.975]).tolist()
    return ci

def assemble(root,out):
    chunks=out/'checkpoints';rows=[];old_metrics=[];ranked=[];bootrows=[]
    arrays=[]
    for name in NAMES:
        files=[chunks/f'outer_{name}_{f}.csv' for f in range(1,6)]
        if not all(p.exists() for p in files):continue
        d=pd.concat([pd.read_csv(p,dtype={'sample_id':str,'group':str}) for p in files],ignore_index=True);arrays.append(d)
        assert d.sample_id.is_unique and len(d)==len(DAT[name]['y'])
        y=d.y.to_numpy();q1,q3=np.quantile(y,[.25,.75]);sd=float(np.std(y,ddof=1));iqr=q3-q1
        for key in ('M1','M2','M3','M4','M5','M5_r9'):
            p=d['pred_'+key].to_numpy();r=y-p;rmse=float(np.sqrt(np.mean(r*r)));row={'dataset':name,'model':key,'n':len(d),'Q25':q1,'Q75':q3,'IQR':iqr,'SD_sample':sd,'MAE':float(np.mean(abs(r))),'RMSE':rmse,'RPIQ':iqr/rmse,'RPD':sd/rmse,'R2':1-float(np.sum(r*r)/np.sum((y-y.mean())**2))}
            (old_metrics if key=='M5_r9' else rows).append(row)
        e1=np.abs(y-d.pred_M1);e4=np.abs(y-d.pred_M4);ci=boot_idx(d.group)
        scores={'SOC':'pred_M1','U1':'U1','U2':'U2'}
        for budget in BUDGETS:
            errors={key:e1.copy() for key in (*scores,'random')};total=0
            for f,sub in d.groupby('outer_fold',sort=False):
                ix=sub.index.to_numpy();k=int(np.floor(budget*len(sub)+1e-10));q=k/len(sub);total+=k
                for key,col in scores.items():
                    sel=sub.sort_values([col,'sample_id'],ascending=[False,True],kind='mergesort').index[:k]
                    errors[key].loc[sel]=e4.loc[sel]
                errors['random'].loc[ix]=(1-q)*e1.loc[ix]+q*e4.loc[ix]
            for key,z in errors.items():
                lo,hi=ci(z);ranked.append({'dataset':name,'budget':float(budget),'n':len(d),'k':total,'strategy':key,'MAE':float(z.mean()),'ci_lo':lo,'ci_hi':hi})
            for key in ('U1','U2','random'):
                delta=errors['SOC']-errors[key];lo,hi=ci(delta);bootrows.append({'dataset':name,'budget':float(budget),'k':total,'comparison':'SOC_minus_'+key,'delta_MAE':float(delta.mean()),'ci_lo':lo,'ci_hi':hi})
    if arrays:
        pd.concat(arrays).to_csv(out/'outer_predictions_r10.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'Table_S2_R10.csv',index=False);pd.DataFrame(old_metrics).to_csv(out/'M5_r9_comparison.csv',index=False)
        pd.DataFrame(ranked).to_csv(out/'U1_U2_budget_curves.csv',index=False);pd.DataFrame(bootrows).to_csv(out/'U1_U2_paired_bootstrap.csv',index=False)
    countries=[];preds=[]
    for p in sorted(chunks.glob('loco_*.json')):
        obj=json.loads(p.read_text(encoding='utf-8'));countries.append(obj);preds.append(pd.read_csv(p.with_suffix('.csv'),dtype={'sample_id':str,'group':str}))
    if countries:
        df=pd.concat(preds,ignore_index=True);assert df.sample_id.is_unique
        df.to_csv(out/'ICRAF_LOCO_M4_predictions.csv',index=False)
        pd.DataFrame([{k:v for k,v in r.items() if k!='weight_curve'} for r in countries]).to_csv(out/'ICRAF_LOCO_M4_country.csv',index=False)
        ci=boot_idx(df.country);errors={x:df['error_'+x] for x in ('priority','random_expectation','M4')}
        summary={'countries':len(countries),'n':len(df),'n_selected':int(df.selected.sum()),'expected_countries':37,'expected_n':3370,'complete':len(countries)==37 and len(df)==3370,'improved_countries':sum(r['priority_minus_random']<0 for r in countries),'MAE':{k:float(v.mean()) for k,v in errors.items()},'CI':{k:ci(v) for k,v in errors.items()},'contrasts':{k:{'delta':float((errors['priority']-errors[k]).mean()),'ci':ci(errors['priority']-errors[k])} for k in ('random_expectation','M4')}}
        jd(out/'ICRAF_LOCO_M4_summary.json',summary)
    build_report(out)

def mdtable(df):
    cols=list(df.columns);fmt=lambda v: f'{v:.8f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))

def build_report(out):
    cp=out/'checkpoints';nouter=len(list(cp.glob('outer_*.json')));nloco=len(list(cp.glob('loco_*.json')))
    t=['# R10 computation summary','',f'Completion: outer folds {nouter}/15; country folds {nloco}/37. Partial outputs must not be interpreted as full-dataset estimates.','',
    '## 1. M5 implementation audit','',
    'B00 fitted one global standard deviation per modality block, then divided the independently SNV + SG(11,2,1) preprocessed NIR and MIR blocks by their training-derived factors. The original F01 outer validation scaler excluded the outer test set, but `pick_nc_unscaled(A,...)` received A scaled using its entire calibration partition, so the tuning-validation spectra contributed to each scaling factor. The same issue is present at the sub-inner selection stage. B00 also used the default scale=True for joint-component tuning; F01 corrected this switch but not the fold-local scaler. R10 recomputes both factors within every tuning split and uses scale=False throughout M5. The original global-SD definition is retained; it equalizes block RMS approximately, not total Frobenius block energy. Component candidates remain 2,4,8,12,16,20 and grouped three-fold CV minimizes untruncated mean fold MSE. Predictions are clipped only after model fitting as in R9. No outer labels were used to tune.',
    '', 'The component-path acceleration was checked against independently fitted sklearn models on real spectral subsets (see `pls_path_equivalence.csv`). The outer fitting uses individually fitted selected-component models. Per-fit tuning scores and scaling factors are in the checkpoint CV tables.']
    if (out/'pls_path_equivalence.csv').exists():
        e=pd.read_csv(out/'pls_path_equivalence.csv');t+=['',f'Path validation: {len(e)} comparisons; maximum absolute prediction discrepancy {e.max_abs_error.max():.12g}.']
    if (out/'Table_S2_R10.csv').exists():
        v=pd.read_csv(out/'Table_S2_R10.csv');old=pd.read_csv(out/'M5_r9_comparison.csv');m=v[v.model.isin(['M2','M4','M5'])][['dataset','model','MAE','RMSE','R2']];m=pd.concat([m,old[['dataset','model','MAE','RMSE','R2']]],ignore_index=True)
        t+=['',mdtable(m),'','## 3. RPIQ/RPD (Table S2)','', 'IQR uses linear empirical quantiles; SD uses the sample standard deviation (ddof=1). All predictions paired to reference SOC by dataset and sample ID. M1–M4 retain R9 predictions; M5 is R10-corrected.','',mdtable(v)]
    l=out/'ICRAF_LOCO_M4_summary.json';t+=['','## 2. ICRAF–ISRIC leave-one-country-out M4','', 'Country is restored by exact sample identifier from OSSL v1.3. All countries with >=30 samples are targets; all other countries, including small countries, are calibration data. Each target fold fits independent country-grouped inner/sub-inner CV and selects w using strict calibration OOF predictions on the 0.05 grid. Top floor(20% × n) selected by predicted SOC with sample-ID tie-break. Random comparator is the exact expectation at the same fold-specific quota. Bootstrap resamples countries and uses pooled sample weighting (3000, seed 20261125).']
    if l.exists():
        o=json.loads(l.read_text(encoding='utf-8'));t+=['',f"**{'COMPLETE' if o['complete'] else 'INCOMPLETE — do not use in paper'}:** {o['countries']}/37 countries; n={o['n']}/3370; {o['n_selected']} selected; {o['improved_countries']}/{o['countries']} improved against random.",'',mdtable(pd.DataFrame([{'strategy':k,'MAE':v,'CI_low':o['CI'][k][0],'CI_high':o['CI'][k][1]} for k,v in o['MAE'].items()])),'',mdtable(pd.DataFrame([{'contrast':'priority_minus_'+k,'delta':v['delta'],'CI_low':v['ci'][0],'CI_high':v['ci'][1]} for k,v in o['contrasts'].items()]))]
    t+=['','## 4. Uncertainty and spectral-distance baselines','', 'U1: SD (ddof=0) across three NIR-only inner calibration models with independently tuned component counts. U2: Euclidean distance in training-whitened PCA score space, using 20 PCs fixed a priori (or fewer nonzero PCs); no SOC/MIR labels/features used in PCA ranking. All policies allocate identical floor-rounded quotas within the original held-out folds; post-acquisition M4 predictions identical. Random is exact expected MAE. Group bootstrap follows R9 (3000, seed 20261125). Report conditional paired differences, negative means SOC priority better.']
    if (out/'U1_U2_budget_curves.csv').exists():
        a=pd.read_csv(out/'U1_U2_budget_curves.csv');b=pd.read_csv(out/'U1_U2_paired_bootstrap.csv');t+=['',mdtable(a[np.isclose(a.budget,.2)]),'',mdtable(b[np.isclose(b.budget,.2)]),'','All 21 budgets (0–100%) and intervals are in the CSV files.']
    t+=['','## Reproducibility','',f'Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}.','No manuscript or previously published R9 files overwritten.']
    (out/'R10_computation_summary.md').write_text('\n'.join(t)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--part',choices=['all','outer','loco','summarize'],default='all');args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    init(args.root)
    if not (out/'pls_path_equivalence.csv').exists():validate_path(out)
    jobs=[]
    if args.part in ('all','outer'):
        for name in NAMES:
            for fold in sorted(set(DAT[name]['fold'])):
                if not (cp/f'outer_{name}_{fold}.json').exists():jobs.append((outer_job,(name,int(fold),cp)))
    if args.part in ('all','loco'):
        v=pd.Series(DAT['icraf']['country']).value_counts();targets=sorted(v[v>=30].index)
        for c in targets:
            if not (cp/f'loco_{hashlib.sha256(c.encode()).hexdigest()[:12]}.json').exists():jobs.append((loco_job,(c,cp)))
    start=time.time();print(f'{len(jobs)} jobs, {args.workers} workers',flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args.root,)) as pool:
            ft={pool.submit(fn,*params):params for fn,params in jobs}
            for i,f in enumerate(as_completed(ft),1):
                val=f.result();print(f'[{i}/{len(jobs)}] {val.get("dataset",val.get("country"))} {val.get("outer_fold", "")} {time.time()-start:.1f}s',flush=True)
    assemble(args.root,out)
    print('Results written to',out,flush=True)

if __name__=='__main__':main()
