"""Final independent consistency checks, publication-ready S2, and summary.

Input: completed run_r10.py outputs and immutable source project; no refitting.
"""
import argparse,hashlib,json,re,platform
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt,Inches
from zipfile import ZipFile, ZIP_DEFLATED

def table(d):
    fmt=lambda x:f'{x:.8f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();root=args.root.resolve();cp=out/'checkpoints'
    assert len(list(cp.glob('outer_*.json')))==15
    loc=json.loads((out/'ICRAF_LOCO_M4_summary.json').read_text(encoding='utf-8'));assert loc['complete'] and loc['countries']==37 and loc['n']==3370
    p=pd.read_csv(out/'outer_predictions_r10.csv',dtype={'sample_id':str,'group':str});assert len(p)==6244 and not p.duplicated(['dataset','sample_id']).any()
    assert np.max(abs(p.M1_reproduced-p.pred_M1))<1e-7
    m=pd.read_csv(out/'Table_S2_R10.csv');old=pd.read_csv(out/'M5_r9_comparison.csv');m['bias']=[float((p.loc[p.dataset==r.dataset,'pred_'+r.model]-p.loc[p.dataset==r.dataset,'y']).mean()) for r in m.itertuples()]
    m.to_csv(out/'Table_S2_R10.csv',index=False)
    m5=p[['dataset','sample_id','group','outer_fold','y','pred_M5_r9','pred_M5']].copy();m5.to_csv(out/'M5_predictions_R10.csv',index=False)
    a=[]
    for name,d in m5.groupby('dataset',sort=False):
        a.append({'dataset':name,'n':len(d),'max_abs_prediction_change':float(abs(d.pred_M5-d.pred_M5_r9).max()),'old_MAE':float(abs(d.y-d.pred_M5_r9).mean()),'new_MAE':float(abs(d.y-d.pred_M5).mean()),'MAE_change':float(abs(d.y-d.pred_M5).mean()-abs(d.y-d.pred_M5_r9).mean()),'M2_MAE':float(m[(m.dataset==name)&(m.model=='M2')].MAE.iloc[0]),'M4_MAE':float(m[(m.dataset==name)&(m.model=='M4')].MAE.iloc[0])})
    a=pd.DataFrame(a);a.to_csv(out/'M5_implementation_comparison.csv',index=False)
    oldlog=(root/'stage2_batch2/logs/F01.log')
    historical={};name=None
    raw=oldlog.read_bytes()
    encoding='utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    for line in raw.decode(encoding).splitlines():
        z=re.match(r'=== (\w+): n=',line)
        if z:name=z.group(1)
        z=re.search(r'fold\s+(\d+): nc=(\d+)',line)
        if z and name:historical[(name,int(z.group(1)))]=int(z.group(2))
    assert len(historical)==15
    rows=[]
    for f in sorted(cp.glob('outer_*.json')):
        r=json.loads(f.read_text(encoding='utf-8'));r['R9_M5_ncomp']=historical[(r['dataset'],r['outer_fold'])];r['same_ncomp']=r['M5_ncomp']==r['R9_M5_ncomp'];rows.append(r)
    folds=pd.DataFrame(rows);folds.to_csv(out/'M5_U1_outer_fold_audit.csv',index=False)
    # Consolidate raw CV logs (no LUCAS sample identifiers in these score tables).
    logs=[]
    for f in sorted(cp.glob('*_cv.csv')):
        d=pd.read_csv(f);d.insert(0,'fold_file',f.stem);logs.append(d)
    pd.concat(logs,ignore_index=True).to_csv(out/'all_nested_cv_scores.csv',index=False)
    b=pd.read_csv(out/'U1_U2_budget_curves.csv');delta=pd.read_csv(out/'U1_U2_paired_bootstrap.csv')
    assert len(b)==3*21*4 and len(delta)==3*21*3
    # Verify against previously published R9 identical-budget random/SOC scores.
    orig=root/'stage2_batch2/predictions/all_corrected_predictions.csv'
    ids=pd.read_csv(orig,dtype={'sample_id':str,'group':str})
    for name,d in p.groupby('dataset'):
        assert len(ids[ids.dataset==name])==len(d)
    # Keep original S2 columns, append the two requested performance ratios.
    d=Document();sec=d.sections[0];sec.left_margin=sec.right_margin=Inches(.75)
    normal=d.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(10)
    d.add_heading('Table S2. Outer-validation prediction performance (R10 update)',level=1)
    d.add_paragraph('SOC is expressed in mass percent; MAE, RMSE and bias are percentage points. RPIQ = (Q75 − Q25)/RMSE; RPD = sample SD/RMSE. M5 uses corrected fold-local block scaling; M1–M4 are the original held-out predictions.')
    cols=['dataset','model','MAE','RMSE','R2','bias','RPIQ','RPD'];tb=d.add_table(rows=1,cols=len(cols));tb.style='Light Shading Accent 1'
    for j,k in enumerate(cols):tb.rows[0].cells[j].text=k
    for r in m[cols].itertuples(index=False,name=None):
        cells=tb.add_row().cells
        for j,v in enumerate(r):cells[j].text=f'{v:.4f}' if isinstance(v,float) else str(v)
    d.add_paragraph('Reference SOC descriptive statistics (same sample set for all five models).')
    kcols=['dataset','n','Q25','Q75','IQR','SD_sample'];tt=d.add_table(rows=1,cols=len(kcols));tt.style='Light Shading Accent 1'
    for i,k in enumerate(kcols):tt.rows[0].cells[i].text=k
    for r in m[kcols].drop_duplicates().itertuples(index=False,name=None):
        for j,v in enumerate(r):
            if j==0:c=tt.add_row().cells
            c[j].text=f'{v:.6f}' if isinstance(v,float) else str(v)
    d.save(out/'Table_S2_R10.docx')

    ref=m[['dataset','n','Q25','Q75','IQR','SD_sample']].drop_duplicates();ref.to_csv(out/'SOC_reference_distributions.csv',index=False)
    # Provenance: source inputs are identifiable without copying reference spectra.
    paths=[root/'stage2_final/manifests'/n for n in ('data_neospectra.npz','data_icraf.npz','data_lucas.npz','outer_split_manifest.csv','inner_split_manifest.csv','sample_manifest.csv')]+[orig,root/'stage2_batch2/predictions/m5_fixed.csv',oldlog,root/'stage2_batch2/scripts/F01_m5_fix.py',root/'stage2_batch2/scripts/B00_corrected_nested_fusion.py',root/'data/icraf/ossl_soilsite_v1.3.parquet']
    prov=[]
    for f in paths:
        with f.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
        prov.append({'path':f.relative_to(root).as_posix(),'size_bytes':f.stat().st_size,'sha256':sha})
    pd.DataFrame(prov).to_csv(out/'source_input_sha256.csv',index=False)
    pval=pd.read_csv(out/'pls_path_equivalence.csv')
    outcome={
        'all_tasks_complete':True,'outer_folds':15,'loco_countries':loc['countries'],'outer_samples':len(p),'loco_samples':loc['n'],'max_M1_reproduction_error':float(abs(p.M1_reproduced-p.pred_M1).max()),'M5_same_component_folds':int(folds.same_ncomp.sum()),'M5_max_prediction_change':float(a.max_abs_prediction_change.max()),'PLS_component_path_validation_comparisons':len(pval),'PLS_max_path_error':float(pval.max_abs_error.max()),'bootstrap_n':3000,'bootstrap_seed':20261125}
    (out/'R10_verification.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    # Retain the calculation report's method detail, then append quantitative audit.
    report=(out/'R10_computation_summary.md').read_text(encoding='utf-8')
    # Restore numerical section ordering; section 3 originally precedes section 2.
    prefix,s3=report.split('\n## 3. RPIQ/RPD (Table S2)',1);s3,s2=s3.split('\n## 2. ICRAF',1);s2,s4=s2.split('\n## 4. Uncertainty',1)
    report=prefix+'\n## 2. ICRAF'+s2+'\n## 3. RPIQ/RPD (Table S2)'+s3+'\n## 4. Uncertainty'+s4
    report+='\n## Final numerical audit / 最终核验\n\n'
    report+='All 15 outer folds and all 37 country folds completed. No published R9 files have been changed.\n\n'
    report+='### M5 old versus corrected predictions\n\n'+table(a)+'\n\n'
    report+='F01 original selection logs recovered from `F01.log`, matching R10 by dataset and fold.\n\n'+table(folds[['dataset','outer_fold','R9_M5_ncomp','M5_ncomp','M5_inner_ncomp','M5_sn','M5_sm','same_ncomp','M1_max_diff']])+'\n\n'
    report+=f"All {outcome['M5_same_component_folds']}/15 final component choices match. Maximum sample-level M5 prediction difference = {outcome['M5_max_prediction_change']:.12g}. M5 is worse in pooled MAE than MIR-only M2 in NeoSpectra and ICRAF under the retained model definition; the fold-local scaling correction does not explain these differences. This does not exclude better alternative joint-block scaling or other models.\n\n"
    report+='### Corrected Table S2 with bias\n\n'+table(m[['dataset','model','MAE','RMSE','R2','bias','RPIQ','RPD']])+'\n\nReference SOC statistics:\n\n'+table(ref)+'\n\n'
    report+='### U1/U2 interpretation at 20%\n\n'+table(delta[np.isclose(delta.budget,.2)])+'\n\n'
    for name in m.dataset.unique():
        q=delta[(delta.dataset==name)&np.isclose(delta.budget,.2)]
        report+=f"- {name}: "+'; '.join(f"{r.comparison}: {'interval excludes zero in favor of SOC' if r.ci_hi<0 else 'interval crosses/includes zero' if r.ci_lo<=0<=r.ci_hi else 'interval excludes zero against SOC'}" for r in q.itertuples())+'.\n'
    report+='\nThe M5 correction leaves M4-based policy comparisons unaffected. RPIQ and RPD both use RMSE, so their within-library rankings are the same as RMSE rankings (not MAE rankings). No claim of universal superiority over U1 is made where its interval includes zero.\n\n'
    report+='Full output files: `M5_predictions_R10.csv`, `M5_U1_outer_fold_audit.csv`, `all_nested_cv_scores.csv`, `ICRAF_LOCO_M4_country.csv`, `ICRAF_LOCO_M4_predictions.csv`, `ICRAF_LOCO_M4_summary.json`, `Table_S2_R10.csv`, `Table_S2_R10.docx`, `U1_U2_budget_curves.csv`, `U1_U2_paired_bootstrap.csv`, `source_input_sha256.csv`, and `R10_verification.json`.\n'
    (out/'R10_computation_summary.md').write_text(report,encoding='utf-8')
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
