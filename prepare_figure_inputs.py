from pathlib import Path
from shutil import copy2

base = Path(__file__).resolve().parent / 'manuscript_study/stage2_batch2/results'
for old, new in [('F02_lucas_loco.json','lucas_country_holdout_results.json'),('F03_gain_mechanism_final.json','icraf_gain_regression_results.json')]:
    copy2(base/old,base/new)
