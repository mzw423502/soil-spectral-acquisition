"""Cross-check the completed Cubist predictions and matched-budget scoring."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main(folder):
    read = lambda f: json.loads((folder / f).read_text(encoding='utf-8'))
    s = read('cubist_summary.json')
    p = read('fixed_fold_PLSR_validation.json')
    assert s['status'] == 'complete' and s['model'] == 'Cubist'
    assert s['n'] == p['n'] == 3698 and s['group_count'] == p['group_count'] == 670
    assert s['input_sha256'] == p['input_sha256']
    d = pd.read_csv(folder / 'cubist_icraf_oof_predictions.csv',
                    dtype={'sample_id': str, 'group': str}, float_precision='round_trip')
    c = pd.read_csv(folder / 'cubist_icraf_M4_calibration.csv')
    assert len(d) == 3698 and d.sample_id.is_unique and len(c) == 5 * 21
    assert d.groupby('group').outer_fold.nunique().max() == 1
    y, n, m, f = (d[k].to_numpy(float) for k in ['y', 'pred_M1', 'pred_M2', 'pred_M4'])
    w = d.weight_M4_NIR.to_numpy(float)
    assert np.isfinite([y, n, m, f]).all() and np.min([n, m, f]) >= 0
    np.testing.assert_allclose(f, w * n + (1 - w) * m, atol=1e-12, rtol=1e-12)
    en, em, ef = [np.abs(y - z) for z in (n, m, f)]
    chosen = np.zeros(len(d), dtype=bool)
    q = np.zeros(len(d), dtype=float)
    for fold, sub in d.groupby('outer_fold', sort=True):
        grid = c[c.outer_fold == fold]
        best = grid[grid.selected]
        assert len(best) == 1
        np.testing.assert_allclose(best.inner_oof_MAE.iloc[0], grid.inner_oof_MAE.min())
        np.testing.assert_allclose(w[sub.index], best.weight_M4_NIR.iloc[0])
        k = len(sub) // 5
        ids = sub.sort_values(['pred_M1', 'sample_id'], ascending=[False, True]).index
        chosen[ids[:k]] = True
        q[sub.index] = k / len(sub)
    assert np.array_equal(chosen, d.selected_20.to_numpy(bool)) and int(chosen.sum()) == 738
    np.testing.assert_allclose(q, d.random_fraction_20.to_numpy(float))
    a = np.where(chosen, ef, en).mean()
    b = ((1 - q) * en + q * ef).mean()
    actual = {
        'G_M_negative_n': int(np.sum(en < em)),
        'G_F_negative_n': int(np.sum(en < ef)),
        'MAE_M1': en.mean(), 'MAE_M2': em.mean(), 'MAE_M4': ef.mean(),
        'MAE_priority_20': a, 'MAE_random_20': b,
        'priority_minus_random': a - b, 'rel_reduction_pct': 100 * (1 - a / b),
    }
    for k, v in actual.items():
        np.testing.assert_allclose(v, s[k], rtol=1e-10, atol=1e-10, err_msg=k)
    print('PASS: 3,698 held-out predictions, 670 profiles, five fusion weights, 738 acquisitions')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE)
    main(ap.parse_args().out)
