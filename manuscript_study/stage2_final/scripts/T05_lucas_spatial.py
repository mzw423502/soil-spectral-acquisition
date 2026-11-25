#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T05_lucas_spatial.py —— LUCAS 空间分组敏感性（0.5° / 1° / 2°）

主分析已固定为 1°（T01 采用），本脚本只做敏感性对照，
**不根据结果反过来改主分析网格**。

每个网格大小独立重分组 → 独立五折 → 重训 M1(NIR)/M2(MIR) → 算 20% 高SOC优先 vs 随机。

用法：python3 T05_lucas_spatial.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SPLITS = 5
NCOMP_CANDS = (2, 4, 8, 12, 16, 20)


def snv(X):
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def sg1(X, window=11, poly=2):
    from scipy.signal import savgol_filter
    w = window if window % 2 == 1 else window + 1
    if w >= X.shape[1]:
        w = X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2
    return savgol_filter(X, window_length=w, polyorder=poly, deriv=1, axis=1)


def prep(Xtr, Xte):
    return sg1(snv(Xtr)), sg1(snv(Xte))


def oof_pls(X, y, groups):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    n = len(y); k = min(3, len(set(groups)))
    best = (np.inf, NCOMP_CANDS[0])
    for nc in NCOMP_CANDS:
        if nc > X.shape[1]:
            continue
        errs = []
        for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
            A, B = prep(X[tr], X[va])
            m = PLSRegression(n_components=nc).fit(A, y[tr])
            errs.append(np.mean((y[va] - m.predict(B).ravel()) ** 2))
        r = float(np.mean(errs))
        if r < best[0]:
            best = (r, nc)
    nc = best[1]
    oof = np.full(n, np.nan)
    for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
        A, B = prep(X[tr], X[va])
        m = PLSRegression(n_components=nc).fit(A, y[tr])
        oof[va] = np.clip(m.predict(B).ravel(), 0, None)
    return oof, nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_final"

    d = np.load(S / "manifests" / "data_lucas.npz", allow_pickle=True)
    Xn = d["X_nir"].astype(np.float64)
    Xm = d["X_mir"].astype(np.float64)
    y = d["y"].astype(np.float64)
    ids = d["ids"].astype(str)

    # 从原始 parquet 恢复经纬度（npz 没存）
    import pyarrow.parquet as pq
    site = pq.read_table(root / "data" / "lucas" / "ossl_soilsite_v1.3.parquet",
                         columns=["id.layer_local_c", "longitude.point_wgs84_dd",
                                  "latitude.point_wgs84_dd"]).to_pandas()
    site = site.drop_duplicates("id.layer_local_c").set_index("id.layer_local_c")
    lon = site.reindex(ids)["longitude.point_wgs84_dd"].to_numpy(float)
    lat = site.reindex(ids)["latitude.point_wgs84_dd"].to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    print(f"LUCAS n={len(y)}  有坐标 {ok.sum()}")

    rows = []
    for grid in (0.5, 1.0, 2.0, 5.0):
        gx = np.floor(lon / grid).astype(int)
        gy = np.floor(lat / grid).astype(int)
        g = np.array([f"{a}_{b}" for a, b in zip(gx, gy)])
        ng = len(set(g))
        oof_n, nc_n = oof_pls(Xn, y, g)
        oof_m, nc_m = oof_pls(Xm, y, g)
        e1 = np.abs(y - oof_n); em = np.abs(y - oof_m)
        n = len(y)
        k = int(np.floor(0.20 * n))
        s = np.zeros(n, bool); s[np.argsort(-oof_n)[:k]] = True
        hi = float(np.where(s, em, e1).mean())
        rnd = float(e1.mean() * (1 - k / n) + em.mean() * k / n)
        so = np.zeros(n, bool); so[np.argsort(-(e1 - em))[:k]] = True
        orc = float(np.where(so, em, e1).mean())
        rows.append({"grid_deg": grid, "n_groups": ng,
                     "max_per_group": int(pd.Series(g).value_counts().iloc[0]),
                     "mae_nir": float(e1.mean()), "mae_mir": float(em.mean()),
                     "mae_high20": hi, "mae_random20": rnd, "mae_oracle20": orc,
                     "vs_random_pct": (rnd - hi) / rnd * 100,
                     "eta_pct": (rnd - hi) / (rnd - orc) * 100 if rnd > orc else np.nan,
                     "ncomp_nir": nc_n, "ncomp_mir": nc_m})
        print(f"  {grid}°: {ng} 组 (最大 {rows[-1]['max_per_group']} 样品)  "
              f"NIR {rows[-1]['mae_nir']:.3f}  MIR {rows[-1]['mae_mir']:.3f}  "
              f"高SOC {hi:.3f} vs 随机 {rnd:.3f} ({rows[-1]['vs_random_pct']:+.1f}%)  "
              f"η={rows[-1]['eta_pct']:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(S / "results" / "lucas_spatial_results.csv", index=False)
    print(f"\n[OK] {S/'results'/'lucas_spatial_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
