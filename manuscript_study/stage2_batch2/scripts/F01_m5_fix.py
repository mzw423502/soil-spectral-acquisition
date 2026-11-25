#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F01_m5_fix.py —— M5 缩放与成分选择统一（只重算 M5，不动 M1–M4）

修正点：
  · 成分数选择与最终训练**都用 PLSRegression(scale=False)**
  · 内层每个训练折独立拟合两块尺度（sn, sm），应用到对应验证折
  · 外层：训练集拟合尺度 → 应用到测试折
  · 外层划分与成分候选区间保持不变

复用 B00 的 prep / fit_joint_scaler / transform_joint，避免口径漂移。

输出：predictions/m5_fixed.csv（dataset, sample_id, outer_fold, y, pred_M5_fixed, ncomp）

用法：python3 F01_m5_fix.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

NCOMP_CANDS = (2, 4, 8, 12, 16, 20)
N_INNER = 3


def load_b00(root):
    p = root / "stage2_batch2" / "scripts" / "B00_corrected_nested_fusion.py"
    spec = importlib.util.spec_from_file_location("b00", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pick_nc_unscaled(Xp, y, groups, cands=NCOMP_CANDS):
    """成分数选择：PLSRegression(scale=False)，与最终模型一致"""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    k = min(N_INNER, len(set(groups)))
    best = (np.inf, cands[0])
    for nc in cands:
        if nc > Xp.shape[1]:
            continue
        errs = []
        for tr, va in GroupKFold(n_splits=k).split(Xp, y, groups):
            m = PLSRegression(n_components=nc, scale=False).fit(Xp[tr], y[tr])
            errs.append(np.mean((y[va] - m.predict(Xp[va]).ravel()) ** 2))
        r = float(np.mean(errs))
        if r < best[0]:
            best = (r, nc)
    return best[1]


def m5_inner_oof(b00, Xn, Xm, y, groups):
    """严格交叉拟合的内层 OOF：每折独立拟合缩放 + 独立选成分"""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    n = len(y)
    k = min(N_INNER, len(set(groups)))
    oof = np.full(n, np.nan)
    ncs = []
    for tr, va in GroupKFold(n_splits=k).split(Xn, y, groups):
        Pn_tr, Pn_va = b00.prep(Xn[tr]), b00.prep(Xn[va])
        Pm_tr, Pm_va = b00.prep(Xm[tr]), b00.prep(Xm[va])
        sc = b00.fit_joint_scaler(Pn_tr, Pm_tr)          # 只在内层训练子集
        A = b00.transform_joint(Pn_tr, Pm_tr, sc, True)
        B = b00.transform_joint(Pn_va, Pm_va, sc, True)   # 同一缩放
        nc = pick_nc_unscaled(A, y[tr], groups[tr])
        ncs.append(nc)
        m = PLSRegression(n_components=nc, scale=False).fit(A, y[tr])
        oof[va] = np.clip(m.predict(B).ravel(), 0, None)
    return oof, ncs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    F = root / "stage2_final"
    b00 = load_b00(root)

    rows, summary = [], {}
    for name in ("neospectra", "icraf", "lucas"):
        d = np.load(F / "manifests" / f"data_{name}.npz", allow_pickle=True)
        Xn = d["X_nir"].astype(np.float64)
        Xm = d["X_mir"].astype(np.float64)
        y = d["y"].astype(np.float64)
        groups = d["groups"].astype(str)
        ids = d["ids"].astype(str)
        outs = pd.read_csv(F / "manifests" / "outer_split_manifest.csv")
        fold_of = outs.loc[outs["dataset"] == name, "outer_fold"].to_numpy(int)

        print(f"\n=== {name}: n={len(y)} ===", flush=True)
        pred = np.full(len(y), np.nan)
        ncs_all = []
        for f in sorted(set(fold_of)):
            te = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            oof, ncs = m5_inner_oof(b00, Xn[tr], Xm[tr], y[tr], groups[tr])
            ncs_all += ncs
            # 外层：训练集拟合缩放，应用到测试
            Pn_tr, Pn_te = b00.prep(Xn[tr]), b00.prep(Xn[te])
            Pm_tr, Pm_te = b00.prep(Xm[tr]), b00.prep(Xm[te])
            sc = b00.fit_joint_scaler(Pn_tr, Pm_tr)
            A = b00.transform_joint(Pn_tr, Pm_tr, sc, True)
            B = b00.transform_joint(Pn_te, Pm_te, sc, True)
            nc = pick_nc_unscaled(A, y[tr], groups[tr])
            from sklearn.cross_decomposition import PLSRegression
            m = PLSRegression(n_components=nc, scale=False).fit(A, y[tr])
            pred[te] = np.clip(m.predict(B).ravel(), 0, None)
            print(f"  fold {f}: nc={nc}  内层OOF MAE {np.mean(np.abs(y[tr]-oof)):.4f}  "
                  f"测试 MAE {np.nanmean(np.abs(y[te]-pred[te])):.4f}", flush=True)

        mae = float(np.nanmean(np.abs(y - pred)))
        # 与旧 M5 对比
        old = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
        old = old[old["dataset"] == name].set_index("sample_id").reindex(ids)
        old_m5 = old["pred_M5"].to_numpy(float)
        old_mae = float(np.nanmean(np.abs(y - old_m5)))
        print(f"  → M5 修正后 MAE {mae:.4f}  (修正前 {old_mae:.4f}, 差 {mae-old_mae:+.4f})")
        summary[name] = {"n": int(len(y)), "mae_fixed": mae, "mae_old": old_mae,
                         "delta": mae - old_mae, "ncomp_median": float(np.median(ncs_all))}
        for i in range(len(y)):
            rows.append({"dataset": name, "sample_id": ids[i],
                         "outer_fold": int(fold_of[i]), "y": float(y[i]),
                         "pred_M5_fixed": float(pred[i])})

    out = pd.DataFrame(rows)
    out.to_csv(S / "predictions" / "m5_fixed.csv", index=False)
    (S / "results" / "F01_m5_fix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {S/'results'/'F01_m5_fix.json'}")

    # 与 M4 对比
    old = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    print(f"\n=== M5(修正) vs M4 ===")
    for name in ("neospectra", "icraf", "lucas"):
        o = old[old["dataset"] == name]
        m4 = float(np.mean(np.abs(o["y"] - o["pred_M4"])))
        m5 = summary[name]["mae_fixed"]
        print(f"  {name:<12} M4 {m4:.4f}  M5 {m5:.4f}  M5−M4 {m5-m4:+.4f}  "
              f"{'M5更好' if m5 < m4 else 'M4更好'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
