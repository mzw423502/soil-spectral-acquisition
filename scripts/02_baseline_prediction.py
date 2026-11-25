#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02_baseline_prediction.py —— 第二阶段：SOC 基线预测能力

四个模型（样品级，非扫描级）：
  M0  训练折均值（最低基线）
  M1  NIR
  M2  MIR
  M3  NIR + MIR 拼接

验证：
  · 主：按土壤剖面分组 5 折（GroupKFold by pedon）—— 同一剖面的样品不跨折
  · 补：实验室留出（Lab 留一：Woodwell / KSSL 互测）—— 检验跨实验室泛化

纪律：所有预处理（SNV / SG 一阶导）、标准化、PLS 主成分数选择
      全部在**训练折内**完成，测试折只做 transform。

用法：python3 02_baseline_prediction.py --npz results/spectra_A.npz --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 预处理
def snv(X):
    """标准正态变换：逐样品去均值除标准差。对 2D (n, p)。"""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def sg_first_deriv(X, window=11, poly=2):
    """Savitzky-Golay 一阶导数（沿波段轴）。"""
    from scipy.signal import savgol_filter
    w = window if window % 2 == 1 else window + 1
    if w >= X.shape[1]:
        w = X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2
    return savgol_filter(X, window_length=w, polyorder=poly, deriv=1, axis=1)


def preprocess(Xtr, Xte, mode="snv_sg"):
    """在训练集上确定参数，再套用到测试集。SNV 是无参的逐样品操作，安全。"""
    if mode == "raw":
        return Xtr, Xte
    a = sg_first_deriv(snv(Xtr))
    b = sg_first_deriv(snv(Xte))
    return a, b


# ---------------------------------------------------------------- PLSR
def fit_pls(Xtr, ytr, groups, max_comp=20, inner_splits=3, seed=42):
    """内层 CV 选主成分数。返回 (model, best_ncomp)。"""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold

    n = min(max_comp, Xtr.shape[1], max(2, Xtr.shape[0] - 1))
    ug = len(set(groups))
    k = min(inner_splits, ug)
    if k < 2:
        m = PLSRegression(n_components=min(n, 5)).fit(Xtr, ytr)
        return m, m.n_components

    best = (np.inf, 1)
    for nc in range(1, n + 1):
        errs = []
        for tr, va in GroupKFold(n_splits=k).split(Xtr, ytr, groups):
            m = PLSRegression(n_components=nc)
            m.fit(Xtr[tr], ytr[tr])
            pred = m.predict(Xtr[va]).ravel()
            errs.append(np.mean((pred - ytr[va]) ** 2))
        r = float(np.mean(errs))
        if r < best[0]:
            best = (r, nc)
    m = PLSRegression(n_components=best[1]).fit(Xtr, ytr)
    return m, best[1]


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else float("nan")
    return {"n": int(len(y)), "rmse": rmse, "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "r2": r2,
            "rpd": sd / rmse if rmse > 0 else float("nan"),
            "rpiq": (float(np.percentile(y, 75) - np.percentile(y, 25))) / rmse if rmse > 0 else float("nan")}


# ---------------------------------------------------------------- 主流程
def run_cv(Xn, Xm, y, groups, label, out_rows, n_splits=5, seed=42):
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    oof = {k: np.full(len(y), np.nan) for k in ("M0", "M1", "M2", "M3")}
    ncomp = {k: [] for k in ("M1", "M2", "M3")}
    fold = 0
    for tr, te in gkf.split(Xn, y, groups):
        fold += 1
        # M0: 训练折均值
        oof["M0"][te] = y[tr].mean()

        # M1 NIR
        An, Bn = preprocess(Xn[tr], Xn[te])
        m1, nc = fit_pls(An, y[tr], groups[tr], seed=seed)
        oof["M1"][te] = m1.predict(Bn).ravel()
        ncomp["M1"].append(nc)

        # M2 MIR
        Am, Bm = preprocess(Xm[tr], Xm[te])
        m2, nc = fit_pls(Am, y[tr], groups[tr], seed=seed)
        oof["M2"][te] = m2.predict(Bm).ravel()
        ncomp["M2"].append(nc)

        # M3 NIR+MIR 拼接（各自预处理后 hstack）
        A3 = np.hstack([An, Am]); B3 = np.hstack([Bn, Bm])
        m3, nc = fit_pls(A3, y[tr], groups[tr], seed=seed)
        oof["M3"][te] = m3.predict(B3).ravel()
        ncomp["M3"].append(nc)

        print(f"    fold {fold}/{gkf.get_n_splits()}  "
              f"ncomp NIR={ncomp['M1'][-1]} MIR={ncomp['M2'][-1]} NIR+MIR={ncomp['M3'][-1]}",
              flush=True)

    res = {}
    for k in ("M0", "M1", "M2", "M3"):
        m = metrics(y, oof[k])
        res[k] = m
        print(f"  [{label}] {k}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"R2={m['r2']:.4f}  RPD={m['rpd']:.2f}")
    for i in range(len(y)):
        out_rows.append({"scheme": label, "i": i, "group": groups[i],
                         "y_true": y[i], "y_M0": oof["M0"][i], "y_M1": oof["M1"][i],
                         "y_M2": oof["M2"][i], "y_M3": oof["M3"][i]})
    return res, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-comp", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    Xn = d["nir"].astype(np.float64)
    Xm = d["mir"].astype(np.float64)
    y = d["y"].astype(np.float64)
    pedon = d["pedon"].astype(str)
    keys = d["keys"].astype(str)
    print(f"载入: NIR {Xn.shape}, MIR {Xm.shape}, y {y.shape}, pedon {len(set(pedon))}")

    # 剔除 y 缺失/异常与 pedon 空
    ok = np.isfinite(y) & (y > 0) & (pedon != "") & (pedon != "None")
    Xn, Xm, y, pedon, keys = Xn[ok], Xm[ok], y[ok], pedon[ok], keys[ok]
    print(f"有效样品: {len(y)}  剖面: {len(set(pedon))}  y 范围 {y.min():.3f}-{y.max():.3f}")

    rows = []
    print("\n=== 主验证：按剖面分组 5 折 ===")
    res_pedon, oof_pedon = run_cv(Xn, Xm, y, pedon, "pedon_group5", rows)

    # 实验室留出：需要 Lab 信息 -> 从 paired csv 读
    lab = {}
    meta_csv = out / "paired_samples_A.csv"
    if meta_csv.is_file():
        with open(meta_csv, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab[r["kssl_id"]] = r.get("labs", "")
    labv = np.array([lab.get(k, "") for k in keys])
    res_lab = {}
    for hold in ("Woodwell", "KSSL"):
        m = np.array([hold in s for s in labv])
        if m.sum() < 50 or (~m).sum() < 50:
            print(f"  [lab holdout {hold}] 样本不足，跳过（{m.sum()}/{len(m)}）")
            continue
        print(f"\n=== 实验室留出：训练 {hold} 之外的样品，测试 {hold}（n_train={(~m).sum()}, n_test={m.sum()}）===")
        tr, te = np.where(~m)[0], np.where(m)[0]
        sub = {}
        for tag, X in (("M1", Xn), ("M2", Xm), ("M3", None)):
            XL = Xn if tag == "M1" else (Xm if tag == "M2" else np.hstack([Xn, Xm]))
            A, B = preprocess(XL[tr], XL[te])
            mod, nc = fit_pls(A, y[tr], pedon[tr], max_comp=args.max_comp)
            p = mod.predict(B).ravel()
            sub[tag] = metrics(y[te], p)
            print(f"  [{hold}] {tag}: RMSE={sub[tag]['rmse']:.4f} R2={sub[tag]['r2']:.4f} "
                  f"ncomp={nc}")
        sub["M0"] = metrics(y[te], np.full(te.size, y[tr].mean()))
        print(f"  [{hold}] M0(mean): RMSE={sub['M0']['rmse']:.4f}")
        res_lab[hold] = sub

    # 写出
    with open(out / "oof_predictions.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "baseline_metrics.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["scheme", "model", "n", "rmse", "mae", "bias", "r2", "rpd", "rpiq"])
        for k, m in res_pedon.items():
            wr.writerow(["pedon_group5", k, m["n"], f"{m['rmse']:.5f}", f"{m['mae']:.5f}",
                         f"{m['bias']:.5f}", f"{m['r2']:.5f}", f"{m['rpd']:.3f}", f"{m['rpiq']:.3f}"])
        for hold, sub in res_lab.items():
            for k, m in sub.items():
                wr.writerow([f"lab_holdout_{hold}", k, m["n"], f"{m['rmse']:.5f}",
                             f"{m['mae']:.5f}", f"{m['bias']:.5f}", f"{m['r2']:.5f}",
                             f"{m['rpd']:.3f}", f"{m['rpiq']:.3f}"])

    summary = {"n_samples": int(len(y)), "n_pedons": int(len(set(pedon))),
               "pedon_group5": res_pedon, "lab_holdout": res_lab}
    (out / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {out/'baseline_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
