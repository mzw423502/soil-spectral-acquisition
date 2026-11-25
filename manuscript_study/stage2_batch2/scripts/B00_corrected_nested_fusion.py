#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B00_corrected_nested_fusion.py —— 修正三处口径错误 + 保存真正的内层 OOF

修正内容（对应用户 §3.1–3.4）：
  ① 成分数选择改用**预处理后**光谱（原版用原始光谱选、再用预处理谱训练）
  ② M5 的数据块缩放参数**只在训练集拟合**（原版训练/测试各算 std）
     并提供两个 M5 变体：scale=False 的显式等权块 + 传统联合 PLSR 对照
  ③ 随机基线按**每个 aug 模型**各自计算（原版 M3/M4/M5 共用 M2 的随机值）
  ④ 严格交叉拟合：内层每个验证折的超参数只由其内层训练子集决定
     并独立保存到 cache/inner_oof/

参数（锁定）：
  NCOMP_CANDS=[2,4,8,12,16,20]  N_INNER=3
  SG: window=11, polyorder=2, deriv=1  SEED=42

用法：python3 B00_corrected_nested_fusion.py --root <manuscript_study 根> [--dataset icraf]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_INNER = 3
NCOMP_CANDS = (2, 4, 8, 12, 16, 20)
SG_W, SG_P, SG_D = 11, 2, 1
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
BUDGETS = np.round(np.arange(0, 1.0001, 0.05), 2)


# ---------------------------------------------------------------- 预处理
def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def sg1(X):
    from scipy.signal import savgol_filter
    w = SG_W
    if w >= X.shape[1]:
        w = X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2
    return savgol_filter(X, window_length=w, polyorder=SG_P, deriv=SG_D, axis=1)


def prep(X):
    """单块完整预处理：SNV → SG1"""
    return sg1(snv(X))


# ---------------------------------------------------------------- M5 缩放器
def fit_joint_scaler(Pn_tr, Pm_tr):
    """只在训练集拟合每块尺度（std），返回 (sn, sm)"""
    sn = float(np.std(Pn_tr))
    sm = float(np.std(Pm_tr))
    return (sn if sn > 0 else 1.0, sm if sm > 0 else 1.0)


def transform_joint(Pn, Pm, scaler, equalize=True):
    sn, sm = scaler
    if equalize:
        Pn = Pn / sn
        Pm = Pm / sm
    return np.hstack([Pn, Pm])


# ---------------------------------------------------------------- PLSR 工具
def pick_ncomp(Xp, y, groups, cands=NCOMP_CANDS, inner=N_INNER):
    """在已预处理的光谱上选成分数（关键修正①）"""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    ug = len(set(groups)); k = min(inner, ug)
    best = (np.inf, cands[0])
    if k < 2:
        return min(cands[0], Xp.shape[1])
    for nc in cands:
        if nc > Xp.shape[1]:
            continue
        errs = []
        for tr, va in GroupKFold(n_splits=k).split(Xp, y, groups):
            m = PLSRegression(n_components=nc).fit(Xp[tr], y[tr])
            errs.append(np.mean((y[va] - m.predict(Xp[va]).ravel()) ** 2))
        r = float(np.mean(errs))
        if r < best[0]:
            best = (r, nc)
    return min(best[1], Xp.shape[1])


def oof_inner(Xraw, y, groups, equalize_block=None):
    """严格交叉拟合的内层 OOF（关键修正④）

    对每个内层验证折：成分数只由该折的**内层训练子集**决定，
    避免用于选超参的标签同时影响该样品的收益标签。
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    n = len(y)
    ug = len(set(groups)); k = min(N_INNER, ug)
    oof = np.full(n, np.nan)
    ncs = []
    Xsplit = Xraw[0] if isinstance(Xraw, tuple) else Xraw
    if k < 2:
        Xp = prep(Xraw[0]) if isinstance(Xraw, tuple) else prep(Xraw)
        nc = min(NCOMP_CANDS[0], Xp.shape[1])
        m = PLSRegression(n_components=nc).fit(Xp, y)
        return np.clip(m.predict(Xp).ravel(), 0, None), [nc]
    for tr, va in GroupKFold(n_splits=k).split(Xsplit, y, groups):
        if isinstance(Xraw, tuple):        # M5：两块联合
            Xn, Xm = Xraw
            Pn_tr, Pn_va = prep(Xn[tr]), prep(Xn[va])
            Pm_tr, Pm_va = prep(Xm[tr]), prep(Xm[va])
            sc = fit_joint_scaler(Pn_tr, Pm_tr)      # 只在内层训练子集拟合
            A = transform_joint(Pn_tr, Pm_tr, sc, equalize_block)
            B = transform_joint(Pn_va, Pm_va, sc, equalize_block)
            nc = pick_ncomp(A, y[tr], groups[tr])
            m = PLSRegression(n_components=nc, scale=False).fit(A, y[tr])
            oof[va] = np.clip(m.predict(B).ravel(), 0, None)
        else:
            A = prep(Xraw[tr]); B = prep(Xraw[va])
            nc = pick_ncomp(A, y[tr], groups[tr])
            m = PLSRegression(n_components=nc).fit(A, y[tr])
            oof[va] = np.clip(m.predict(B).ravel(), 0, None)
        ncs.append(nc)
    return oof, ncs


def fit_predict_outer(Xraw, y, groups, Xte_raw, equalize_block=None):
    """外层：训练集预处理 → 选成分 → 训练 → 预测测试"""
    from sklearn.cross_decomposition import PLSRegression
    if isinstance(Xraw, tuple):
        Xn, Xm = Xraw
        Pn_tr, Pm_tr = prep(Xn), prep(Xm)
        sc = fit_joint_scaler(Pn_tr, Pm_tr)
        A = transform_joint(Pn_tr, Pm_tr, sc, equalize_block)
        Pn_te, Pm_te = prep(Xte_raw[0]), prep(Xte_raw[1])
        B = transform_joint(Pn_te, Pm_te, sc, equalize_block)
    else:
        A = prep(Xraw)
        B = prep(Xte_raw)
    nc = pick_ncomp(A, y, groups)
    m = PLSRegression(n_components=nc, scale=False if isinstance(Xraw, tuple) else True)
    m.fit(A, y)
    return np.clip(m.predict(B).ravel(), 0, None), nc


# ---------------------------------------------------------------- 主流程
def run(name, root, S):
    M = S / "manifests" if (S / "manifests").exists() else root / "stage2_final" / "manifests"
    d = np.load(M / f"data_{name}.npz", allow_pickle=True)
    Xn, Xm = d["X_nir"].astype(np.float64), d["X_mir"].astype(np.float64)
    y = d["y"].astype(np.float64)
    groups = d["groups"].astype(str)
    ids = d["ids"].astype(str)
    n = len(y)

    man = pd.read_csv(M / "sample_manifest.csv")
    man = man[man["dataset"] == name].set_index("sample_id")
    depth = man.reindex(ids)["depth_top"].to_numpy(float)
    outs = pd.read_csv(M / "outer_split_manifest.csv")
    fold_of = outs.loc[outs["dataset"] == name, "outer_fold"].to_numpy(int)

    print(f"\n{'='*72}\n{name}: n={n} groups={len(set(groups))} "
          f"NIR {Xn.shape[1]} MIR {Xm.shape[1]}", flush=True)

    oof_dir = S / "cache" / "inner_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    pred_recs, fold_rows = [], []
    t0 = time.time()

    for f in sorted(set(fold_of)):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        print(f"\n--- fold {f}: train {len(tr)} / test {len(te)} ---", flush=True)

        # ---------- 内层 OOF（严格交叉拟合，五模型） ----------
        cache = oof_dir / f"inner_{name}_fold{f}.csv"
        if cache.exists():
            inn = pd.read_csv(cache)
            print(f"    [cache] {cache.name}", flush=True)
        else:
            o1, nc1 = oof_inner(Xn[tr], y[tr], groups[tr])
            o2, nc2 = oof_inner(Xm[tr], y[tr], groups[tr])
            o3 = 0.5 * o1 + 0.5 * o2
            # M4 权重（内层 OOF 上选）
            bw, bwm = 0.5, np.inf
            for w in WEIGHTS:
                mae = float(np.mean(np.abs(y[tr] - (w * o1 + (1 - w) * o2))))
                if mae < bwm:
                    bwm, bw = mae, float(w)
            o4 = bw * o1 + (1 - bw) * o2
            o5, nc5 = oof_inner((Xn[tr], Xm[tr]), y[tr], groups[tr], equalize_block=True)
            # 内层预算（用 M4 口径，因为 M4 是主追加方式）
            e1i = np.abs(y[tr] - o1); e4i = np.abs(y[tr] - o4)
            curve = []
            for b in BUDGETS:
                k = int(np.floor(b * len(tr)))
                s = np.zeros(len(tr), bool)
                if k:
                    s[np.argsort(-o1)[:k]] = True
                curve.append((float(b), float(np.where(s, e4i, e1i).mean())))
            bb, bbm = min(curve, key=lambda t: t[1])
            inn = pd.DataFrame({
                "sample_id": ids[tr], "group": groups[tr], "y": y[tr], "depth": depth[tr],
                "pred_M1": o1, "pred_M2": o2, "pred_M3": o3, "pred_M4": o4, "pred_M5": o5,
                "weight_M4": bw, "budget_M4": bb,
                "ncomp": f"{np.median(nc1):.0f}/{np.median(nc2):.0f}/{np.median(nc5):.0f}",
                "budget_curve": ";".join(f"{b}:{m:.5f}" for b, m in curve),
            })
            inn.to_csv(cache, index=False)
            print(f"    OOF: M1 {np.mean(np.abs(y[tr]-o1)):.4f}  M2 {np.mean(np.abs(y[tr]-o2)):.4f}  "
                  f"M4 {np.mean(np.abs(y[tr]-o4)):.4f}  M5 {np.mean(np.abs(y[tr]-o5)):.4f}  "
                  f"w={bw:.2f} p={bb:.0%}", flush=True)

        bw = float(inn["weight_M4"].iloc[0])
        bb = float(inn["budget_M4"].iloc[0])

        # ---------- 外层 ----------
        p1, c1 = fit_predict_outer(Xn[tr], y[tr], groups[tr], Xn[te])
        p2, c2 = fit_predict_outer(Xm[tr], y[tr], groups[tr], Xm[te])
        p5, c5 = fit_predict_outer((Xn[tr], Xm[tr]), y[tr], groups[tr], (Xn[te], Xm[te]),
                                   equalize_block=True)
        p3 = 0.5 * p1 + 0.5 * p2
        p4 = bw * p1 + (1 - bw) * p2
        preds = {"M1": p1, "M2": p2, "M3": p3, "M4": p4, "M5": p5}
        yt = y[te]
        base = {k: float(np.mean(np.abs(yt - v))) for k, v in preds.items()}
        print(f"    测试: " + "  ".join(f"{k}={v:.4f}" for k, v in base.items()), flush=True)

        # ---------- 策略（随机基线按每个 aug 单独算，关键修正③） ----------
        nt = len(te)
        e1 = np.abs(yt - p1)
        for pname, pfrac in (("fixed20", 0.20), ("fixed50", 0.50), ("adaptive", bb), ("all", 1.0)):
            k = int(np.floor(pfrac * nt))
            sel = np.zeros(nt, bool)
            if 0 < k:
                sel[np.argsort(-p1)[:k]] = True
            q = k / nt if nt else 0.0
            for aug in ("M2", "M3", "M4", "M5"):
                ea = np.abs(yt - preds[aug])
                mae = float(np.where(sel, ea, e1).mean())
                rnd = float((1 - q) * e1.mean() + q * ea.mean())      # 该模型自己的随机期望
                fold_rows.append({"dataset": name, "outer_fold": f, "policy": pname,
                                  "aug": aug, "budget": pfrac, "k": k, "n_test": nt,
                                  "mae": mae, "mae_random": rnd,
                                  "gain_vs_base": float(e1.mean() - mae)})
        # Oracle（20%）
        for aug in ("M2", "M3", "M4", "M5"):
            ea = np.abs(yt - preds[aug]); g = e1 - ea
            k20 = int(np.floor(0.20 * nt))
            sel = np.zeros(nt, bool)
            if k20:
                sel[np.argsort(-g)[:k20]] = True
            fold_rows.append({"dataset": name, "outer_fold": f, "policy": "oracle20",
                              "aug": aug, "budget": 0.20, "k": k20, "n_test": nt,
                              "mae": float(np.where(sel, ea, e1).mean()),
                              "mae_random": np.nan, "gain_vs_base": np.nan})

        for i, idx in enumerate(te):
            pred_recs.append({"dataset": name, "sample_id": ids[idx], "group": groups[idx],
                              "outer_fold": f, "y": float(yt[i]), "depth": float(depth[idx]),
                              "pred_M1": float(p1[i]), "pred_M2": float(p2[i]),
                              "pred_M3": float(p3[i]), "pred_M4": float(p4[i]),
                              "pred_M5": float(p5[i]),
                              "weight_M4": bw, "budget_adaptive": bb,
                              "ncomp": f"{c1}/{c2}/{c5}"})
        print(f"    (累计 {time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(pred_recs).to_csv(S / "predictions" / f"{name}_corrected.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(S / "results" / f"corrected_fusion_{name}.csv", index=False)
    print(f"[OK] {name} 完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="all")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    for d in ("scripts", "cache/inner_oof", "predictions", "results", "figures", "reports", "logs"):
        (S / d).mkdir(parents=True, exist_ok=True)

    names = ["neospectra", "icraf", "lucas"] if args.dataset == "all" else [args.dataset]
    for nm in names:
        run(nm, root, S)

    fc, pr = [], []
    for nm in names:
        f = S / "results" / f"corrected_fusion_{nm}.csv"
        p = S / "predictions" / f"{nm}_corrected.csv"
        if f.exists():
            fc.append(pd.read_csv(f))
        if p.exists():
            pr.append(pd.read_csv(p))
    if fc:
        pd.concat(fc, ignore_index=True).to_csv(
            S / "results" / "corrected_fusion_comparison.csv", index=False)
    if pr:
        pd.concat(pr, ignore_index=True).to_csv(
            S / "predictions" / "all_corrected_predictions.csv", index=False)
    print("\n[OK] 汇总完成")


if __name__ == "__main__":
    raise SystemExit(main())
