#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F02_lucas_loco_strict.py —— LUCAS 整国留出（复用 B00 严格交叉拟合）

与旧 B05 的差别：内层 OOF 不再自实现，直接复用 B00.oof_inner
（每折独立预处理、独立选成分、负预测截断）。

六个国家：Italy / Finland / Poland / Portugal / Austria / Czech Republic

**重要区分**（用户 §F02）：
  · 目标国的 NIR/MIR 真实误差差距是**事后解释**，不能当作可部署规则
  · 实际预算必须由**训练集**决定（内层 OOF 曲线）

输出：
  predictions/lucas_loco_predictions.csv   逐样品
  results/F02_lucas_loco.json              每国汇总 + 训练集定预算

用法：python3 F02_lucas_loco_strict.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
BUDGETS = np.round(np.arange(0, 1.0001, 0.05), 2)
MIN_N = 30


def load_b00(root):
    p = root / "stage2_batch2" / "scripts" / "B00_corrected_nested_fusion.py"
    spec = importlib.util.spec_from_file_location("b00", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    F = root / "stage2_final"
    b00 = load_b00(root)

    d = np.load(F / "manifests" / "data_lucas.npz", allow_pickle=True)
    Xn = d["X_nir"].astype(np.float64)
    Xm = d["X_mir"].astype(np.float64)
    y = d["y"].astype(np.float64)
    ids = d["ids"].astype(str)

    site = pq.read_table(root / "data" / "lucas" / "ossl_soilsite_v1.3.parquet",
                         columns=["id.layer_local_c", "loc.country_src_txt"]).to_pandas()
    site = site.drop_duplicates("id.layer_local_c").set_index("id.layer_local_c")
    country = site.reindex(ids)["loc.country_src_txt"].astype(str).to_numpy()

    vc = pd.Series(country).value_counts()
    targets = [c for c, n in vc.items() if n >= MIN_N]
    print(f"LUCAS n={len(y)}  目标国 {len(targets)}: {targets}", flush=True)

    rows, recs = [], []
    for c in targets:
        te = np.where(country == c)[0]
        tr = np.where(country != c)[0]
        print(f"\n--- LOCO {c}: train {len(tr)} / test {len(te)} ---", flush=True)
        gtr = pd.Series(country[tr]).astype(str).to_numpy()

        # 严格交叉拟合的内层 OOF（复用 B00）
        o1, nc1 = b00.oof_inner(Xn[tr], y[tr], gtr)
        o2, nc2 = b00.oof_inner(Xm[tr], y[tr], gtr)
        bw, bwm = 0.5, np.inf
        for w in WEIGHTS:
            mae = float(np.mean(np.abs(y[tr] - (w * o1 + (1 - w) * o2))))
            if mae < bwm:
                bwm, bw = mae, float(w)
        o4 = bw * o1 + (1 - bw) * o2

        # 训练集定预算（可部署规则）
        e1i, e4i = np.abs(y[tr] - o1), np.abs(y[tr] - o4)
        curve = []
        for b in BUDGETS:
            k = int(np.floor(b * len(tr)))
            s = np.zeros(len(tr), bool)
            if k:
                s[np.argsort(-o1, kind="stable")[:k]] = True
            curve.append((float(b), float(np.where(s, e4i, e1i).mean())))
        bb = min(curve, key=lambda t: t[1])[0]
        tau = float(np.quantile(o1, 1 - 0.20))     # 20% 分位阈值

        # 外层
        p1, _ = b00.fit_predict_outer(Xn[tr], y[tr], gtr, Xn[te])
        p2, _ = b00.fit_predict_outer(Xm[tr], y[tr], gtr, Xm[te])
        p4 = bw * p1 + (1 - bw) * p2
        yt = y[te]; nt = len(te)
        e1 = np.abs(yt - p1); e2 = np.abs(yt - p2); e4 = np.abs(yt - p4)

        def run(mask):
            k = int(mask.sum()); q = k / nt
            return (float(np.where(mask, e4, e1).mean()),
                    float((1 - q) * e1.mean() + q * e4.mean()), k, q)

        rng = np.random.default_rng(42)
        kk = int(np.floor(0.20 * nt))
        m_hi = np.zeros(nt, bool)
        if kk:
            m_hi[np.argsort(-p1, kind="stable")[:kk]] = True
        a_hi, r_hi, k_hi, q_hi = run(m_hi)
        rr = []
        for _ in range(200):
            m = np.zeros(nt, bool)
            if kk:
                m[rng.choice(nt, kk, replace=False)] = True
            rr.append(float(np.where(m, e4, e1).mean()))
        r_rand = float(np.mean(rr))

        ka = int(np.floor(bb * nt))
        m_ad = np.zeros(nt, bool)
        if ka:
            m_ad[np.argsort(-p1, kind="stable")[:ka]] = True
        a_ad, _, k_ad, q_ad = run(m_ad)

        m_t = p1 > tau
        a_t, r_t, k_t, q_t = run(m_t)

        rec = {"country": c, "n": nt, "n_train": len(tr),
               "w_M4": bw, "budget_train": bb, "tau20": tau,
               "mae_NIR": float(e1.mean()), "mae_MIR": float(e2.mean()),
               "mae_M4": float(e4.mean()),
               "mae_sel20": a_hi, "mae_rand20_analytic": r_hi,
               "mae_rand200": r_rand, "k_sel20": k_hi,
               "mae_adaptive": a_ad, "k_adaptive": k_ad, "p_adaptive": q_ad,
               "mae_thresh": a_t, "k_thresh": k_t, "p_thresh": q_t,
               "cost_all_M4": 1.10, "cost_adaptive": 0.10 + q_ad,
               "cost_sel20": 0.10 + q_hi,
               # 事后解释用（不可部署）
               "gap_NIR_minus_MIR": float(e1.mean() - e2.mean())}
        rows.append(rec)
        print(f"    NIR {rec['mae_NIR']:.3f} MIR {rec['mae_MIR']:.3f} M4 {rec['mae_M4']:.3f} | "
              f"sel20 {a_hi:.3f} (随机 {r_rand:.3f}) | 自适应 {a_ad:.3f}(p={bb:.0%}) | "
              f"阈值 {a_t:.3f}({q_t:.0%})", flush=True)

        for i in range(nt):
            recs.append({"country": c, "sample_id": ids[te[i]], "y": float(yt[i]),
                         "pred_M1": float(p1[i]), "pred_M2": float(p2[i]),
                         "pred_M4": float(p4[i]), "w_M4": bw, "budget_train": bb})

    t = pd.DataFrame(rows)
    pd.DataFrame(recs).to_csv(S / "predictions" / "lucas_loco_predictions.csv", index=False)

    print(f"\n{'='*96}\n整国留出（严格交叉拟合）")
    cols = ["mae_NIR", "mae_MIR", "mae_M4", "mae_sel20", "mae_rand200", "mae_adaptive", "mae_thresh"]
    print(f"{'国家':<16} " + " ".join(f"{c.replace('mae_',''):>10}" for c in cols) + "   budget")
    for _, r in t.iterrows():
        print(f"{r['country']:<16} " + " ".join(f"{r[c]:>10.3f}" for c in cols)
              + f"   {r['budget_train']:.0%}")
    print(f"\n{'样品加权':<16} " + " ".join(
        f"{np.average(t[c], weights=t['n']):>10.3f}" for c in cols))
    print(f"{'国家等权':<16} " + " ".join(f"{t[c].mean():>10.3f}" for c in cols))

    w = t["n"].to_numpy(float)
    print(f"\n关键对比（样品加权）：")
    for lab, a, bb_ in (("sel20 vs 随机", "mae_sel20", "mae_rand200"),
                        ("sel20 vs 全M4", "mae_sel20", "mae_M4"),
                        ("自适应 vs 全M4", "mae_adaptive", "mae_M4"),
                        ("自适应 vs 随机", "mae_adaptive", "mae_rand200")):
        va = float(np.average(t[a], weights=w)); vb = float(np.average(t[bb_], weights=w))
        print(f"  {lab:<18} {va:.4f} vs {vb:.4f}  ({(vb-va)/vb*100:+.1f}%)")
    print(f"  sel20 优于随机的国家: {(t['mae_sel20'] < t['mae_rand200']).sum()}/{len(t)}")
    print(f"  自适应优于 20% 的国家: {(t['mae_adaptive'] < t['mae_sel20']).sum()}/{len(t)}")

    out = {"countries": rows,
           "sample_weighted": {c: float(np.average(t[c], weights=w)) for c in cols},
           "country_equal": {c: float(t[c].mean()) for c in cols},
           "note": "gap_NIR_minus_MIR 为事后解释，不作为可部署预算规则；"
                   "budget_train 才是训练集决定的预算"}
    (S / "results" / "F02_lucas_loco.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {S/'results'/'F02_lucas_loco.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
