#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B04_depth_gain_policy.py —— SOC + 深度联合的收益预测策略

四组策略（ICRAF 主分析，NeoSpectra 无深度故跳过深度项）：
  D1  预测 SOC 直接排序
  D2  采样深度直接排序
  D3  收益单变量回归：G_F ~ log(1+SOC_hat)
  D4  收益双变量回归：G_F ~ log(1+SOC_hat) + log(1+Depth)

收益标签 G_F = |y - yN| - |y - yF|（M4 融合收益），
在**内层 OOF** 上生成；Ridge(alpha) 由内层再分折选。

外层测试只用于评价，不参与任何拟合。
只有 D4 相对 D3 有稳定改善，才认为深度提供了独立决策增量。

用法：python3 B04_depth_gain_policy.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALPHAS = (0.01, 0.1, 1, 10, 100)
BUDGETS = (0.20, 0.50)


def ridge_fit(X, y, groups):
    """内层选 alpha，返回最优模型"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    k = min(3, len(set(groups)))
    if k < 2 or len(y) < 30:
        a = 1.0
    else:
        best = (np.inf, ALPHAS[0])
        for al in ALPHAS:
            errs = []
            for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                m = Ridge(alpha=al).fit(sc.transform(X[tr]), y[tr])
                errs.append(np.mean((y[va] - m.predict(sc.transform(X[va]))) ** 2))
            r = float(np.mean(errs))
            if r < best[0]:
                best = (r, al)
        a = best[1]
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=a).fit(sc.transform(X), y)
    return sc, m, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    pred = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    oof_dir = S / "cache" / "inner_oof"

    rows = []
    for name in ("icraf", "neospectra"):
        d = pred[pred["dataset"] == name]
        print(f"\n=== {name} ===", flush=True)
        for f, g in d.groupby("outer_fold"):
            cf = oof_dir / f"inner_{name}_fold{f}.csv"
            if not cf.exists():
                continue
            inn = pd.read_csv(cf)
            yt = g["y"].to_numpy(float)
            p1 = np.clip(g["pred_M1"].to_numpy(float), 0, None)
            p4 = np.clip(g["pred_M4"].to_numpy(float), 0, None)
            dep = g["depth"].to_numpy(float)
            e1 = np.abs(yt - p1); e4 = np.abs(yt - p4)

            yi = inn["y"].to_numpy(float)
            pi1 = np.clip(inn["pred_M1"].to_numpy(float), 0, None)
            pi4 = np.clip(inn["pred_M4"].to_numpy(float), 0, None)
            di = inn["depth"].to_numpy(float)
            G_i = np.abs(yi - pi1) - np.abs(yi - pi4)
            gi = inn["group"].astype(str).to_numpy()

            ok = np.isfinite(G_i) & np.isfinite(di)
            if ok.sum() < 50:
                print(f"  fold {f}: 内层有效 {ok.sum()} 太少，跳过")
                continue

            # 构造特征
            Li = np.log1p(np.maximum(pi1, 0)); Di = np.log1p(np.maximum(di, 0))
            Lt = np.log1p(np.maximum(p1, 0)); Dt = np.log1p(np.maximum(dep, 0))

            X3 = Li[ok].reshape(-1, 1)
            X4 = np.column_stack([Li[ok], Di[ok]])
            sc3, m3, a3 = ridge_fit(X3, G_i[ok], gi[ok])
            sc4, m4, a4 = ridge_fit(X4, G_i[ok], gi[ok])

            scores = {
                "D1_predSOC": p1,
                "D2_depth": np.where(np.isfinite(dep), -dep, np.nan),  # 浅层优先
                "D3_soc_reg": m3.predict(sc3.transform(Lt.reshape(-1, 1))),
                "D4_soc_depth": m4.predict(sc4.transform(np.column_stack([Lt, Dt]))),
            }
            n = len(yt)
            for pol, sc in scores.items():
                sc = np.asarray(sc, float)
                if not np.isfinite(sc).all():
                    sc = np.where(np.isfinite(sc), sc, -np.inf)
                for b in BUDGETS:
                    k = int(np.floor(b * n))
                    sel = np.zeros(n, bool)
                    if k:
                        sel[np.argsort(-sc, kind="stable")[:k]] = True
                    mae = float(np.where(sel, e4, e1).mean())
                    q = k / n
                    rnd = float((1 - q) * e1.mean() + q * e4.mean())
                    # Oracle
                    orc_mask = np.zeros(n, bool)
                    if k:
                        orc_mask[np.argsort(-(e1 - e4), kind="stable")[:k]] = True
                    orc = float(np.where(orc_mask, e4, e1).mean())
                    rows.append({"dataset": name, "outer_fold": f, "policy": pol,
                                 "budget": b, "n_test": n, "k": k, "mae": mae,
                                 "mae_random": rnd, "mae_oracle": orc,
                                 "alpha_d3": a3, "alpha_d4": a4})
            print(f"  fold {f}: alpha D3={a3} D4={a4}  有效内层 {ok.sum()}", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(S / "results" / "depth_policy_results.csv", index=False)

    print(f"\n{'='*76}\n收益预测策略（aug=M4，样品加权）")
    for name in ("icraf", "neospectra"):
        d = r[r["dataset"] == name]
        if d.empty:
            continue
        print(f"\n{name}:")
        print(f"  {'策略':<16} {'预算':>6} {'MAE':>9} {'随机':>9} {'vs随机':>8} {'捕获率η':>9}")
        for pol in ("D1_predSOC", "D2_depth", "D3_soc_reg", "D4_soc_depth"):
            for b in BUDGETS:
                dd = d[(d["policy"] == pol) & (d["budget"] == b)]
                if dd.empty:
                    continue
                w = dd["n_test"].to_numpy(float)
                mae = float(np.average(dd["mae"], weights=w))
                rnd = float(np.average(dd["mae_random"], weights=w))
                orc = float(np.average(dd["mae_oracle"], weights=w))
                eta = (rnd - mae) / (rnd - orc) * 100 if rnd > orc else np.nan
                print(f"  {pol:<16} {b:>6.0%} {mae:>9.4f} {rnd:>9.4f} "
                      f"{(rnd-mae)/rnd*100:>+7.1f}% {eta:>8.1f}%")

    # D4 vs D3
    print(f"\n=== D4(含深度) vs D3(仅SOC) ===")
    summary = {}
    for name in ("icraf", "neospectra"):
        d = r[r["dataset"] == name]
        if d.empty:
            continue
        for b in BUDGETS:
            d3 = d[(d["policy"] == "D3_soc_reg") & (d["budget"] == b)]
            d4 = d[(d["policy"] == "D4_soc_depth") & (d["budget"] == b)]
            if d3.empty or d4.empty:
                continue
            w = d3["n_test"].to_numpy(float)
            m3 = float(np.average(d3["mae"], weights=w))
            m4 = float(np.average(d4["mae"], weights=w))
            d1 = d[(d["policy"] == "D1_predSOC") & (d["budget"] == b)]
            m1 = float(np.average(d1["mae"], weights=w)) if not d1.empty else np.nan
            print(f"  {name:<12} 预算{b:.0%}: D1={m1:.4f}  D3={m3:.4f}  D4={m4:.4f}  "
                  f"D4−D3={m4-m3:+.4f} ({(m3-m4)/m3*100:+.1f}%)")
            summary[f"{name}_{b}"] = {"D1": m1, "D3": m3, "D4": m4, "D4_minus_D3": m4 - m3}

    print(f"\n结论提示：若 D4−D3 ≥ 0（即含深度没有改善），推荐用更简单的 D1 或 D3。")
    (S / "results" / "depth_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {S/'results'/'depth_policy_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
