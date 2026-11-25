#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B02_nested_cost_policy.py —— 训练集内独立选择成本方案，外层验证

三个决策场景：
  S1 固定成本上限 → 选内部 MAE 最低的可行方案
  S2 固定精度容忍 → 选满足 MAE ≤ (1+δ)·E_full 的最低成本方案
  S3 无约束 → 选内部 MAE 最低方案

候选方案（不预先假定"按 MIR 替换选的预算对 M4 也最优"）：
  all_NIR        (r,            M1)
  all_MIR        (1,            M2)
  all_M4         (1+r,          M4)
  sel_p_M2       (r+p,          M2 替换)
  sel_p_M4       (r+p,          M4 融合)
  p ∈ {0.05, ..., 0.95}

决策只用**内层 OOF**（cache/inner_oof/），测试集只用于最后评价。

用法：python3 B02_nested_cost_policy.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COST_RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)
TOTAL_COST_CAPS = (0.3, 0.5, 0.7, 0.9, 1.0, 1.1)
TOLERANCES = (0.00, 0.02, 0.05, 0.10)
PS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def candidates(df, r):
    """在给定样品集（内层 OOF 或外层测试）上枚举全部候选方案"""
    y = df["y"].to_numpy(float)
    p1 = df["pred_M1"].to_numpy(float)
    p2 = df["pred_M2"].to_numpy(float)
    p4 = df["pred_M4"].to_numpy(float)
    n = len(y)
    e1 = np.abs(y - p1); e2 = np.abs(y - p2); e4 = np.abs(y - p4)
    rows = [{"strategy": "all_NIR", "model": "M1", "p": 0.0, "cost": r, "mae": float(e1.mean()), "k": 0},
            {"strategy": "all_MIR", "model": "M2", "p": 1.0, "cost": 1.0, "mae": float(e2.mean()), "k": n},
            {"strategy": "all_M4", "model": "M4", "p": 1.0, "cost": 1.0 + r, "mae": float(e4.mean()), "k": n}]
    for p in PS:
        k = int(np.floor(p * n))
        if k == 0:
            continue
        sel = np.zeros(n, bool)
        sel[np.argsort(-p1, kind="stable")[:k]] = True
        q = k / n
        for mdl, ea in (("M2", e2), ("M4", e4)):
            mae = float(np.where(sel, ea, e1).mean())
            rows.append({"strategy": f"sel_{mdl}", "model": mdl, "p": float(p),
                         "cost": r + q, "mae": mae, "k": k})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    oof_dir = S / "cache" / "inner_oof"
    pred = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")

    s1_rows, s2_rows = [], []
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n=== {name} ===", flush=True)
        d_test = pred[pred["dataset"] == name]
        folds = sorted(d_test["outer_fold"].unique())
        for f in folds:
            cf = oof_dir / f"inner_{name}_fold{f}.csv"
            if not cf.exists():
                print(f"  [skip] 缺 {cf.name}")
                continue
            inn = pd.read_csv(cf)
            dte = d_test[d_test["outer_fold"] == f]
            for r in COST_RATIOS:
                # ---------- S3 基准：全量 M4 的内层 MAE ----------
                ci = candidates(inn, r)
                e_full = float(ci.loc[ci["strategy"] == "all_M4", "mae"].iloc[0])

                # ---------- S1 固定成本上限 ----------
                for cap in TOTAL_COST_CAPS:
                    feas = ci[ci["cost"] <= cap + 1e-9]
                    if feas.empty:
                        s1_rows.append({"dataset": name, "outer_fold": f, "r": r, "cap": cap,
                                        "chosen": "infeasible", "inner_mae": np.nan,
                                        "inner_cost": np.nan, "test_mae": np.nan,
                                        "test_cost": np.nan, "n_test": len(dte)})
                        continue
                    b = feas.loc[feas["mae"].idxmin()]
                    # 外层执行
                    ct = candidates(dte, r)
                    match = ct[(ct["strategy"] == b["strategy"]) &
                               (np.abs(ct["p"] - b["p"]) < 1e-9)]
                    if match.empty:
                        match = ct[(ct["strategy"] == b["strategy"])].head(1)
                    t = match.iloc[0]
                    s1_rows.append({"dataset": name, "outer_fold": f, "r": r, "cap": cap,
                                    "chosen": f"{b['strategy']}@p={b['p']:.2f}",
                                    "inner_mae": float(b["mae"]), "inner_cost": float(b["cost"]),
                                    "test_mae": float(t["mae"]), "test_cost": float(t["cost"]),
                                    "actual_p": float(t["k"] / len(dte)),
                                    "n_test": len(dte)})

                # ---------- S2 固定精度容忍 ----------
                for tol in TOLERANCES:
                    thr = (1 + tol) * e_full
                    ok = ci[ci["mae"] <= thr + 1e-12]
                    if ok.empty:
                        s2_rows.append({"dataset": name, "outer_fold": f, "r": r, "tol": tol,
                                        "chosen": "infeasible", "inner_cost": np.nan,
                                        "test_mae": np.nan, "test_cost": np.nan,
                                        "saved_pct": np.nan, "n_test": len(dte)})
                        continue
                    b = ok.loc[ok["cost"].idxmin()]
                    ct = candidates(dte, r)
                    match = ct[(ct["strategy"] == b["strategy"]) &
                               (np.abs(ct["p"] - b["p"]) < 1e-9)]
                    if match.empty:
                        match = ct[(ct["strategy"] == b["strategy"])].head(1)
                    t = match.iloc[0]
                    full_cost = 1.0 + r
                    s2_rows.append({"dataset": name, "outer_fold": f, "r": r, "tol": tol,
                                    "chosen": f"{b['strategy']}@p={b['p']:.2f}",
                                    "inner_cost": float(b["cost"]), "inner_mae": float(b["mae"]),
                                    "e_full": e_full,
                                    "test_mae": float(t["mae"]), "test_cost": float(t["cost"]),
                                    "actual_p": float(t["k"] / len(dte)),
                                    "saved_pct": float((full_cost - t["cost"]) / full_cost * 100),
                                    "meets_tol": bool(t["mae"] <= (1 + tol) * float(
                                        candidates(dte, r).loc[
                                            lambda x: x["strategy"] == "all_M4", "mae"].iloc[0])),
                                    "n_test": len(dte)})

    s1 = pd.DataFrame(s1_rows)
    s2 = pd.DataFrame(s2_rows)
    s1.to_csv(S / "results" / "nested_cost_policy_results.csv", index=False)
    s2.to_csv(S / "results" / "cost_tolerance_results.csv", index=False)

    print(f"\n{'='*70}\nS1 固定成本上限（样品加权）")
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n{name}:")
        for r in (0.10, 0.30):
            d = s1[(s1["dataset"] == name) & (s1["r"] == r) & (s1["chosen"] != "infeasible")]
            if d.empty:
                continue
            line = []
            for cap in TOTAL_COST_CAPS:
                dd = d[d["cap"] == cap]
                if dd.empty:
                    continue
                w = dd["n_test"].to_numpy(float)
                tm = float(np.average(dd["test_mae"], weights=w))
                tc = float(np.average(dd["test_cost"], weights=w))
                mode = dd["chosen"].str.split("@").str[0].mode()
                line.append(f"cap{cap}:{mode.iloc[0] if len(mode) else '?'}({tc:.2f}/{tm:.3f})")
            print(f"  r={r}: " + "  ".join(line))

    print(f"\n{'='*70}\nS2 固定精度容忍（相对全量 M4，样品加权）")
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n{name}:")
        for r in (0.10,):
            d = s2[(s2["dataset"] == name) & (s2["r"] == r) & (s2["chosen"] != "infeasible")]
            if d.empty:
                continue
            for tol in TOLERANCES:
                dd = d[d["tol"] == tol]
                if dd.empty:
                    continue
                w = dd["n_test"].to_numpy(float)
                tc = float(np.average(dd["test_cost"], weights=w))
                tm = float(np.average(dd["test_mae"], weights=w))
                sv = float(np.average(dd["saved_pct"], weights=w))
                rate = float(dd["meets_tol"].mean() * 100)
                print(f"  δ={tol:.0%}: 成本 {tc:.2f} (省 {sv:.0f}%)  MAE {tm:.4f}  外层达标率 {rate:.0f}%")

    print(f"\n[OK] nested_cost_policy_results.csv / cost_tolerance_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
