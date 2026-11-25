#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B01_cost_frontier.py —— 成本—精度曲线与 Pareto 前沿

成本定义（MIR 单次 = 1，NIR = r，追加比例 = p）：
  全部 NIR          C = r
  全部 MIR          C = 1
  全部 M4 融合       C = 1 + r
  选择性追加(替换)   C = r + p
  选择性追加(融合)   C = r + p

跨折汇总按**样品数加权**（不平均每折 MAE）。
随机基线对每个追加模型各自计算（不共用 M2）。

用法：
  python3 B01_cost_frontier.py --root <manuscript_study 根> [--pred-source corrected|batch1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COST_RATIOS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)
BUDGETS = np.round(np.arange(0, 1.0001, 0.05), 2)


def weighted_mae(df, col_err):
    """按 n_test 加权，返回 (加权MAE, 总成本列)"""
    w = df["n_test"].to_numpy(float)
    v = df[col_err].to_numpy(float)
    return float(np.sum(w * v) / np.sum(w))


def load_preds(S, source):
    if source == "corrected":
        p = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    else:
        p = pd.read_csv(S.parent / "stage2_final" / "predictions" / "all_model_predictions.csv")
        p = p.rename(columns={"budget_selected": "budget_adaptive"})
    return p


def per_fold_records(p, name):
    """构造每折 × 每预算 × 每策略的记录"""
    recs = []
    d = p[p["dataset"] == name]
    for f, g in d.groupby("outer_fold"):
        y = g["y"].to_numpy(float)
        p1 = g["pred_M1"].to_numpy(float)
        p2 = g["pred_M2"].to_numpy(float)
        p4 = g["pred_M4"].to_numpy(float)
        n = len(y)
        e1 = np.abs(y - p1)
        e2 = np.abs(y - p2)
        e4 = np.abs(y - p4)
        # 固定端点
        recs.append({"dataset": name, "outer_fold": f, "n_test": n, "budget": 0.0,
                     "strategy": "all_NIR", "model": "M1", "k": 0,
                     "mae": float(e1.mean()), "mae_random": np.nan})
        recs.append({"dataset": name, "outer_fold": f, "n_test": n, "budget": 1.0,
                     "strategy": "all_MIR", "model": "M2", "k": n,
                     "mae": float(e2.mean()), "mae_random": np.nan})
        recs.append({"dataset": name, "outer_fold": f, "n_test": n, "budget": 1.0,
                     "strategy": "all_M4", "model": "M4", "k": n,
                     "mae": float(e4.mean()), "mae_random": np.nan})
        for pp in BUDGETS:
            k = int(np.floor(pp * n))
            if k == 0 or k == n:
                continue
            order = np.argsort(-p1, kind="stable")
            sel = np.zeros(n, bool)
            sel[order[:k]] = True
            q = k / n
            for model, ea in (("M2", e2), ("M4", e4)):
                mae = float(np.where(sel, ea, e1).mean())
                rnd = float((1 - q) * e1.mean() + q * ea.mean())
                recs.append({"dataset": name, "outer_fold": f, "n_test": n, "budget": float(pp),
                             "strategy": f"sel_{model}", "model": model, "k": k,
                             "mae": mae, "mae_random": rnd})
    return pd.DataFrame(recs)


def add_cost(df, r):
    """成本 = r + k/n（选择类）；端点单独处理"""
    d = df.copy()
    frac = d["k"] / d["n_test"]
    cost = r + frac
    cost = np.where(d["strategy"] == "all_NIR", r, cost)
    cost = np.where(d["strategy"] == "all_MIR", 1.0, cost)
    cost = np.where(d["strategy"] == "all_M4", 1.0 + r, cost)
    d["cost"] = cost
    return d


def pareto(df):
    """返回非支配点（成本更低且 MAE 更低）"""
    d = df.sort_values(["cost", "mae"]).reset_index(drop=True)
    keep = []
    best = np.inf
    for i, row in d.iterrows():
        if row["mae"] < best - 1e-12:
            keep.append(i)
            best = row["mae"]
    return d.loc[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pred-source", default="corrected", choices=["corrected", "batch1"])
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    (S / "results").mkdir(parents=True, exist_ok=True)
    (S / "figures").mkdir(parents=True, exist_ok=True)

    p = load_preds(S, args.pred_source)
    print(f"预测来源 {args.pred_source}: {len(p)} 行")

    all_rows, front_rows = [], []
    for name in ("neospectra", "icraf", "lucas"):
        fr = per_fold_records(p, name)
        for r in COST_RATIOS:
            d = add_cost(fr, r)
            # 按策略汇总（样品加权）—— 必须保留 budget，否则 21 档被压成 1 点
            agg = []
            for (strat, model, bud), g in d.groupby(["strategy", "model", "budget"]):
                row = {"dataset": name, "cost_ratio": r, "strategy": strat, "model": model,
                       "budget": float(bud),
                       "n_total": int(g["n_test"].sum()),
                       "cost": float(np.average(g["cost"], weights=g["n_test"])),
                       "mae": weighted_mae(g, "mae")}
                if g["mae_random"].notna().any():
                    gg = g[g["mae_random"].notna()]
                    row["mae_random"] = float(np.average(gg["mae_random"], weights=gg["n_test"]))
                else:
                    row["mae_random"] = np.nan
                agg.append(row)
            agg = pd.DataFrame(agg)
            all_rows.append(agg)
            pf = pareto(agg)
            pf["cost_ratio"] = r
            front_rows.append(pf)
            print(f"  {name} r={r}: Pareto {len(pf)} 点  "
                  f"最优点 {pf.iloc[-1]['strategy']}({pf.iloc[-1]['model'] if pf.iloc[-1]['model'] else ''}) "
                  f"cost={pf.iloc[-1]['cost']:.2f} MAE={pf.iloc[-1]['mae']:.4f}")

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(S / "results" / "cost_efficiency_results.csv", index=False)
    fr = pd.concat(front_rows, ignore_index=True)
    fr.to_csv(S / "results" / "cost_pareto_front.csv", index=False)

    # ---------- 关键问答：给定预算，最优是什么 ----------
    print(f"\n=== 给定总成本上限时的最优策略（样品加权）===")
    caps = (0.3, 0.5, 0.7, 0.9, 1.0, 1.1)
    qa = []
    for name in ("neospectra", "icraf", "lucas"):
        for r in (0.10, 0.30):
            d = res[(res["dataset"] == name) & (res["cost_ratio"] == r)]
            for cap in caps:
                feas = d[d["cost"] <= cap + 1e-9]
                if feas.empty:
                    qa.append({"dataset": name, "r": r, "cap": cap, "strategy": "infeasible",
                               "cost": np.nan, "mae": np.nan})
                    continue
                best = feas.loc[feas["mae"].idxmin()]
                qa.append({"dataset": name, "r": r, "cap": cap,
                           "strategy": best["strategy"], "model": best["model"],
                           "cost": float(best["cost"]), "mae": float(best["mae"])})
    qa = pd.DataFrame(qa)
    qa.to_csv(S / "results" / "cost_budget_optimal.csv", index=False)
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n{name}:")
        s = qa[qa["dataset"] == name]
        for r in (0.10, 0.30):
            ss = s[s["r"] == r]
            line = "  ".join(f"cap{t['cap']}:{t['strategy']}({t['cost']:.2f}/{t['mae']:.3f})"
                             for _, t in ss.iterrows() if t["strategy"] != "infeasible")
            print(f"  r={r}: {line}")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.grid": True,
                             "grid.alpha": 0.3, "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey="row")
        for j, name in enumerate(("neospectra", "icraf", "lucas")):
            for i, r in enumerate((0.10, 0.30)):
                ax = axes[i, j]
                d = res[(res["dataset"] == name) & (res["cost_ratio"] == r)]
                for strat, marker, col, lab in (
                        ("all_NIR", "o", "#999999", "all NIR"),
                        ("all_MIR", "s", "#4c72b0", "all MIR"),
                        ("all_M4", "D", "#55a868", "all M4 (full fusion)")):
                    t = d[d["strategy"] == strat]
                    if len(t):
                        ax.scatter(t["cost"], t["mae"], marker=marker, s=60, color=col,
                                   zorder=5, label=lab if (i == 0 and j == 0) else None)
                for model, col, lab in (("M2", "#dd8452", "select→MIR"),
                                        ("M4", "#c44e52", "select→M4")):
                    t = d[(d["strategy"] == f"sel_{model}")].sort_values("cost")
                    ax.plot(t["cost"], t["mae"], "-", lw=1.4, color=col,
                            label=lab if (i == 0 and j == 0) else None)
                    tr = t[t["mae_random"].notna()]
                    ax.plot(tr["cost"], tr["mae_random"], ":", lw=1.0, color=col, alpha=0.6)
                ax.set_title(f"{name}  r={r}", fontsize=9)
                ax.set_xlabel("relative measurement cost")
                if j == 0:
                    ax.set_ylabel("MAE (SOC %)")
        axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")
        fig.suptitle("Cost–accuracy trade-offs (rows: NIR/MIR cost ratio)", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(S / "figures" / f"Figure_cost_accuracy.{ext}")
        plt.close(fig)
        print(f"\n[OK] 图 -> {S/'figures'/'Figure_cost_accuracy.png'}")
    except Exception as e:
        print(f"[WARN] 绘图失败: {e}")

    print(f"\n[OK] cost_efficiency_results.csv / cost_pareto_front.csv / cost_budget_optimal.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
