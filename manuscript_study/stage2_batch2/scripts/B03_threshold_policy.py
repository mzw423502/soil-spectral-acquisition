#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B03_threshold_policy.py —— 固定绝对浓度阈值 + 训练集分位阈值

逐样品触发规则：ŜOC_NIR > τ 则追加（用 M4 融合），否则保留 NIR。

两类阈值：
  F1–F6  固定 0.5/1.0/1.5/2.0/3.0/5.0 %
  Q10–Q50 训练集内层 OOF 的 (1−p) 分位（p = 10%–50%），锁定后用于测试

关键：固定阈值可能让实际追加比例远高于目标，这是**应用边界**，必须如实报告。

用法：python3 B03_threshold_policy.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIXED_TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
TARGET_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)


def evaluate(y, p1, p4, mask, base_mae):
    """给定触发 mask，算 MAE / 成本 / 同预算随机"""
    n = len(y)
    e1 = np.abs(y - p1); e4 = np.abs(y - p4)
    k = int(mask.sum())
    q = k / n if n else 0.0
    mae = float(np.where(mask, e4, e1).mean())
    rnd = float((1 - q) * e1.mean() + q * e4.mean())
    # 随机（解析期望）同预算
    return mae, rnd, k, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    pred = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    oof_dir = S / "cache" / "inner_oof"

    rows = []
    for name in ("neospectra", "icraf", "lucas"):
        d = pred[pred["dataset"] == name]
        print(f"\n=== {name} ===", flush=True)
        for f, g in d.groupby("outer_fold"):
            y = g["y"].to_numpy(float)
            p1 = g["pred_M1"].to_numpy(float)
            p4 = g["pred_M4"].to_numpy(float)
            n = len(y)
            e1 = np.abs(y - p1)
            base = float(e1.mean())

            # ---- 固定阈值 ----
            for tau in FIXED_TAUS:
                mask = p1 > tau
                mae, rnd, k, q = evaluate(y, p1, p4, mask, base)
                rows.append({"dataset": name, "outer_fold": f, "kind": "fixed",
                             "threshold": tau, "target_rate": np.nan,
                             "n_test": n, "n_trigger": k, "trigger_rate": q,
                             "mae": mae, "mae_random": rnd, "mae_base": base,
                             "budget_dev": np.nan})

            # ---- 训练集分位阈值 ----
            cf = oof_dir / f"inner_{name}_fold{f}.csv"
            if not cf.exists():
                continue
            inn = pd.read_csv(cf)
            p1_in = inn["pred_M1"].to_numpy(float)
            for pr in TARGET_RATES:
                tau = float(np.quantile(p1_in, 1 - pr))
                mask = p1 > tau
                mae, rnd, k, q = evaluate(y, p1, p4, mask, base)
                rows.append({"dataset": name, "outer_fold": f, "kind": "quantile",
                             "threshold": tau, "target_rate": pr,
                             "n_test": n, "n_trigger": k, "trigger_rate": q,
                             "mae": mae, "mae_random": rnd, "mae_base": base,
                             "budget_dev": q - pr})

    res = pd.DataFrame(rows)
    res.to_csv(S / "results" / "threshold_results.csv", index=False)

    # ---------- 汇总 ----------
    print(f"\n{'='*76}\n固定阈值（样品加权）")
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n{name}:")
        print(f"  {'τ':>6} {'触发率':>8} {'MAE':>8} {'随机':>8} {'vs随机':>8} {'vs不追加':>9}")
        dst = res[(res["dataset"] == name) & (res["kind"] == "fixed")]
        for tau in FIXED_TAUS:
            dd = dst[dst["threshold"] == tau]
            if dd.empty:
                continue
            w = dd["n_test"].to_numpy(float)
            tr = float(np.average(dd["trigger_rate"], weights=w))
            mae = float(np.average(dd["mae"], weights=w))
            rnd = float(np.average(dd["mae_random"], weights=w))
            base = float(np.average(dd["mae_base"], weights=w))
            print(f"  {tau:>6.1f} {tr:>8.1%} {mae:>8.4f} {rnd:>8.4f} "
                  f"{(rnd-mae)/rnd*100:>+7.1f}% {(base-mae)/base*100:>+8.1f}%")

    print(f"\n{'='*76}\n训练集分位阈值（目标率 → 实际触发率）")
    for name in ("neospectra", "icraf", "lucas"):
        print(f"\n{name}:")
        print(f"  {'目标':>6} {'实际':>8} {'偏差':>8} {'MAE':>8} {'随机':>8} {'vs随机':>8}")
        dst = res[(res["dataset"] == name) & (res["kind"] == "quantile")]
        for pr in TARGET_RATES:
            dd = dst[np.isclose(dst["target_rate"], pr)]
            if dd.empty:
                continue
            w = dd["n_test"].to_numpy(float)
            tr = float(np.average(dd["trigger_rate"], weights=w))
            dev = float(np.average(dd["budget_dev"], weights=w))
            mae = float(np.average(dd["mae"], weights=w))
            rnd = float(np.average(dd["mae_random"], weights=w))
            print(f"  {pr:>6.0%} {tr:>8.1%} {dev:>+8.1%} {mae:>8.4f} {rnd:>8.4f} "
                  f"{(rnd-mae)/rnd*100:>+7.1f}%")

    # 逐国触发率波动（ICRAF）—— country 从 sample_manifest 补
    print(f"\n=== ICRAF 固定阈值的国家间触发率波动 ===")
    ic = pred[pred["dataset"] == "icraf"].copy()
    man_p = root / "stage2_final" / "manifests" / "sample_manifest.csv"
    if man_p.exists() and "country" not in ic.columns:
        man = pd.read_csv(man_p)
        man = man[man["dataset"] == "icraf"][["sample_id", "country"]]
        ic = ic.merge(man, on="sample_id", how="left")
    if "country" in ic.columns:
        for tau in (1.0, 2.0, 3.0):
            rates = []
            for c, g in ic.groupby("country"):
                if len(g) < 30:
                    continue
                rates.append(float((g["pred_M1"] > tau).mean()))
            if rates:
                print(f"  τ={tau}: {len(rates)} 国  触发率 min={min(rates):.0%} "
                      f"median={np.median(rates):.0%} max={max(rates):.0%}")
    else:
        print("  （缺 country 列，跳过）")

    print(f"\n[OK] {S/'results'/'threshold_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
