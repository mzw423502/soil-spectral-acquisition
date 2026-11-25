#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F03_gain_mechanism_final.py —— 最终机制统计（统一样本 + 国家固定效应）

按用户 §F03 / §一 的方法：
  · 固定**完整协变量样品集**（clay/sand/pH/depth/country 全部非缺失）
  · 同时分析 G_M（MIR 替换）与 G_F（M4 融合）
  · SOC 用 log1p + 二次项；深度用 log1p
  · **加入国家固定效应**
  · 按真实剖面聚类稳健标准误

模型阶梯：
  L1  G ~ log_soc
  L2  G ~ log_soc + log_soc^2
  L3  L2 + log_depth + clay + sand + pH + **C(country)**

用法：python3 F03_gain_mechanism_final.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"

    pred = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    d = pred[pred["dataset"] == "icraf"].copy().rename(columns={"sample_id": "id"})

    lab = pq.read_table(root / "data" / "icraf" / "ossl_soillab_v1.3.parquet",
                        columns=["id.layer_local_c", "clay.tot_usda.a334_w.pct",
                                 "sand.tot_usda.c60_w.pct", "ph.h2o_usda.a268_index"]).to_pandas()
    lab = lab.rename(columns={"id.layer_local_c": "id",
                              "clay.tot_usda.a334_w.pct": "clay",
                              "sand.tot_usda.c60_w.pct": "sand",
                              "ph.h2o_usda.a268_index": "pH"})
    site = pq.read_table(root / "data" / "icraf" / "ossl_soilsite_v1.3.parquet",
                         columns=["id.layer_local_c", "loc.country_src_txt"]).to_pandas()
    site = site.drop_duplicates("id.layer_local_c").rename(
        columns={"id.layer_local_c": "id", "loc.country_src_txt": "country"})

    d = d.merge(lab, on="id", how="left").merge(site, on="id", how="left")

    # 收益
    y = d["y"].to_numpy(float)
    pN = np.clip(d["pred_M1"].to_numpy(float), 0, None)
    pM = np.clip(d["pred_M2"].to_numpy(float), 0, None)
    pF = np.clip(d["pred_M4"].to_numpy(float), 0, None)
    d["G_M"] = np.abs(y - pN) - np.abs(y - pM)
    d["G_F"] = np.abs(y - pN) - np.abs(y - pF)

    # 固定完整协变量子集
    need = ["G_M", "G_F", "y", "depth", "clay", "sand", "pH", "country", "group"]
    full = d.dropna(subset=need).copy()
    full["log_soc"] = np.log1p(np.maximum(full["y"], 0))
    full["log_soc2"] = full["log_soc"] ** 2
    full["log_depth"] = np.log1p(np.maximum(full["depth"], 0))
    print(f"完整协变量样品 {len(full)}  剖面 {full['group'].nunique()}  "
          f"国家 {full['country'].nunique()}")
    print(f"（全库 {len(d)}，因黏粒/pH 缺失剔除 {len(d)-len(full)}）")

    # ---------- 负收益直接统计（同一子集） ----------
    neg = {}
    for k in ("G_M", "G_F"):
        v = full[k].to_numpy(float)
        neg[k] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                  "pos_pct": float((v > 1e-9).mean() * 100),
                  "neg_pct": float((v < -1e-9).mean() * 100),
                  "sd": float(np.std(v, ddof=1))}
    print(f"\n=== 负收益（统一子集 n={len(full)}）===")
    for k, v in neg.items():
        print(f"  {k}: 均值 {v['mean']:+.4f}  负收益 {v['neg_pct']:.1f}%  sd {v['sd']:.4f}")
    print(f"  融合使负收益下降 {neg['G_M']['neg_pct']-neg['G_F']['neg_pct']:+.1f} pp")

    # ---------- 回归阶梯 ----------
    import statsmodels.formula.api as smf
    out = {"n_full": int(len(full)), "n_profiles": int(full["group"].nunique()),
           "n_countries": int(full["country"].nunique()), "negative": neg}
    coef_table = {}
    for k in ("G_M", "G_F"):
        print(f"\n{'='*76}\n=== {k} 回归阶梯（国家固定效应 + 剖面聚类稳健）===")
        res = {}
        specs = {
            "L1_linear": f"{k} ~ log_soc",
            "L2_quadratic": f"{k} ~ log_soc + log_soc2",
            "L3_full": f"{k} ~ log_soc + log_soc2 + log_depth + clay + sand + pH + C(country)",
        }
        for tag, formula in specs.items():
            try:
                m = smf.ols(formula, full).fit(cov_type="cluster",
                                               cov_kwds={"groups": full["group"]})
                r = {"n": int(m.nobs), "r2": float(m.rsquared), "r2_adj": float(m.rsquared_adj),
                     "params": {kk: float(vv) for kk, vv in m.params.items()},
                     "pvalues": {kk: float(vv) for kk, vv in m.pvalues.items()},
                     "bse": {kk: float(vv) for kk, vv in m.bse.items()}}
                res[tag] = r
                if tag == "L3_full":
                    coef_table[k] = {v: {"coef": r["params"].get(v),
                                         "se": r["bse"].get(v),
                                         "p": r["pvalues"].get(v)}
                                     for v in ("log_soc", "log_soc2", "log_depth",
                                               "clay", "sand", "pH")}
                line = f"  {tag:<14} n={int(m.nobs):<5} R2={m.rsquared:.4f}"
                if tag != "L1_linear":
                    for v in ("log_soc", "log_soc2", "log_depth", "clay", "sand", "pH"):
                        if v in m.params.index:
                            star = ("***" if m.pvalues[v] < 0.001 else
                                    ("**" if m.pvalues[v] < 0.01 else
                                     ("*" if m.pvalues[v] < 0.05 else "")))
                            line += f"  {v}={m.params[v]:+.4f}{star}"
                print(line, flush=True)
            except Exception as e:
                res[tag] = {"error": str(e)[:120]}
                print(f"  {tag:<14} 失败: {str(e)[:80]}", flush=True)
        out[k] = res

    out["coef_L3"] = coef_table

    # ---------- 对照：不含国家固定效应 ----------
    print(f"\n=== 对照：L2b = L3 去掉国家固定效应 ===")
    for k in ("G_M", "G_F"):
        f2 = f"{k} ~ log_soc + log_soc2 + log_depth + clay + sand + pH"
        try:
            m = smf.ols(f2, full).fit(cov_type="cluster", cov_kwds={"groups": full["group"]})
            r2_no = float(m.rsquared)
            r2_with = out[k].get("L3_full", {}).get("r2", np.nan)
            print(f"  {k}: R2 无国家={r2_no:.4f}  有国家={r2_with:.4f}  "
                  f"国家贡献 ΔR2={r2_with-r2_no:+.4f}")
            out.setdefault("country_r2_gain", {})[k] = float(r2_with - r2_no)
        except Exception as e:
            print(f"  {k}: 失败 {str(e)[:70]}")

    # ---------- 关键系数表 ----------
    print(f"\n{'='*76}\n=== 最终系数表（L3，两种收益对照）===")
    print(f"  {'变量':<12} {'G_M 系数':>12} {'p':>10} {'G_F 系数':>12} {'p':>10}")
    for v in ("log_soc", "log_soc2", "log_depth", "clay", "sand", "pH"):
        a = coef_table.get("G_M", {}).get(v, {})
        b = coef_table.get("G_F", {}).get(v, {})
        print(f"  {v:<12} {a.get('coef', float('nan')):>+12.5f} {a.get('p', float('nan')):>10.2e} "
              f"{b.get('coef', float('nan')):>+12.5f} {b.get('p', float('nan')):>10.2e}")

    (S / "results" / "F03_gain_mechanism_final.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {S/'results'/'F03_gain_mechanism_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
