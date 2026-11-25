#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B07_spectral_ablation.py —— 修正后的 MIR 波段消融（完整 ICRAF 3,698 样品）

处理顺序（用户 §3.4）：
  原始连续 MIR → SNV+SG1 → 遮蔽波段（含 +/-EDGE 边缘保护区） → 内层选成分 → 外层预测

遮蔽区（cm-1）：
  A1 2800-3000   C-H
  A3  950-1200   有机-矿质重叠
  A4 2300-2500   碳酸盐候选
  C1 与 A3 等宽的随机连续区 x N_RAND
  C2 与 A4 等宽的随机连续区 x N_RAND

统计：MAE/RMSE/R2、相对 A0 的变化、剖面级配对 Bootstrap、随机对照经验排名、SOC 分层。

用法：python3 B07_spectral_ablation.py --root <manuscript_study 根> [--rand 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NCOMP_CANDS = (2, 4, 8, 12, 16, 20)
N_INNER = 3
EDGE = 5
BANDS = {"A1_CH_2800_3000": (2800, 3000),
         "A3_ClayOv_950_1200": (950, 1200),
         "A4_Carb_2300_2500": (2300, 2500)}
SOC_BANDS = [("<0.5", 0, 0.5), ("0.5-1", 0.5, 1), ("1-2", 1, 2), ("2-5", 2, 5), (">5", 5, 1e9)]


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def sg1(X, window=11, poly=2):
    from scipy.signal import savgol_filter
    w = window if window % 2 == 1 else window + 1
    if w >= X.shape[1]:
        w = X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2
    return savgol_filter(X, window_length=w, polyorder=poly, deriv=1, axis=1)


def pick_nc(Xp, y, groups, cands=NCOMP_CANDS):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold
    k = min(N_INNER, len(set(groups)))
    best = (np.inf, cands[0])
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
    return best[1]


def run_variant(P, y, groups, fold_of, nc_fixed=None):
    from sklearn.cross_decomposition import PLSRegression
    n = len(y)
    oof = np.full(n, np.nan)
    ncs = []
    for f in sorted(set(fold_of)):
        te = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        nc = nc_fixed if nc_fixed else pick_nc(P[tr], y[tr], groups[tr])
        ncs.append(nc)
        m = PLSRegression(n_components=nc).fit(P[tr], y[tr])
        oof[te] = np.clip(m.predict(P[te]).ravel(), 0, None)
    return oof, ncs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rand", type=int, default=30)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_batch2"
    F = root / "stage2_final"
    (S / "results").mkdir(parents=True, exist_ok=True)
    (S / "predictions").mkdir(parents=True, exist_ok=True)

    d = np.load(F / "manifests" / "data_icraf.npz", allow_pickle=True)
    Xm = d["X_mir"].astype(np.float64)
    y = d["y"].astype(np.float64)
    groups = d["groups"].astype(str)
    ids = d["ids"].astype(str)
    wave = d["w_mir"].astype(float)

    outs = pd.read_csv(F / "manifests" / "outer_split_manifest.csv")
    fold_of = outs.loc[outs["dataset"] == "icraf", "outer_fold"].to_numpy(int)

    pred_nir = pd.read_csv(S / "predictions" / "all_corrected_predictions.csv")
    pred_nir = pred_nir[pred_nir["dataset"] == "icraf"].set_index("sample_id").reindex(ids)
    pN = np.clip(pred_nir["pred_M1"].to_numpy(float), 0, None)
    depth = pred_nir["depth"].to_numpy(float)

    print("ICRAF n=%d profiles=%d MIR=%s range=%.0f-%.0f"
          % (len(y), len(set(groups)), Xm.shape, wave.min(), wave.max()), flush=True)

    P0 = sg1(snv(Xm))
    print("preprocess done %s" % (P0.shape,), flush=True)

    results, oof_store = {}, {}
    e1 = np.abs(y - pN)

    def report(tag, oof, ndrop, ncs):
        e = np.abs(y - oof)
        r2 = float(1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2))
        results[tag] = {"n_bands": int(P0.shape[1] - ndrop),
                        "mae": float(e.mean()),
                        "rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
                        "r2": r2, "gain_vs_nir": float((e1 - e).mean()),
                        "ncomp": int(np.median(ncs)), "dropped": int(ndrop)}
        oof_store[tag] = oof
        print("    %s MAE %.4f  R2 %.4f  nc=%d" % (tag, e.mean(), r2, int(np.median(ncs))),
              flush=True)

    # A0
    print("\n[A0] full MIR", flush=True)
    oof0, ncs0 = run_variant(P0, y, groups, fold_of)
    nc0 = int(np.median(ncs0))
    report("A0_full", oof0, 0, ncs0)

    # A1 / A3 / A4
    for tag, (lo, hi) in BANDS.items():
        m = ~((wave >= lo - EDGE) & (wave <= hi + EDGE))
        ndrop = int((~m).sum())
        print("\n[%s] drop %d-%d (+/-%d) -> %d bands removed" % (tag, lo, hi, EDGE, ndrop),
              flush=True)
        oof, ncs = run_variant(P0[:, m], y, groups, fold_of)
        report(tag, oof, ndrop, ncs)
        print("    delta vs A0: %+.4f" % (results[tag]["mae"] - results["A0_full"]["mae"]),
              flush=True)

    # C1 / C2 random contiguous
    rng = np.random.default_rng(42)
    for tag_c, tag_a in (("C1_rand_A3width", "A3_ClayOv_950_1200"),
                         ("C2_rand_A4width", "A4_Carb_2300_2500")):
        lo, hi = BANDS[tag_a]
        width = int(np.sum((wave >= lo - EDGE) & (wave <= hi + EDGE)))
        maes = []
        print("\n[%s] %d random contiguous windows, width=%d bands, nc=%d"
              % (tag_c, args.rand, width, nc0), flush=True)
        for it in range(args.rand):
            start = int(rng.integers(0, max(1, len(wave) - width)))
            mask = np.ones(len(wave), bool)
            mask[np.arange(start, start + width)] = False
            oof, _ = run_variant(P0[:, mask], y, groups, fold_of, nc_fixed=nc0)
            maes.append(float(np.abs(y - oof).mean()))
            if (it + 1) % 10 == 0:
                print("    %d/%d  median %.4f" % (it + 1, args.rand, np.median(maes)), flush=True)
        maes = np.array(maes)
        target = results[tag_a]["mae"]
        pct = float((maes <= target).mean() * 100)
        results[tag_c] = {"width": width, "n_iter": args.rand,
                          "mae_median": float(np.median(maes)),
                          "mae_p5": float(np.percentile(maes, 5)),
                          "mae_p95": float(np.percentile(maes, 95)),
                          "target_mae": float(target), "target_percentile": pct,
                          "all_maes": [float(x) for x in maes]}
        print("    random median %.4f [p5 %.4f, p95 %.4f]"
              % (np.median(maes), np.percentile(maes, 5), np.percentile(maes, 95)))
        verdict = "specific" if pct <= 5 else ("marginal" if pct <= 20 else "indistinguishable")
        print("    %s MAE %.4f at %.0f percentile -> %s" % (tag_a, target, pct, verdict), flush=True)

    # per-sample predictions
    recs = []
    for tag, oof in oof_store.items():
        e = np.abs(y - oof)
        for i in range(len(y)):
            recs.append({"sample_id": ids[i], "profile_id": groups[i],
                         "outer_fold": int(fold_of[i]), "variant": tag,
                         "true_SOC": float(y[i]), "pred_NIR": float(pN[i]),
                         "pred_MIR": float(oof[i]), "abs_err_nir": float(e1[i]),
                         "abs_err_mir": float(e[i]), "measurement_gain": float(e1[i] - e[i]),
                         "depth": float(depth[i]) if np.isfinite(depth[i]) else np.nan})
    pd.DataFrame(recs).to_csv(S / "predictions" / "ablation_predictions.csv", index=False)

    # profile-level paired bootstrap
    print("\n=== profile-level paired bootstrap: variant - A0 (3000x) ===", flush=True)
    uped = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uped}
    rngb = np.random.default_rng(42)
    boot = {}
    for tag in oof_store:
        if tag == "A0_full":
            continue
        ea = np.abs(y - oof_store[tag])
        e0 = np.abs(y - oof_store["A0_full"])
        diffs = []
        for _ in range(3000):
            pick = rngb.choice(uped, size=len(uped), replace=True)
            idx = np.concatenate([idx_by[g] for g in pick])
            diffs.append(float(ea[idx].mean() - e0[idx].mean()))
        diffs = np.array(diffs)
        lo_ci, hi_ci = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        boot[tag] = {"mean": float(diffs.mean()), "ci_lo": lo_ci, "ci_hi": hi_ci,
                     "p_better_pct": float((diffs < 0).mean() * 100)}
        print("  %-22s %+.4f  95%%CI [%+.4f, %+.4f]  P(better)=%.1f%%"
              % (tag, diffs.mean(), lo_ci, hi_ci, 100 * (diffs < 0).mean()), flush=True)

    # strata
    print("\n=== stratified MAE by true SOC ===", flush=True)
    strata = []
    for tag, oof in oof_store.items():
        e = np.abs(y - oof)
        row = {"variant": tag}
        for nm, lo, hi in SOC_BANDS:
            m = (y >= lo) & (y < hi)
            row[nm] = float(e[m].mean()) if m.sum() else np.nan
            row["n_" + nm] = int(m.sum())
        strata.append(row)
    st = pd.DataFrame(strata)
    print(st.round(4).to_string(index=False), flush=True)

    out = {"results": results, "bootstrap": boot, "strata_by_soc": strata,
           "edge_protection": EDGE, "n_samples": int(len(y)),
           "n_profiles": int(len(set(groups)))}
    (S / "results" / "spectral_ablation_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n[OK] %s" % (S / "results" / "spectral_ablation_results.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
