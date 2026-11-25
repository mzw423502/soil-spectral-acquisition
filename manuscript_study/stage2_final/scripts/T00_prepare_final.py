#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T00_prepare_final.py —— 第二阶段：三库统一数据与固定验证划分

统一口径（写进 FINAL_ANALYSIS_PROTOCOL.md）：
  · SOC 质量百分比；负预测截断为 0
  · NIR 统一裁到 1350–2500 nm
  · MIR 保留各库有效波段（NeoSpectra/ICRAF 600–4000；LUCAS 628–4000）
  · 预处理（SNV + SG1, w=11, poly=2）**在训练脚本内做**，本脚本只存原始光谱
    —— 因为波段消融要求"先预处理、再删波段"
  · 分组：NeoSpectra=pedon；ICRAF=profile_plotcode；LUCAS=1°空间网格
  · 外层 GroupKFold(5) 固定；内层 GroupKFold(3)

输出（写 stage2_final/）：
  manifests/data_{dataset}.npz          原始光谱 + 标签 + 分组
  manifests/sample_manifest.csv
  manifests/outer_split_manifest.csv
  manifests/inner_split_manifest.csv

用法：python3 T00_prepare_final.py --root <manuscript_study 根>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NIR_LO, NIR_HI = 1350, 2500
N_OUTER, N_INNER = 5, 3
SEED = 42


def load_icraf(root: Path):
    D = root / "data" / "icraf"
    lab = pq.read_table(D / "ossl_soillab_v1.3.parquet",
                        columns=["id.layer_local_c", "oc_usda.c1059_w.pct",
                                 "clay.tot_usda.a334_w.pct", "sand.tot_usda.c60_w.pct",
                                 "ph.h2o_usda.a268_index", "caco3_usda.a54_w.pct"]).to_pandas()
    site = pq.read_table(D / "ossl_soilsite_v1.3.parquet",
                         columns=["id.layer_local_c", "loc.country_src_txt",
                                  "layer.upper.depth_usda_cm"]).to_pandas()
    cw = pd.read_csv(root / "manifests" / "ICRAF_ISRIC_v1.3_profile_crosswalk.tsv", sep="\t")

    def spec(fname, lo, hi):
        pf = pq.ParquetFile(D / fname)
        names = pf.schema_arrow.names
        cols = [c for c in names if c.startswith("scan_")]
        wave = np.array([int(re.search(r"\.(\d+)_", c).group(1)) for c in cols])
        o = np.argsort(wave); cols = [cols[i] for i in o]; wave = wave[o]
        m = (wave >= lo) & (wave <= hi)
        cols = [c for c, k in zip(cols, m) if k]; wave = wave[m]
        t = pq.read_table(D / fname, columns=["id.layer_local_c"] + cols).to_pandas()
        return t.groupby("id.layer_local_c", as_index=False)[cols].mean(), wave

    mir, w_mir = spec("ossl_mir_v1.3.parquet", 600, 4000)
    nir, w_nir = spec("ossl_visnir_v1.3.parquet", NIR_LO, NIR_HI)

    df = lab.rename(columns={"oc_usda.c1059_w.pct": "soc"}).merge(site, on="id.layer_local_c")
    df = df.merge(nir, on="id.layer_local_c").merge(mir, on="id.layer_local_c")
    df = df.merge(cw[["id.layer_local_c", "profile_plotcode"]], on="id.layer_local_c", how="left")
    # 与 audit_pack 口径对齐：保留 SOC>=0（不剔 0），不设上限（与 3,698 可比）
    df = df[df["soc"].notna() & (df["soc"] >= 0)]
    df = df[df["profile_plotcode"].notna()]
    df["group"] = df["profile_plotcode"].astype(str)
    nir_cols = [c for c in nir.columns if c != "id.layer_local_c"]
    mir_cols = [c for c in mir.columns if c != "id.layer_local_c"]
    return df, nir_cols, mir_cols, w_nir, w_mir, "profile_plotcode"


def load_lucas(root: Path, grid: float = 1.0):
    D = root / "data" / "lucas"
    lab = pq.read_table(D / "ossl_soillab_v1.3.parquet").to_pandas()
    site = pq.read_table(D / "ossl_soilsite_v1.3.parquet").to_pandas()

    def spec(fname, lo, hi):
        pf = pq.ParquetFile(D / fname)
        names = pf.schema_arrow.names
        cols = [c for c in names if c.startswith("scan_")]
        wave = np.array([int(re.search(r"\.(\d+)_", c).group(1)) for c in cols])
        o = np.argsort(wave); cols = [cols[i] for i in o]; wave = wave[o]
        m = (wave >= lo) & (wave <= hi)
        cols = [c for c, k in zip(cols, m) if k]; wave = wave[m]
        t = pq.read_table(D / fname, columns=["id.layer_local_c"] + cols).to_pandas()
        return t.groupby("id.layer_local_c", as_index=False)[cols].mean(), wave

    mir, w_mir = spec("ossl_mir_v1.3.parquet", 600, 4000)
    nir, w_nir = spec("ossl_visnir_v1.3.parquet", NIR_LO, NIR_HI)

    df = lab[["id.layer_local_c", "oc_iso.10694_w.pct"]].rename(
        columns={"oc_iso.10694_w.pct": "soc"})
    df = df.merge(site[["id.layer_local_c", "loc.country_src_txt",
                        "longitude.point_wgs84_dd", "latitude.point_wgs84_dd",
                        "layer.upper.depth_usda_cm"]]
                  .rename(columns={"loc.country_src_txt": "country"}), on="id.layer_local_c")
    df = df.merge(nir, on="id.layer_local_c").merge(mir, on="id.layer_local_c")
    df = df[df["soc"].notna() & (df["soc"] > 0) & (df["soc"] <= 40)]
    df = df[df["longitude.point_wgs84_dd"].notna() & df["latitude.point_wgs84_dd"].notna()]
    gx = np.floor(df["longitude.point_wgs84_dd"] / grid).astype(int)
    gy = np.floor(df["latitude.point_wgs84_dd"] / grid).astype(int)
    df["group"] = gx.astype(str) + "_" + gy.astype(str)
    nir_cols = [c for c in nir.columns if c != "id.layer_local_c"]
    mir_cols = [c for c in mir.columns if c != "id.layer_local_c"]
    return df, nir_cols, mir_cols, w_nir, w_mir, f"{grid}deg_grid"


def load_neospectra(root: Path):
    npz = root.parent / "stage2" / "results" / "single_device_data.npz"
    d = np.load(npz, allow_pickle=True)
    w_nir = d["nir_wavelengths"].astype(float)
    w_mir = d["mir_wavenumbers"].astype(float)
    keep = (w_nir >= NIR_LO) & (w_nir <= NIR_HI)
    Xn = d["nir"][:, keep].astype(np.float64)
    Xm = d["mir"].astype(np.float64)
    w_nir = w_nir[keep]
    df = pd.DataFrame({
        "id.layer_local_c": d["keys"].astype(str),
        "soc": d["y"].astype(float),
        "group": d["pedon"].astype(str),
        "country": d["lab"].astype(str),
        "layer.upper.depth_usda_cm": np.nan,
    })
    n = len(df)
    nir_cols = [f"n{i}" for i in range(Xn.shape[1])]
    mir_cols = [f"m{i}" for i in range(Xm.shape[1])]
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(Xn, columns=nir_cols),
                    pd.DataFrame(Xm, columns=mir_cols)], axis=1)
    return df, nir_cols, mir_cols, w_nir, w_mir, "pedon"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    S = root / "stage2_final"
    M = S / "manifests"
    M.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import GroupKFold

    samples, outs, ins = [], [], []
    meta = {}

    for name, fn in (("neospectra", load_neospectra), ("icraf", load_icraf), ("lucas", load_lucas)):
        print(f"\n=== {name} ===", flush=True)
        df, nir_cols, mir_cols, w_nir, w_mir, gsrc = fn(root)
        df = df.reset_index(drop=True)
        y = df["soc"].to_numpy(float)
        groups = df["group"].to_numpy(str)
        Xn = df[nir_cols].to_numpy(np.float32)
        Xm = df[mir_cols].to_numpy(np.float32)
        print(f"  n={len(df)}  分组 {len(set(groups))}  NIR {Xn.shape} ({w_nir.min():.0f}-{w_nir.max():.0f}) "
              f"MIR {Xm.shape} ({w_mir.min():.0f}-{w_mir.max():.0f})")
        # 分辨率（用于避免"11点=相同物理宽度"的误述）
        d_nir = float(np.median(np.diff(w_nir))) if len(w_nir) > 1 else np.nan
        d_mir = float(np.median(np.diff(w_mir))) if len(w_mir) > 1 else np.nan
        print(f"  分辨率: NIR {d_nir:.1f} nm/点, MIR {d_mir:.1f} cm-1/点")

        np.savez_compressed(M / f"data_{name}.npz",
                            X_nir=Xn, X_mir=Xm, y=y.astype(np.float32),
                            groups=groups, ids=df["id.layer_local_c"].to_numpy(str),
                            w_nir=w_nir.astype(np.float32), w_mir=w_mir.astype(np.float32))

        # 外层五折（固定）
        ng = len(set(groups))
        nsp = min(N_OUTER, ng)
        gkf = GroupKFold(n_splits=nsp)
        fold_of = np.full(len(df), -1, int)
        for f, (tr, te) in enumerate(gkf.split(Xn, y, groups), 1):
            fold_of[te] = f
            # 内层三折（在外层训练集内部）
            gtr = groups[tr]
            nin = min(N_INNER, len(set(gtr)))
            for fi, (itr, iva) in enumerate(GroupKFold(n_splits=nin).split(tr, groups=gtr), 1):
                for j in iva:
                    ins.append({"dataset": name, "sample_id": df["id.layer_local_c"].iloc[tr[j]],
                                "outer_fold": f, "inner_fold": fi})
        for i in range(len(df)):
            outs.append({"dataset": name, "sample_id": df["id.layer_local_c"].iloc[i],
                         "group": groups[i], "outer_fold": int(fold_of[i]), "y": float(y[i])})
            samples.append({"dataset": name, "sample_id": df["id.layer_local_c"].iloc[i],
                            "group": groups[i], "grouping": gsrc,
                            "country": df["country"].iloc[i] if "country" in df.columns else "",
                            "soc": float(y[i]),
                            "depth_top": df["layer.upper.depth_usda_cm"].iloc[i]
                            if "layer.upper.depth_usda_cm" in df.columns else np.nan})
        meta[name] = {"n": int(len(df)), "n_groups": int(ng), "n_folds": int(nsp),
                      "nir_bands": int(Xn.shape[1]), "mir_bands": int(Xm.shape[1]),
                      "nir_range": [float(w_nir.min()), float(w_nir.max())],
                      "mir_range": [float(w_mir.min()), float(w_mir.max())],
                      "nir_resolution_nm": d_nir, "mir_resolution_cm1": d_mir,
                      "grouping": gsrc,
                      "soc": {"min": float(y.min()), "median": float(np.median(y)),
                              "max": float(y.max())}}
        print(f"  外层 {nsp} 折: " + ", ".join(
            f"{f}:{int((fold_of==f).sum())}" for f in range(1, nsp + 1)))

    pd.DataFrame(samples).to_csv(M / "sample_manifest.csv", index=False)
    pd.DataFrame(outs).to_csv(M / "outer_split_manifest.csv", index=False)
    pd.DataFrame(ins).to_csv(M / "inner_split_manifest.csv", index=False)
    (M / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] manifests 写入 {M}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
