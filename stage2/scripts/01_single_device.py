#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_single_device.py —— 任务1：重建真实的单设备测量场景

problem：第一阶段把同一样品在**多台仪器**上的扫描平均了。
         真实手持设备场景下只能拿到一台仪器的结果，那种"信息优势"必须排除。

做法：
  1. 原始 48,568 条 NIR 扫描 -> 按 (kssl_id, scanner_SerialNo) 分组
  2. 保留扫描数 >= MIN_SCANS 的组合
  3. 每个样品**只选一台设备**，选择规则 = sha256(kssl_id) 取模，
     **完全不使用 SOC / 光谱质量 / 预测效果**，且可复现
  4. 从该设备的扫描里确定性抽 MIN_SCANS 次，算平均光谱与扫描间标准差
  5. 按 kssl_id 连 MIR 与 SOC
  6. 重算第一阶段四项指标（NIR 误差 / MIR 误差 / 随机 20% / 理想 20%）

门禁：若单设备口径下 理想20% 相对不追加的 MAE 降幅 < 10%，说明追加收益是
      多设备平均带来的假象，立即停止后续算法开发。

用法：
  python3 01_single_device.py --root ~/soil_spectral_validation --out stage2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MIN_SCANS = 6
SEED_SALT = "stage2_single_device_v1"


def sha_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def to_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s or s.upper() in ("NA", "NAN", "NULL", ".", "-"):
        return None
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="soil_spectral_validation 根目录")
    ap.add_argument("--out", default="stage2")
    ap.add_argument("--min-scans", type=int, default=MIN_SCANS)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = root / args.out
    for d in ("data_audit", "scripts", "models", "results", "figures", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    nir_p = root / "data" / "neospectra" / "Neospectra_WoodwellKSSL_NIR.csv"
    mir_p = root / "data" / "neospectra" / "Neospectra_WoodwellKSSL_MIR.csv"

    # ---------------- 1. 读 NIR，按 (kssl, scanner) 分组 ----------------
    print(f"[1] 读 NIR {nir_p.name}")
    with open(nir_p, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        idx = {n: i for i, n in enumerate(hdr)}
        wcols = [i for i, n in enumerate(hdr) if n.replace(".", "").isdigit()]
        wcols.sort(key=lambda i: float(hdr[i]))
        wavelengths = np.array([float(hdr[i]) for i in wcols], dtype=np.float32)

        i_k, i_s = idx["kssl_id"], idx["scanner_SerialNo"]
        i_e, i_p = idx["eoc_tot_c"], idx["lims.pedon.id"]
        i_scan = idx.get("SCAN_ID")
        i_lab = idx.get("Lab")
        i_top, i_bot = idx.get("lay.depth.to.top"), idx.get("lay.depth.to.bottom")

        groups = defaultdict(list)   # (kssl, scanner) -> [(scan_id, spec, eoc)]
        meta = {}
        nrow = 0
        for r in rd:
            if not r or len(r) < len(hdr):
                continue
            nrow += 1
            k, sc = r[i_k], r[i_s]
            vec = np.empty(len(wcols), dtype=np.float32)
            ok = True
            for jj, j in enumerate(wcols):
                v = to_float(r[j])
                if v is None:
                    ok = False
                    break
                vec[jj] = v
            if not ok:
                continue
            e = to_float(r[i_e])
            sid = r[i_scan] if i_scan is not None else f"{k}_{sc}_{len(groups[(k, sc)])}"
            groups[(k, sc)].append((sid, vec, e))
            if k not in meta:
                meta[k] = {
                    "pedon": r[i_p] if i_p is not None else "",
                    "lab": r[i_lab] if i_lab is not None else "",
                    "top": to_float(r[i_top]) if i_top is not None else None,
                    "bot": to_float(r[i_bot]) if i_bot is not None else None,
                }
    print(f"    扫描 {nrow:,}，(kssl,scanner) 组合 {len(groups):,}，样品 {len(meta):,}")

    # ---------------- 2. 每样品的候选设备 ----------------
    by_sample = defaultdict(dict)
    for (k, sc), lst in groups.items():
        if len(lst) >= args.min_scans:
            by_sample[k][sc] = lst
    n_multi = sum(1 for v in by_sample.values() if len(v) > 1)
    print(f"[2] 合格组合(>= {args.min_scans} 扫描) 覆盖样品 {len(by_sample):,}；"
          f"其中多设备可选 {n_multi:,}")

    # ---------------- 3. 确定性选设备 + 抽扫描 ----------------
    chosen = []
    scan_pool_extra = defaultdict(list)   # kssl -> 其他设备的平均光谱（泛化实验备用）
    for k, devs in by_sample.items():
        dl = sorted(devs.keys())
        h = sha_int(f"{SEED_SALT}|{k}")
        sc = dl[h % len(dl)]
        lst = sorted(devs[sc], key=lambda t: str(t[0]))
        rng = np.random.default_rng(sha_int(f"{SEED_SALT}|scan|{k}"))
        pick = np.sort(rng.choice(len(lst), size=args.min_scans, replace=False))
        chosen.append((k, sc, [lst[i] for i in pick], len(lst), len(dl)))
        # 其他设备留作后续跨设备实验
        for other, olst in devs.items():
            if other == sc:
                continue
            m = np.mean([t[1] for t in olst], axis=0)
            scan_pool_extra[k].append((other, m))
    print(f"[3] 选定单设备样品 {len(chosen):,}")

    # ---------------- 4. MIR ----------------
    print(f"[4] 读 MIR")
    with open(mir_p, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        rd = csv.reader(fh)
        mh = next(rd)
        m_wcols = [i for i, n in enumerate(mh) if n.replace(".", "").isdigit()]
        m_wcols.sort(key=lambda i: float(mh[i]))
        mir = {}
        for r in rd:
            if not r or len(r) < len(mh):
                continue
            vec = np.empty(len(m_wcols), dtype=np.float32)
            ok = True
            for jj, j in enumerate(m_wcols):
                v = to_float(r[j])
                if v is None:
                    ok = False
                    break
                vec[jj] = v
            if ok:
                mir[r[0]] = vec
    m_wavenums = np.array([float(mh[i]) for i in m_wcols], dtype=np.float32)
    print(f"    MIR 样品 {len(mir):,}，波段 {len(m_wcols)}")

    # ---------------- 5. 组装 ----------------
    recs = []
    for k, sc, scans, n_dev_scans, n_devs in chosen:
        eocs = [e for (_, _, e) in scans if e is not None and 0 <= e <= 100]
        if not eocs:
            continue
        specs = np.stack([t[1] for t in scans])
        mean_spec = specs.mean(axis=0)
        scan_sd = specs.std(axis=0)
        recs.append({
            "kssl_id": k,
            "pedon": meta[k]["pedon"], "lab": meta[k]["lab"],
            "depth_top": meta[k]["top"], "depth_bottom": meta[k]["bot"],
            "scanner": sc,
            "scan_ids": "|".join(str(t[0]) for t in scans),
            "n_scans_this_device": n_dev_scans,
            "n_devices_available": n_devs,
            "soc_ref": float(st.median(eocs)),
            "soc_sd": float(st.pstdev(eocs)) if len(eocs) > 1 else 0.0,
            "nir_mean": mean_spec,
            "nir_scan_sd_mean": float(scan_sd.mean()),
            "nir_scans": specs,
            "has_mir": k in mir,
            "mir": mir.get(k),
        })
    recs = [r for r in recs if r["has_mir"]]
    recs.sort(key=lambda r: r["kssl_id"])
    print(f"[5] 单设备 + MIR + SOC 齐全: {len(recs):,}")
    print(f"    选中的设备分布: {dict(Counter(r['scanner'] for r in recs).most_common())}")
    print(f"    实验室分布: {dict(Counter(r['lab'] for r in recs))}")
    print(f"    SOC: {min(r['soc_ref'] for r in recs):.3f} .. {max(r['soc_ref'] for r in recs):.3f}, "
          f"中位 {st.median([r['soc_ref'] for r in recs]):.3f}")
    print(f"    剖面数: {len({r['pedon'] for r in recs if r['pedon']})}")

    # 审计 CSV
    audit_p = out / "data_audit" / "single_device_audit.csv"
    with open(audit_p, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["kssl_id", "pedon", "lab", "depth_top", "depth_bottom",
                     "scanner_SerialNo", "six_scan_ids", "n_scans_device",
                     "n_devices_available", "soc_ref", "soc_sd",
                     "nir_scan_sd_mean", "has_mir"])
        for r in recs:
            wr.writerow([r["kssl_id"], r["pedon"], r["lab"], r["depth_top"], r["depth_bottom"],
                         r["scanner"], r["scan_ids"], r["n_scans_this_device"],
                         r["n_devices_available"], f"{r['soc_ref']:.4f}", f"{r['soc_sd']:.4f}",
                         f"{r['nir_scan_sd_mean']:.6f}", int(r["has_mir"])])
    print(f"[OK] {audit_p}")

    # npz（供后续任务）
    npz_p = out / "results" / "single_device_data.npz"
    np.savez_compressed(
        npz_p,
        keys=np.array([r["kssl_id"] for r in recs]),
        pedon=np.array([r["pedon"] for r in recs]),
        lab=np.array([r["lab"] for r in recs]),
        scanner=np.array([r["scanner"] for r in recs]),
        y=np.array([r["soc_ref"] for r in recs], dtype=np.float32),
        y_sd=np.array([r["soc_sd"] for r in recs], dtype=np.float32),
        nir=np.stack([r["nir_mean"] for r in recs]).astype(np.float32),
        nir_scans=np.stack([r["nir_scans"] for r in recs]).astype(np.float32),
        nir_wavelengths=wavelengths,
        mir=np.stack([r["mir"] for r in recs]).astype(np.float32),
        mir_wavenumbers=m_wavenums,
        n_devices=np.array([r["n_devices_available"] for r in recs], dtype=np.int16),
    )
    print(f"[OK] {npz_p}")

    summary = {
        "raw_scans": nrow,
        "kssl_scanner_groups": len(groups),
        "samples_total": len(meta),
        "samples_with_ge_min_scans": len(by_sample),
        "min_scans_required": args.min_scans,
        "samples_multi_device": n_multi,
        "single_device_samples_complete": len(recs),
        "n_pedons": len({r["pedon"] for r in recs if r["pedon"]}),
        "device_selection_rule": "sha256(salt|kssl_id) % n_devices（与 SOC 无关，可复现）",
        "scan_pick_rule": "sorted(SCAN_ID) 后由 sha256(salt|scan|kssl_id) 播种的无放回抽样",
        "scanner_counts": dict(Counter(r["scanner"] for r in recs)),
        "lab_counts": dict(Counter(r["lab"] for r in recs)),
    }
    (out / "data_audit" / "single_device_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------------- 6. 重算第一阶段四项指标（门禁） ----------------
    print("\n[6] 重算第一阶段四项指标（单设备口径）")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bp", str(root / "scripts" / "02_baseline_prediction.py"))
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)

    Xn = np.stack([r["nir_mean"] for r in recs]).astype(np.float64)
    Xm = np.stack([r["mir"] for r in recs]).astype(np.float64)
    y = np.array([r["soc_ref"] for r in recs], dtype=np.float64)
    ped = np.array([r["pedon"] for r in recs])

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(5, len(set(ped))))
    oof = {k: np.full(len(y), np.nan) for k in ("M0", "M1", "M2", "M3")}
    for f, (tr, te) in enumerate(gkf.split(Xn, y, ped), 1):
        oof["M0"][te] = y[tr].mean()
        A, B = bp.preprocess(Xn[tr], Xn[te]); m, _ = bp.fit_pls(A, y[tr], ped[tr]); oof["M1"][te] = m.predict(B).ravel()
        A, B = bp.preprocess(Xm[tr], Xm[te]); m, _ = bp.fit_pls(A, y[tr], ped[tr]); oof["M2"][te] = m.predict(B).ravel()
        A, B = bp.preprocess(np.hstack([Xn[tr], Xm[tr]]), np.hstack([Xn[te], Xm[te]]))
        m, _ = bp.fit_pls(A, y[tr], ped[tr]); oof["M3"][te] = m.predict(B).ravel()
        print(f"    fold {f}/{gkf.get_n_splits()}", flush=True)

    res = {}
    for k in ("M0", "M1", "M2", "M3"):
        res[k] = bp.metrics(y, oof[k])
        r = res[k]
        print(f"    {k}: RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} R2={r['r2']:.4f} RPD={r['rpd']:.2f}")

    e1 = np.abs(y - oof["M1"]); e3 = np.abs(y - oof["M3"])
    G = e1 - e3
    n = len(G)
    b = 0.20
    k = int(round(b * n))
    order = np.argsort(-G)
    sel = np.zeros(n, bool); sel[order[:k]] = True
    mae_none = float(e1.mean())
    mae_rand = float(e1.mean() * (1 - b) + e3.mean() * b)
    mae_ideal = float(np.where(sel, e3, e1).mean())
    mae_full = float(e3.mean())
    drop_ideal = (mae_none - mae_ideal) / mae_none * 100
    drop_rand = (mae_none - mae_rand) / mae_none * 100

    print(f"\n    增益分布: 改善 {(G>0).mean()*100:.1f}%  变差 {(G<0).mean()*100:.1f}%  "
          f"中位 {np.median(G):+.4f}")
    print(f"    不追加 {mae_none:.4f} | 随机20% {mae_rand:.4f} ({drop_rand:+.1f}%) | "
          f"理想20% {mae_ideal:.4f} ({drop_ideal:+.1f}%) | 全部 {mae_full:.4f}")

    gate = drop_ideal >= 10.0
    print(f"\n    门禁（理想20% 降幅 >= 10%）: {drop_ideal:.1f}% -> "
          f"{'✅ 通过，继续任务2-8' if gate else '❌ 未通过，按方案停止算法开发'}")

    # 存 OOF 供后续复用
    with open(out / "results" / "stage1_recomputed_oof.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["i", "kssl_id", "pedon", "y_true", "y_M0", "y_M1", "y_M2", "y_M3",
                     "gain_M1_M3"])
        for i in range(n):
            wr.writerow([i, recs[i]["kssl_id"], ped[i], f"{y[i]:.4f}",
                         f"{oof['M0'][i]:.4f}", f"{oof['M1'][i]:.4f}",
                         f"{oof['M2'][i]:.4f}", f"{oof['M3'][i]:.4f}", f"{G[i]:+.4f}"])

    gate_out = {
        "n_single_device_samples": len(recs),
        "metrics": res,
        "gain": {"improve_pct": float((G > 0).mean() * 100),
                 "worsen_pct": float((G < 0).mean() * 100),
                 "median": float(np.median(G))},
        "budget_20": {"none": mae_none, "random": mae_rand, "ideal": mae_ideal, "full": mae_full,
                      "drop_random_pct": drop_rand, "drop_ideal_pct": drop_ideal},
        "gate_ideal_drop_ge_10pct": bool(gate),
        "meta": summary,
    }
    (out / "data_audit" / "stage1_recompute.json").write_text(
        json.dumps(gate_out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[OK] {out/'data_audit'/'stage1_recompute.json'}")
    return 0 if gate else 3


if __name__ == "__main__":
    sys.exit(main())
