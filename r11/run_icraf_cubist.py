"""ICRAF–ISRIC Cubist experiment on frozen R10 outer/inner folds.

Use --validate-inputs without Cubist to validate the frozen cohort, folds and
allocation calculation against the historical, already fitted PLSR benchmark.
Normal execution trains Cubist and never substitutes another estimator.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


N = 3698
SEED = 20261125
B = 3000
WEIGHTS = np.arange(21, dtype=np.float64) / 20
DEFAULT_CONFIG = {"n_rules": 500, "n_committees": 1, "neighbors": None,
                  "random_state": SEED}
DATA = {}


def read_inputs(root):
    path = root / "stage2_final/manifests"
    files = {x: path / x for x in ("data_icraf.npz", "outer_split_manifest.csv",
                                  "inner_split_manifest.csv")}
    with np.load(files["data_icraf.npz"], allow_pickle=True) as raw:
        ids = raw["ids"].astype(str)
        g = raw["groups"].astype(str)
        y = raw["y"].astype(np.float64)
        nir = raw["X_nir"].astype(np.float64)
        mir = raw["X_mir"].astype(np.float64)
    assert len(ids) == len(np.unique(ids)) == len(y) == N
    assert nir.shape == (N, 576) and mir.shape == (N, 1701)
    assert all(np.isfinite(x).all() for x in (y, nir, mir))
    outer = pd.read_csv(files["outer_split_manifest.csv"],
                        dtype={"sample_id": str, "group": str})
    outer = outer.query("dataset == 'icraf'").set_index("sample_id")
    assert outer.index.is_unique and set(outer.index) == set(ids)
    outer = outer.loc[ids]
    # The manifest serializes raw decimal SOC; the frozen NPZ stores float32.
    np.testing.assert_array_equal(outer["y"].to_numpy(np.float32), y.astype(np.float32))
    assert np.array_equal(outer["group"].to_numpy(str), g)
    f = outer["outer_fold"].to_numpy(int)
    assert set(f) == set(range(1, 6))
    irows = pd.read_csv(files["inner_split_manifest.csv"], dtype={"sample_id": str})
    irows = irows.query("dataset == 'icraf'")
    assert set(irows.outer_fold) == set(f)
    inner = {}
    for fold in range(1, 6):
        calibration = (f != fold)
        assert not set(g[calibration]) & set(g[~calibration])
        ipart = irows[irows.outer_fold == fold].set_index("sample_id")
        assert ipart.index.is_unique and set(ipart.index) == set(ids[calibration])
        si = ipart.loc[ids[calibration], "inner_fold"].to_numpy(int)
        assert set(si) == {1, 2, 3}
        for sub in (1, 2, 3):
            assert not set(g[calibration][si == sub]) & set(g[calibration][si != sub])
        inner[fold] = si
    h = {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in files.items()}
    return {"id": ids, "group": g, "y": y, "fold": f, "inner": inner,
            "nir": nir, "mir": mir, "sha256": h}


def prep(x):
    sd = x.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return savgol_filter((x - x.mean(1, keepdims=True)) / sd, 11, 2, deriv=1, axis=1)


def init_worker(root, vendor):
    if vendor is not None:
        sys.path.insert(0, str(vendor))
    global DATA
    DATA = read_inputs(root)
    DATA["nir"] = prep(DATA["nir"])
    DATA["mir"] = prep(DATA["mir"])


def fit_predict(train_x, train_y, test_x):
    from cubist import Cubist
    model = Cubist(**DEFAULT_CONFIG)
    model.fit(train_x, train_y)
    p = np.maximum(0, np.asarray(model.predict(test_x), dtype=float).ravel())
    assert len(p) == len(test_x) and np.isfinite(p).all()
    return p


def fit_fold(fold):
    d = DATA
    te = np.flatnonzero(d["fold"] == fold)
    tr = np.flatnonzero(d["fold"] != fold)
    assignments = d["inner"][fold]
    y = d["y"]
    inner_oof, outer_pred = {}, {}
    for source in ("nir", "mir"):
        print(f"Fold {fold} {source}: starting 3 group-held-out calibration fits and outer fit", flush=True)
        x = d[source]
        pred = np.full(len(tr), np.nan)
        for sub in (1, 2, 3):
            itr = tr[assignments != sub]
            iva = tr[assignments == sub]
            pred[assignments == sub] = fit_predict(x[itr], y[itr], x[iva])
            print(f"Fold {fold} {source}: calibration fit {sub}/3 complete", flush=True)
        assert np.isfinite(pred).all()
        inner_oof[source] = pred
        outer_pred[source] = fit_predict(x[tr], y[tr], x[te])
    stack = np.vstack([w * inner_oof["nir"] + (1 - w) * inner_oof["mir"] for w in WEIGHTS])
    scores = np.abs(stack - y[tr]).mean(1)
    w = float(WEIGHTS[np.argmin(scores)])
    p1, p2 = outer_pred["nir"], outer_pred["mir"]
    p4 = w * p1 + (1 - w) * p2
    return (pd.DataFrame({"sample_id": d["id"][te], "group": d["group"][te],
                          "outer_fold": fold, "y": y[te], "pred_M1": p1,
                          "pred_M2": p2, "pred_M4": p4, "weight_M4_NIR": w}),
            pd.DataFrame({"outer_fold": fold, "weight_M4_NIR": WEIGHTS,
                          "inner_oof_MAE": scores, "selected": WEIGHTS == w}))


def bootstrap(groups, measures):
    labs, group_id = np.unique(groups.astype(str), return_inverse=True)
    count = np.bincount(group_id)
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(labs), size=(B, len(labs)))
    denom = count[sample].sum(1)
    metrics = {}
    boot = {}
    for key, values in measures.items():
        total = np.bincount(group_id, weights=values)
        boot[key] = total[sample].sum(1) / denom
    boot["rel_reduction_pct"] = -100 * boot["priority_minus_random"] / boot["MAE_random_20"]
    for key, vals in boot.items():
        metrics[key] = np.quantile(vals, [.025, .975]).tolist()
    return len(labs), metrics


def summarize(df):
    df = df.copy().reset_index(drop=True)
    assert len(df) == N and df.sample_id.is_unique
    errs = {m: np.abs(df["y"].to_numpy(float) - df[f"pred_{m}"].to_numpy(float))
            for m in ("M1", "M2", "M4")}
    for m, e in errs.items():
        df[f"error_{m}"] = e
    df["G_M"] = errs["M1"] - errs["M2"]
    df["G_F"] = errs["M1"] - errs["M4"]
    df["selected_20"] = False
    df["random_fraction_20"] = np.nan
    folds = []
    for fold, sub in df.groupby("outer_fold", sort=True):
        k = len(sub) // 5
        rnk = sub.sort_values(["pred_M1", "sample_id"], ascending=[False, True])
        df.loc[rnk.index[:k], "selected_20"] = True
        df.loc[sub.index, "random_fraction_20"] = k / len(sub)
        folds.append({"outer_fold": int(fold), "n_test": int(len(sub)), "selected_20": int(k)})
    sel = df["selected_20"].to_numpy(bool)
    q = df["random_fraction_20"].to_numpy(float)
    df["error_priority_20"] = np.where(sel, errs["M4"], errs["M1"])
    df["error_random_20"] = (1 - q) * errs["M1"] + q * errs["M4"]
    df["priority_minus_random"] = df["error_priority_20"] - df["error_random_20"]
    metrics = {
        **{f"MAE_{m}": es for m, es in errs.items()},
        "MAE_priority_20": df["error_priority_20"].to_numpy(),
        "MAE_random_20": df["error_random_20"].to_numpy(),
        "priority_minus_random": df["priority_minus_random"].to_numpy(),
        "G_M_negative_pct": (df["G_M"].to_numpy() < 0).astype(float) * 100,
        "G_F_negative_pct": (df["G_F"].to_numpy() < 0).astype(float) * 100,
    }
    ngrp, ci = bootstrap(df["group"].to_numpy(str), metrics)
    summary = {key: float(values.mean()) for key, values in metrics.items()}
    summary["rel_reduction_pct"] = -100 * summary["priority_minus_random"] / summary["MAE_random_20"]
    selected = int(sel.sum())
    summary.update(n=N, group_count=ngrp, selected_20=selected,
                   realized_budget=selected / N, bootstrap_n=B, bootstrap_seed=SEED,
                   cluster_CI_95=ci,
                   G_M_negative_n=int((df["G_M"] < 0).sum()),
                   G_F_negative_n=int((df["G_F"] < 0).sum()))
    for fold_row in folds:
        sub = df[df.outer_fold == fold_row["outer_fold"]]
        keys = ["error_M1", "error_M2", "error_M4", "error_priority_20",
                "error_random_20", "priority_minus_random"]
        fold_row.update(sub[keys].mean().to_dict())
        if "weight_M4_NIR" in sub.columns:
            w = sub["weight_M4_NIR"].unique()
            assert len(w) == 1
            fold_row["weight_M4_NIR"] = float(w[0])
    return df, summary, pd.DataFrame(folds)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def input_check(root, out, d):
    p = root / "stage2_batch2/predictions/all_corrected_predictions.csv"
    ref = pd.read_csv(p, dtype={"sample_id": str, "group": str})
    ref = ref.query("dataset == 'icraf'").set_index("sample_id")
    assert ref.index.is_unique and set(ref.index) == set(d["id"])
    ref = ref.loc[d["id"]].reset_index()
    assert np.array_equal(ref["outer_fold"], d["fold"])
    assert np.array_equal(ref["group"].to_numpy(str), d["group"])
    np.testing.assert_allclose(ref["y"], d["y"], rtol=1e-12, atol=1e-12)
    _, check, folds = summarize(ref)
    check.update(status="validated_frozen_inputs_and_historical_PLSR_only",
                 fitted_Cubist=False, input_sha256=d["sha256"])
    write_json(out / "fixed_fold_PLSR_validation.json", check)
    folds.to_csv(out / "fixed_fold_PLSR_validation_folds.csv", index=False)
    print(json.dumps(check, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--cubist-target", type=Path, help="Separate directory housing the Cubist wheel installation")
    ap.add_argument("--workers", type=int, choices=range(1, 6), default=2)
    ap.add_argument("--folds", type=int, nargs="+", choices=range(1, 6),
                    default=list(range(1, 6)), help="Fit only selected outer folds")
    ap.add_argument("--finalize-only", action="store_true",
                    help="Assemble outputs from all five saved fold checkpoints")
    ap.add_argument("--validate-inputs", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = read_inputs(a.root)
    if a.validate_inputs:
        input_check(a.root, a.out, d)
        return
    if a.cubist_target is not None:
        sys.path.insert(0, str(a.cubist_target.resolve()))
    try:
        import cubist
    except ImportError as e:
        raise SystemExit(f"Cubist unavailable; no model was trained: {e}") from e
    version = importlib.metadata.version("cubist")
    t = perf_counter()
    runtime = a.out / "cubist_progress"
    runtime.mkdir(exist_ok=True)
    work = {}
    if a.finalize_only:
        for fold in range(1, 6):
            work[fold] = tuple(pd.read_csv(runtime / f"fold_{fold}_{label}.csv",
                                          dtype={"sample_id": str, "group": str},
                                          float_precision="round_trip")
                               for label in ("predictions", "calibration"))
    else:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker,
                                 initargs=(a.root, a.cubist_target)) as pool:
            futures = {pool.submit(fit_fold, fold): fold for fold in sorted(set(a.folds))}
            for x in as_completed(futures):
                fold = futures[x]
                work[fold] = x.result()
                for label, data in zip(("predictions", "calibration"), work[fold]):
                    data.to_csv(runtime / f"fold_{fold}_{label}.csv", index=False)
        if len(work) < 5:
            print(f"Finished partial outer folds: {sorted(work)}", flush=True)
            return
    pred = pd.concat([work[f][0] for f in sorted(work)], ignore_index=True)
    calibration = pd.concat([work[f][1] for f in sorted(work)], ignore_index=True)
    assert pred.sample_id.is_unique and set(pred.sample_id) == set(d["id"])
    locked = pred.set_index("sample_id").loc[d["id"]]
    np.testing.assert_array_equal(locked["outer_fold"].to_numpy(), d["fold"])
    np.testing.assert_array_equal(locked["group"].to_numpy(str), d["group"])
    np.testing.assert_array_equal(locked["y"].to_numpy(np.float64), d["y"])
    detail, summary, folds = summarize(pred)
    summary.update(
        status="complete", dataset="ICRAF–ISRIC", model="Cubist", cubist_version=version,
        python_version=platform.python_version(), parameters=DEFAULT_CONFIG,
        input_sha256=d["sha256"],
        elapsed_seconds=None if a.finalize_only else perf_counter() - t,
        execution_mode="assembled_from_saved_outer_folds" if a.finalize_only else "direct_fit",
        outer_folds=5, inner_folds=3, preproc="SNV + SG window 11, order 2, derivative 1",
        M4="Convex weight minimizing inner OOF MAE (grid 0.00–1.00 in increments of 0.05)",
        random_reference="Exact matched-quota expected loss within each held-out fold",
        uncertainty="Profile-cluster bootstrap conditional on original fitted predictions and selections",
    )
    detail.to_csv(a.out / "cubist_icraf_oof_predictions.csv", index=False)
    folds.to_csv(a.out / "cubist_icraf_folds.csv", index=False)
    calibration.to_csv(a.out / "cubist_icraf_M4_calibration.csv", index=False)
    write_json(a.out / "cubist_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
