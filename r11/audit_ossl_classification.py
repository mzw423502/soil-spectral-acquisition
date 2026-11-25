"""Check coverage of taxonomy in the exact OSSL v1.3 site parquet inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq


DISPLAY = {"icraf": "ICRAF–ISRIC", "lucas": "LUCAS–Woodwell"}
TERM = re.compile(r"(wrb|taxon|classification|reference[._ -]soil|great[._ -]group|suborder|soil[._ -]order|soil[._ -]group|soil[._ -]class|fao)", re.I)
PLACEHOLDERS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not classified", "not available", "<na>", "-"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "stage2_final/manifests/outer_split_manifest.csv"
    folds = pd.read_csv(manifest, dtype={"sample_id": str})
    rows, detail = [], {}
    for code, title in DISPLAY.items():
        rel = Path("data") / code / "ossl_soilsite_v1.3.parquet"
        path = args.root / rel
        table = pq.read_table(path).to_pandas()
        key = "id.layer_local_c"
        assert table[key].is_unique and table[key].notna().all()
        ids = folds.loc[folds.dataset == code, "sample_id"]
        assert ids.is_unique
        sites = table.set_index(key)
        matched = ids.isin(sites.index)
        assert matched.all(), (code, int(matched.sum()), len(ids))
        cohort = sites.loc[ids]
        candidates = [c for c in table.columns if TERM.search(c)]
        labels = {}
        any_label = pd.Series(False, index=cohort.index)
        for cname in candidates:
            values = cohort[cname].astype("string").str.strip()
            present = values.notna() & ~values.str.lower().fillna("").isin(PLACEHOLDERS)
            any_label |= present
            labels[cname] = {"present": int(present.sum()),
                             "counts": values[present].value_counts().to_dict()}
        report = {
            "dataset": title, "source_rows": len(table), "analysis_rows": len(ids),
            "exact_id_matches": int(matched.sum()), "site_fields": len(table.columns),
            "candidate_fields": len(candidates), "with_classification": int(any_label.sum()),
            "without_classification": int((~any_label).sum()),
            "missing_pct": round(float((~any_label).mean()) * 100, 4),
            "absence_type": "field absent (structural)" if not candidates else "field present",
        }
        rows.append(report)
        detail[code] = {**report, "site_column_names": list(table.columns),
                        "classification_columns": candidates, "counts_by_field": labels,
                        "source_path": rel.as_posix(), "sample_join_key": key,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output = {
        "scope": "ICRAF–ISRIC and LUCAS–Woodwell analytical samples; NeoSpectra not audited",
        "note": ("USDA-suffixed depth and layer sequence fields describe layer metadata, "
                 "not USDA Soil Taxonomy. Land-use fields are not soil taxonomic classes. "
                 "Absence applies to the OSSL v1.3 inputs examined, not to every source archive."),
        "outer_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "datasets": detail,
    }
    pd.DataFrame(rows).to_csv(args.out / "soil_classification_audit.csv", index=False)
    (args.out / "soil_classification_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
