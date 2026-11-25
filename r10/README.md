# R10 reproducibility update

This update adds the analysis scripts and aggregate results for the R10 manuscript.
The original analysis and its data-source instructions remain in the repository's
root README. The previous v1.0.0 release is a separate historical snapshot.

- **M5:** recalculates block scaling within each calibration fold when selecting
  the component count for joint-spectrum partial least squares regression.
- **ICRAF–ISRIC country holdouts:** recalibrates both component counts and optimized
  NIR–MIR fusion weights using only non-target countries, then evaluates matched
  acquisition budgets in the 37 held-out countries.
- **U1 and U2:** compares first-stage SOC ranking against NIR submodel prediction
  disagreement and training-calibrated NIR spectral distance, respectively.
- **Model evaluation:** supplies pooled RPIQ and RPD for all five models in the
  three libraries, with the SOC distribution summaries used in their denominators.

## Use the publication tables

Start with `R10_computation_summary.md` and `R10_verification.json`.
`Table_S2_R10.csv` contains the supplementary model table.
`ICRAF_LOCO_M4_country.csv` and `ICRAF_LOCO_M4_summary.json` contain country and
pooled results. The two `U1_U2_*.csv` tables contain the budget curves and
paired cluster bootstrap intervals.

## Rerun the analysis

Use Python 3.12 with the repository dependencies and `python-docx`. Run the
source reconstruction and model steps in the root README first (through the
corrected joint-spectrum model). Use the reconstructed local `manuscript_study`
directory as the root, with the ICRAF site parquet at
`data/icraf/ossl_soilsite_v1.3.parquet` relative to that root. The R10 finalizer
also needs the joint-spectrum fit log at `stage2_batch2/logs/F01.log`; retain or
capture the stdout of the historical joint-spectrum model command there. The
relative input paths and original checksums are recorded in
`source_input_sha256.csv`; a newly generated log may differ in encoding and
its output path while reporting the same selections.

```bash
python -m pip install python-docx
python r10/run_r10.py --root manuscript_study --out r10_local_results --workers 3
python r10/finalize_r10.py --root manuscript_study --out r10_local_results
```

The public update contains no original spectra, reference SOC at sample level,
LUCAS coordinates or spatial identifiers, or per-sample prediction files.
Keep locally generated checkpoints and sample-level tables outside the public
archive. Source-library access and reuse remain governed by the respective
data-use terms. The published summaries do not replace the source data needed
to rerun the models.
