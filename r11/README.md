# R11: Cubist sensitivity analysis and soil-classification audit

This directory adds the ICRAF–ISRIC Cubist sensitivity analysis (Supplementary
Table S11) and the OSSL site-table classification audit (Supplementary Table
S10). The primary analysis and its source-data instructions are unchanged.

## Results and checks

- `results/cubist_summary.json`: pooled metrics, profile-cluster bootstrap
  intervals, model settings and input checksums.
- `results/cubist_icraf_folds.csv`: five outer-fold summaries, including the
  exact acquisition quotas, selected NIR fusion weights and comparator errors.
- `results/cubist_icraf_M4_calibration.csv`: the 21 candidate fusion weights
  per outer fold and their inner out-of-fold MAEs.
- `results/fixed_fold_PLSR_validation*.{json,csv}`: checks and metrics for the
  historical **PLSR** comparator evaluated on the same frozen partitions.
- `results/soil_classification_audit.{json,csv}`: availability of soil
  taxonomic fields in the ICRAF–ISRIC and LUCAS–Woodwell analytical samples.
  Neither examined OSSL site table contains a classification field; this
  does not establish absence of taxonomy in other archives.

The Cubist run pooled 3,698 held-out samples from 670 profiles. At 738
acquisitions, predicted-SOC priority reduced MAE from 0.6272 to 0.4869 SOC
percentage points compared with the exact matched-budget random expectation.
The deposited results were checked against the local outer-fold predictions
using the verification script below, including the fusion weights, quotas
and both negative-gain counts.

## Reproduce locally

Use Python 3.12 and the [repository dependencies](../requirements.txt).
Install `cubist==1.2.2` separately. Follow the [source reconstruction
instructions](../README.md#reconstruct-the-primary-analysis) through the
corrected nested PLSR analysis. For exact verification of the historical run,
compare the SHA-256 fingerprints in `results/cubist_summary.json` with the
local frozen inputs under `manuscript_study`:

- `stage2_final/manifests/data_icraf.npz`
- `stage2_final/manifests/outer_split_manifest.csv`
- `stage2_final/manifests/inner_split_manifest.csv`

The PLSR validation step additionally reads
`stage2_batch2/predictions/all_corrected_predictions.csv`. The classification
audit also requires the ICRAF and LUCAS site parquets under
`data/icraf/` and `data/lucas/`, respectively. Source files are obtained or
reconstructed locally; the public aggregate outputs do not substitute for
those inputs. Load NPZ inputs only if obtained from a trusted source.

Run from the repository root in an activated Python environment:

```sh
python -m pip install cubist==1.2.2
python r11/run_icraf_cubist.py --root manuscript_study --out r11_local_results --validate-inputs
python r11/run_icraf_cubist.py --root manuscript_study --out r11_local_results --workers 2
python r11/verify_cubist_r11.py --out r11_local_results
python r11/audit_ossl_classification.py --root manuscript_study --out r11_local_results
```

The verifier checks the pooled outputs from the locally generated held-out
predictions. It cannot perform that check with the public summaries alone.
Bootstrap intervals are conditional on the fitted predictions and allocation
decisions. Keep local outputs, per-sample reference values, predictions and
fold checkpoints out of public commits; only aggregate results are deposited
here. Source-library reuse remains subject to the original terms.
