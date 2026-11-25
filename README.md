# Selective soil spectral acquisition: reproducibility package

The [R11 update](r11/README.md) adds the five-fold Cubist sensitivity analysis, fusion-weight audit, and OSSL soil-classification metadata checks.

The [R10 update](r10/README.md) adds fold-local M5 component selection, ICRAF–ISRIC whole-country validation with optimized NIR–MIR fusion, NIR-only ranking baselines, and publication-ready aggregate tables. The original v1.0.0 analysis is retained below. No source spectra or sample-level reference measurements are redistributed.

## Sources and reuse

* NeoSpectra v1.2: Sanderman et al. (2024), [10.5281/zenodo.13122321](https://doi.org/10.5281/zenodo.13122321), CC BY 4.0; study description Partida et al. (2025), [10.1016/j.dib.2024.111229](https://doi.org/10.1016/j.dib.2024.111229).
* OSSL v1.3: Safanelli et al. (2026), [10.5281/zenodo.23815431](https://doi.org/10.5281/zenodo.23815431); OSSL paper [10.1371/journal.pone.0296545](https://doi.org/10.1371/journal.pone.0296545).
* ICRAF–ISRIC: World Agroforestry and ISRIC (2021), [10.34725/DVN/MFHA9C](https://doi.org/10.34725/DVN/MFHA9C), CC BY 4.0; Terhoeven-Urselmans et al. (2010), [10.2136/sssaj2009.0218](https://doi.org/10.2136/sssaj2009.0218). The supplied profile mapping matches 4,073 layer identifiers to 718 original profiles; it contains no geographic coordinates. It is a derived identifier mapping, not the entire original archive.
* LUCAS–Woodwell: [OSSL source data card](https://docs.soilspectroscopy.org/datasets/LUCAS.html) specifies spatial anonymization for redistribution. This package therefore excludes all LUCAS sample-level records, grid identifiers, and spectra. LUCAS partitions can be reconstructed locally from the obtained original source files; review the original use terms before any sharing.
* Swiss Agroscope: Metzger et al. (2024), [10.5281/zenodo.11204174](https://doi.org/10.5281/zenodo.11204174); the separate supplementary analysis is not part of the three primary NIR–MIR libraries.

## Reconstruct the primary analysis

Run from the repository root using Python 3. The historical analysis scripts retain their original names for reproducibility. Dependency names are supplied; the precise original environment lock has not been recovered. The assembled package has been checked for syntax, source paths, and reference outputs, but has not been rerun in full from downloaded sources. The figure script requires Arial or Helvetica. If original source files are already available locally, place them at the paths printed by `download_sources.py --plan` and extract the four parquet tables for each OSSL subset; no new download is necessary.

```bash
python -m venv .venv
# Activate the environment using the command appropriate to your operating system.
python -m pip install -r requirements.txt
python download_sources.py --plan
python download_sources.py --dataset neospectra
python download_sources.py --dataset icraf
python download_sources.py --dataset lucas
python stage2/scripts/01_single_device.py --root .
python manuscript_study/stage2_final/scripts/T00_prepare_final.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B00_corrected_nested_fusion.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/F01_m5_fix.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B01_cost_frontier.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B02_nested_cost_policy.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B03_threshold_policy.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B04_depth_gain_policy.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/F02_lucas_loco_strict.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/F03_gain_mechanism_final.py --root manuscript_study
python manuscript_study/stage2_batch2/scripts/B07_spectral_ablation.py --root manuscript_study --rand 30
python manuscript_study/stage2_final/scripts/T05_lucas_spatial.py --root manuscript_study
python prepare_figure_inputs.py
python soil_r5_figures.py --source manuscript_study --out reproduced_figures
```

`published_results/` contains the archived aggregate reference tables; `validation_partitions/` contains NeoSpectra and ICRAF partition records without measured SOC, country, or depth. Compare the regenerated partitions and summary outputs to these references; do not upload generated sample-level LUCAS files. The `audit/` directory preserves the supplementary ICRAF and Agroscope evaluation scripts. To run them, copy the four original ICRAF parquet tables to `audit/icraf_data/` locally and use `python download_sources.py --dataset agroscope`; run `icraf_profile_validation.py`, `icraf_profile_validation.py --same-coord-cohort`, `icraf_profile_validation.py --country`, `fixed_fusion_audit.py`, and `swiss_lab_qc.py` from `audit/`.

Cost parameters are normalized scenario assumptions, not measured laboratory expenditures. Random allocation is an exact matched-budget expectation; retrospective optimum curves use reference errors and are not deployable policies.
