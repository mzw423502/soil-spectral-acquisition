# R10 computation summary

Completion: outer folds 15/15; country folds 37/37. Partial outputs must not be interpreted as full-dataset estimates.

## 1. M5 implementation audit

B00 fitted one global standard deviation per modality block, then divided the independently SNV + SG(11,2,1) preprocessed NIR and MIR blocks by their training-derived factors. The original F01 outer validation scaler excluded the outer test set, but `pick_nc_unscaled(A,...)` received A scaled using its entire calibration partition, so the tuning-validation spectra contributed to each scaling factor. The same issue is present at the sub-inner selection stage. B00 also used the default scale=True for joint-component tuning; F01 corrected this switch but not the fold-local scaler. R10 recomputes both factors within every tuning split and uses scale=False throughout M5. The original global-SD definition is retained; it equalizes block RMS approximately, not total Frobenius block energy. Component candidates remain 2,4,8,12,16,20 and grouped three-fold CV minimizes untruncated mean fold MSE. Predictions are clipped only after model fitting as in R9. No outer labels were used to tune.

The component-path acceleration was checked against independently fitted sklearn models on real spectral subsets (see `pls_path_equivalence.csv`). The outer fitting uses individually fitted selected-component models. Per-fit tuning scores and scaling factors are in the checkpoint CV tables.

Path validation: 54 comparisons; maximum absolute prediction discrepancy 4.26325641456e-14.

| dataset | model | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| neospectra | M2 | 0.58319122 | 1.02284559 | 0.86348887 |
| neospectra | M4 | 0.52909329 | 1.12373212 | 0.83523176 |
| neospectra | M5 | 0.61597258 | 1.08838271 | 0.84543499 |
| icraf | M2 | 0.64012832 | 1.44976419 | 0.70265620 |
| icraf | M4 | 0.58469082 | 1.52388692 | 0.67147409 |
| icraf | M5 | 0.67653061 | 1.50423597 | 0.67989233 |
| lucas | M2 | 1.77903371 | 3.35262275 | 0.75724551 |
| lucas | M4 | 1.68983777 | 3.24681145 | 0.77232674 |
| lucas | M5 | 1.67762952 | 2.82283766 | 0.82790443 |
| neospectra | M5_r9 | 0.61597258 | 1.08838271 | 0.84543499 |
| icraf | M5_r9 | 0.67653061 | 1.50423597 | 0.67989233 |
| lucas | M5_r9 | 1.67762952 | 2.82283766 | 0.82790443 |

## 2. ICRAF–ISRIC leave-one-country-out M4

Country is restored by exact sample identifier from OSSL v1.3. All countries with >=30 samples are targets; all other countries, including small countries, are calibration data. Each target fold fits independent country-grouped inner/sub-inner CV and selects w using strict calibration OOF predictions on the 0.05 grid. Top floor(20% × n) selected by predicted SOC with sample-ID tie-break. Random comparator is the exact expectation at the same fold-specific quota. Bootstrap resamples countries and uses pooled sample weighting (3000, seed 20261125).

**COMPLETE:** 37/37 countries; n=3370/3370; 659 selected; 33/37 improved against random.

| strategy | MAE | CI_low | CI_high |
| --- | --- | --- | --- |
| priority | 0.67544901 | 0.54032524 | 0.84746739 |
| random_expectation | 0.77459358 | 0.62367142 | 0.95201195 |
| M4 | 0.63227047 | 0.50466116 | 0.79309586 |

| contrast | delta | CI_low | CI_high |
| --- | --- | --- | --- |
| priority_minus_random_expectation | -0.09914457 | -0.13217114 | -0.06372234 |
| priority_minus_M4 | 0.04317855 | 0.01410676 | 0.08058045 |

## 3. RPIQ/RPD (Table S2)

IQR uses linear empirical quantiles; SD uses the sample standard deviation (ddof=1). All predictions paired to reference SOC by dataset and sample ID. M1–M4 retain R9 predictions; M5 is R10-corrected.

| dataset | model | n | Q25 | Q75 | IQR | SD_sample | MAE | RMSE | RPIQ | RPD | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neospectra | M1 | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 | 0.93673885 | 2.04878991 | 1.00337280 | 1.35157033 | 0.45229997 |
| neospectra | M2 | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 | 0.58319122 | 1.02284559 | 2.00978532 | 2.70723525 | 0.86348887 |
| neospectra | M3 | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 | 0.58879956 | 1.36388488 | 1.50723870 | 2.03029133 | 0.75728124 |
| neospectra | M4 | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 | 0.52909329 | 1.12373212 | 1.82935063 | 2.46418483 | 0.83523176 |
| neospectra | M5 | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 | 0.61597258 | 1.08838271 | 1.88876581 | 2.54421869 | 0.84543499 |
| icraf | M1 | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 | 0.80175553 | 2.09302940 | 0.46822087 | 1.27043171 | 0.38025247 |
| icraf | M2 | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 | 0.64012832 | 1.44976419 | 0.67597203 | 1.83412651 | 0.70265620 |
| icraf | M3 | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 | 0.60352071 | 1.63789838 | 0.59832775 | 1.62345294 | 0.62047704 |
| icraf | M4 | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 | 0.58469082 | 1.52388692 | 0.64309237 | 1.74491355 | 0.67147409 |
| icraf | M5 | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 | 0.67653061 | 1.50423597 | 0.65149356 | 1.76770864 | 0.67989233 |
| lucas | M1 | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 | 2.47583331 | 4.21738760 | 0.53054169 | 1.61486975 | 0.61586408 |
| lucas | M2 | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 | 1.77903371 | 3.35262275 | 0.66738793 | 2.03140412 | 0.75724551 |
| lucas | M3 | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 | 1.78473691 | 3.31866702 | 0.67421647 | 2.05218892 | 0.76213790 |
| lucas | M4 | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 | 1.68983777 | 3.24681145 | 0.68913763 | 2.09760616 | 0.77232674 |
| lucas | M5 | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 | 1.67762952 | 2.82283766 | 0.79264209 | 2.41265439 | 0.82790443 |

## 4. Uncertainty and spectral-distance baselines

U1: SD (ddof=0) across three NIR-only inner calibration models with independently tuned component counts. U2: Euclidean distance in training-whitened PCA score space, using 20 PCs fixed a priori (or fewer nonzero PCs); no SOC/MIR labels/features used in PCA ranking. All policies allocate identical floor-rounded quotas within the original held-out folds; post-acquisition M4 predictions identical. Random is exact expected MAE. Group bootstrap follows R9 (3000, seed 20261125). Report conditional paired differences, negative means SOC priority better.

| dataset | budget | n | k | strategy | MAE | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neospectra | 0.20000000 | 1974 | 394 | SOC | 0.73577549 | 0.68891981 | 0.78573156 |
| neospectra | 0.20000000 | 1974 | 394 | U1 | 0.78851266 | 0.73642092 | 0.84528108 |
| neospectra | 0.20000000 | 1974 | 394 | U2 | 0.83788125 | 0.77544273 | 0.90787036 |
| neospectra | 0.20000000 | 1974 | 394 | random | 0.85534730 | 0.78575181 | 0.93221944 |
| icraf | 0.20000000 | 3698 | 738 | SOC | 0.63612324 | 0.57607434 | 0.70114342 |
| icraf | 0.20000000 | 3698 | 738 | U1 | 0.69078899 | 0.62194882 | 0.76521460 |
| icraf | 0.20000000 | 3698 | 738 | U2 | 0.72900854 | 0.65775733 | 0.80474827 |
| icraf | 0.20000000 | 3698 | 738 | random | 0.75842760 | 0.67993386 | 0.84169802 |
| lucas | 0.20000000 | 572 | 112 | SOC | 2.00966495 | 1.76334643 | 2.27648849 |
| lucas | 0.20000000 | 572 | 112 | U1 | 2.02246348 | 1.78394333 | 2.28932740 |
| lucas | 0.20000000 | 572 | 112 | U2 | 2.29281570 | 2.00495075 | 2.62071590 |
| lucas | 0.20000000 | 572 | 112 | random | 2.32248195 | 2.04972797 | 2.63771378 |

| dataset | budget | k | comparison | delta_MAE | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- |
| neospectra | 0.20000000 | 394 | SOC_minus_U1 | -0.05273717 | -0.08044469 | -0.02676069 |
| neospectra | 0.20000000 | 394 | SOC_minus_U2 | -0.10210576 | -0.14156235 | -0.06735586 |
| neospectra | 0.20000000 | 394 | SOC_minus_random | -0.11957181 | -0.15664522 | -0.08586687 |
| icraf | 0.20000000 | 738 | SOC_minus_U1 | -0.05466575 | -0.07895341 | -0.03237104 |
| icraf | 0.20000000 | 738 | SOC_minus_U2 | -0.09288530 | -0.12048127 | -0.06624607 |
| icraf | 0.20000000 | 738 | SOC_minus_random | -0.12230436 | -0.15363668 | -0.09123357 |
| lucas | 0.20000000 | 112 | SOC_minus_U1 | -0.01279853 | -0.14984951 | 0.12352491 |
| lucas | 0.20000000 | 112 | SOC_minus_U2 | -0.28315076 | -0.47682691 | -0.11275828 |
| lucas | 0.20000000 | 112 | SOC_minus_random | -0.31281701 | -0.48816182 | -0.15689967 |

All 21 budgets (0–100%) and intervals are in the CSV files.

## Reproducibility

Python 3.12.10, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0.
No manuscript or previously published R9 files overwritten.

## Final numerical audit / 最终核验

All 15 outer folds and all 37 country folds completed. No published R9 files have been changed.

### M5 old versus corrected predictions

| dataset | n | max_abs_prediction_change | old_MAE | new_MAE | MAE_change | M2_MAE | M4_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neospectra | 1974 | 0.00000000 | 0.61597258 | 0.61597258 | -0.00000000 | 0.58319122 | 0.52909329 |
| icraf | 3698 | 0.00000000 | 0.67653061 | 0.67653061 | 0.00000000 | 0.64012832 | 0.58469082 |
| lucas | 572 | 0.00000000 | 1.67762952 | 1.67762952 | 0.00000000 | 1.77903371 | 1.68983777 |

F01 original selection logs recovered from `F01.log`, matching R10 by dataset and fold.

| dataset | outer_fold | R9_M5_ncomp | M5_ncomp | M5_inner_ncomp | M5_sn | M5_sm | same_ncomp | M1_max_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icraf | 1 | 20 | 20 | [16, 12, 20] | 0.03620248 | 0.01913749 | True | 0.00000000 |
| icraf | 2 | 12 | 12 | [12, 16, 12] | 0.03606260 | 0.01906987 | True | 0.00000000 |
| icraf | 3 | 16 | 16 | [12, 16, 12] | 0.03607215 | 0.01905581 | True | 0.00000000 |
| icraf | 4 | 20 | 20 | [12, 16, 12] | 0.03640683 | 0.01899102 | True | 0.00000000 |
| icraf | 5 | 20 | 20 | [16, 16, 12] | 0.03588778 | 0.01915439 | True | 0.00000000 |
| lucas | 1 | 12 | 12 | [12, 12, 12] | 0.03019528 | 0.01614049 | True | 0.00000000 |
| lucas | 2 | 8 | 8 | [12, 8, 8] | 0.03013829 | 0.01602815 | True | 0.00000000 |
| lucas | 3 | 8 | 8 | [12, 12, 12] | 0.03027474 | 0.01611508 | True | 0.00000000 |
| lucas | 4 | 8 | 8 | [8, 8, 12] | 0.02983248 | 0.01611681 | True | 0.00000000 |
| lucas | 5 | 8 | 8 | [16, 8, 8] | 0.02999650 | 0.01603796 | True | 0.00000000 |
| neospectra | 1 | 20 | 20 | [20, 20, 20] | 0.05667990 | 0.01592047 | True | 0.00000000 |
| neospectra | 2 | 20 | 20 | [20, 20, 20] | 0.05650373 | 0.01592566 | True | 0.00000000 |
| neospectra | 3 | 20 | 20 | [20, 12, 20] | 0.05656415 | 0.01590878 | True | 0.00000000 |
| neospectra | 4 | 20 | 20 | [20, 16, 20] | 0.05669639 | 0.01595299 | True | 0.00000000 |
| neospectra | 5 | 20 | 20 | [20, 8, 20] | 0.05666255 | 0.01593325 | True | 0.00000000 |

All 15/15 final component choices match. Maximum sample-level M5 prediction difference = 1.56319401867e-13. M5 is worse in pooled MAE than MIR-only M2 in NeoSpectra and ICRAF under the retained model definition; the fold-local scaling correction does not explain these differences. This does not exclude better alternative joint-block scaling or other models.

### Corrected Table S2 with bias

| dataset | model | MAE | RMSE | R2 | bias | RPIQ | RPD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neospectra | M1 | 0.93673885 | 2.04878991 | 0.45229997 | 0.07624635 | 1.00337280 | 1.35157033 |
| neospectra | M2 | 0.58319122 | 1.02284559 | 0.86348887 | 0.05379563 | 2.00978532 | 2.70723525 |
| neospectra | M3 | 0.58879956 | 1.36388488 | 0.75728124 | 0.06502099 | 1.50723870 | 2.03029133 |
| neospectra | M4 | 0.52909329 | 1.12373212 | 0.83523176 | 0.05940831 | 1.82935063 | 2.46418483 |
| neospectra | M5 | 0.61597258 | 1.08838271 | 0.84543499 | 0.06425025 | 1.88876581 | 2.54421869 |
| icraf | M1 | 0.80175553 | 2.09302940 | 0.38025247 | 0.08206395 | 0.46822087 | 1.27043171 |
| icraf | M2 | 0.64012832 | 1.44976419 | 0.70265620 | 0.13273824 | 0.67597203 | 1.83412651 |
| icraf | M3 | 0.60352071 | 1.63789838 | 0.62047704 | 0.10740109 | 0.59832775 | 1.62345294 |
| icraf | M4 | 0.58469082 | 1.52388692 | 0.67147409 | 0.11769726 | 0.64309237 | 1.74491355 |
| icraf | M5 | 0.67653061 | 1.50423597 | 0.67989233 | 0.14173878 | 0.65149356 | 1.76770864 |
| lucas | M1 | 2.47583331 | 4.21738760 | 0.61586408 | 0.40913474 | 0.53054169 | 1.61486975 |
| lucas | M2 | 1.77903371 | 3.35262275 | 0.75724551 | 0.14126349 | 0.66738793 | 2.03140412 |
| lucas | M3 | 1.78473691 | 3.31866702 | 0.76213790 | 0.27519912 | 0.67421647 | 2.05218892 |
| lucas | M4 | 1.68983777 | 3.24681145 | 0.77232674 | 0.19262787 | 0.68913763 | 2.09760616 |
| lucas | M5 | 1.67762952 | 2.82283766 | 0.82790443 | 0.18240327 | 0.79264209 | 2.41265439 |

Reference SOC statistics:

| dataset | n | Q25 | Q75 | IQR | SD_sample |
| --- | --- | --- | --- | --- | --- |
| neospectra | 1974 | 0.64999998 | 2.70570004 | 2.05570006 | 2.76908364 |
| icraf | 3698 | 0.22000000 | 1.20000005 | 0.98000005 | 2.65905093 |
| lucas | 572 | 1.10750002 | 3.34499997 | 2.23749995 | 6.81053167 |

### U1/U2 interpretation at 20%

| dataset | budget | k | comparison | delta_MAE | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- |
| neospectra | 0.20000000 | 394 | SOC_minus_U1 | -0.05273717 | -0.08044469 | -0.02676069 |
| neospectra | 0.20000000 | 394 | SOC_minus_U2 | -0.10210576 | -0.14156235 | -0.06735586 |
| neospectra | 0.20000000 | 394 | SOC_minus_random | -0.11957181 | -0.15664522 | -0.08586687 |
| icraf | 0.20000000 | 738 | SOC_minus_U1 | -0.05466575 | -0.07895341 | -0.03237104 |
| icraf | 0.20000000 | 738 | SOC_minus_U2 | -0.09288530 | -0.12048127 | -0.06624607 |
| icraf | 0.20000000 | 738 | SOC_minus_random | -0.12230436 | -0.15363668 | -0.09123357 |
| lucas | 0.20000000 | 112 | SOC_minus_U1 | -0.01279853 | -0.14984951 | 0.12352491 |
| lucas | 0.20000000 | 112 | SOC_minus_U2 | -0.28315076 | -0.47682691 | -0.11275828 |
| lucas | 0.20000000 | 112 | SOC_minus_random | -0.31281701 | -0.48816182 | -0.15689967 |

- neospectra: SOC_minus_U1: interval excludes zero in favor of SOC; SOC_minus_U2: interval excludes zero in favor of SOC; SOC_minus_random: interval excludes zero in favor of SOC.
- icraf: SOC_minus_U1: interval excludes zero in favor of SOC; SOC_minus_U2: interval excludes zero in favor of SOC; SOC_minus_random: interval excludes zero in favor of SOC.
- lucas: SOC_minus_U1: interval crosses/includes zero; SOC_minus_U2: interval excludes zero in favor of SOC; SOC_minus_random: interval excludes zero in favor of SOC.

The M5 correction leaves M4-based policy comparisons unaffected. RPIQ and RPD both use RMSE, so their within-library rankings are the same as RMSE rankings (not MAE rankings). No claim of universal superiority over U1 is made where its interval includes zero.

Full output files: `M5_predictions_R10.csv`, `M5_U1_outer_fold_audit.csv`, `all_nested_cv_scores.csv`, `ICRAF_LOCO_M4_country.csv`, `ICRAF_LOCO_M4_predictions.csv`, `ICRAF_LOCO_M4_summary.json`, `Table_S2_R10.csv`, `Table_S2_R10.docx`, `U1_U2_budget_curves.csv`, `U1_U2_paired_bootstrap.csv`, `source_input_sha256.csv`, and `R10_verification.json`.
