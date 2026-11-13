# Details of the code and data

This page holds the details behind the [first page](../README.md). Paths are relative to the root of
the repository.

## What the paper computes

The paper builds a scenario bank of an all-inverter IEEE 39-bus system. It labels each scenario with
a collapse margin from its full-order small-signal model. On this bank it trains a glass-box monitor
(AMBER, an additive margin model) and the comparison models. It then checks the monitor's explanations
against the full-order model's sensitivities. Last, it turns the predictions into three risk tiers.

## Figures

Figs. 1 to 7 are diagrams drawn in Figma for the paper. They are not reproduced here, and no step
draws them. The values they show are in the files here:

- Fig. 2: the -4.7 s^-1 tick is the bank's smallest stable-side margin (`data/dataset.csv`, column
  `sigma_dom`).
- Fig. 3: it prints k* = 1.9730 (`results/feasibility/fs3_boundary.csv`). Its inset draws the runs at
  k = 1.96, 1.97 and 2.00 of `results/final/e4_straddle.mat` (the terminal voltages `Vterm` of the ten
  plants).
- Fig. 4: the three shapes (`V4`, `NGFM` and `fpll,10`) are the thumbnails that `thumbnails()` of
  `python/r24f_gallery.py` draws from the deployed M+D model at seed 0.
- Fig. 6: the tier edges are in `results/final/tables/e1_tier_edges.csv`.

The figure modules `python/e7_r8_figs.py` and `python/r17f_figs.py` to `python/r24f_*.py` are
libraries that the Fig. 8 and Fig. 9 scripts import. No step runs their own figure functions. The
names of these functions (fig5a_..., fig11_..., fig13_...) follow an earlier figure numbering.

The paper also embeds every figure as a vector SVG. Figs. 8 and 9 are reproduced as PNG and as SVG.
The steps `fig8` and `fig9` write the PNG and a vector PDF into `results/figures/`, a folder that
this repository does not ship. The step
`fig-vector` (`python/fig_vector.py`) turns the two PDFs into the paper's SVGs,
`results/svg/r29_fig8_fidelity_h160.svg` and `results/svg/r29_fig9_glassbox.svg`. It uses PyMuPDF
1.28.0, pinned in `requirements.txt`. It writes text as paths and masks as clip paths. It also scales
the user space by 8, so that Word keeps every stroke width.

## Software

### MATLAB

- MATLAB R2026a with [MATPOWER](https://matpower.org) 8.1 on the MATLAB path. The code uses `runpf`,
  `mpoption`, `makeYbus` and `loadcase`.
- The Statistics and Machine Learning Toolbox. The code uses `lhsdesign` (in `build_bank.m`,
  `r18_freeze.m`), `randsample` (in `build_bank.m`, `r17_rays.m`, `r18_freeze.m`) and `quantile` (in
  `q1_hbar_drift.m`, `r16_cp_relabel.m`, `cp30_relabel_merge.m`).
- The Parallel Computing Toolbox. `parpool` and `gcp` open a pool of workers for the `parfor` loops.
  `build_bank.m` and `e2_sens_production.m` open four workers (`e2_sens_production.m` uses the
  number in `E2_POOL` if it is set). `r18_freeze.m` opens three. `e4_emt_production.m` opens none.
- The `matlab` command on the system PATH: `run_all.py` runs the MATLAB steps as `matlab -batch`.

The run logs of the computation name MATLAB 26.1.0.3276743 (R2026a) Update 3 with MATPOWER 8.1. They
do not record the versions of the two toolboxes.

The device-data probes and the 30% constant-power labels were computed on a computing cluster with
MATLAB 26.1.0.3346908 (R2026a) Update 5 and MATPOWER 8.1. Their job logs record this, and these logs
are not shipped. Before the 30% labels, the same code relabeled the bank at 60% on the cluster. It
reproduced `results/tables/r16/r16_cp_relabel.csv`. The label and the class were the same on all 7,324
rows, and the 3,310 flips were the same. The decay rate agreed to a relative difference of at most
1.87e-7. The output of this check is `results/cp30/cp_relabel_gate_060.txt`, with the time stamp of
the run.

### Python

Python 3.13 and the packages in `requirements.txt`. Each package is pinned to the version the results
were computed with. The timing file `results/final/tables/e6_timing_python.csv` records Python
3.13.9, and `results/tables/r21/r21_design_baseline.json` records 3.13.15. Install the packages with

```
pip install -r requirements.txt
```

The pinned versions are:

```
interpret==0.7.8
matplotlib==3.10.7
numpy==2.3.5
optuna==4.6.0
pandas==2.3.3
pillow==12.0.0
psutil==7.2.2
pygam==0.12.0
pymupdf==1.28.0
scikit-learn==1.7.2
scipy==1.16.3
shap==0.50.0
statsmodels==0.14.6
torch==2.9.1
xgboost==3.3.0
```

`check_paper_values.py` needs only the Python standard library.

## Checking the numbers

```
python check_paper_values.py
```

The script compares the result files in `results/` with the numbers printed in Tables II to V and
with 130 numbers printed in the text. That is 293 checks in all. A number marked "(derived)" is
computed from the shipped files, by the operation that `derived()` in the script states. The script
ends with a line such as

```
293 checks, 0 failed
```

The tolerance is half a unit of the last printed digit. Where a table prints a fraction as a
percentage, the check multiplies the file's value by 100.

## Running everything

```
python run_all.py --list
python run_all.py --all
```

`run_all.py` runs the 77 steps that `python run_all.py --list` prints, in that order, and ends with
`check_paper_values.py`. MATLAB steps run in `matlab/` and Python steps in `python/`. One step runs
with `python run_all.py STEP`, and a step and all later ones with `python run_all.py --from STEP`.
Every step writes into `results/` (the bank into `data/`), over the files there.

The listing marks some outputs "[written by a rerun, not shipped]". Such a step writes files that this
repository does not hold (work files and large intermediate tables). Every other output that the
listing names by its path is shipped under `results/`. An output named in words only is one of two
kinds. It is a result file of the table below (shipped), or an intermediate file that the step writes
and this repository does not hold. Examples of the second kind are the test predictions (step
`paired`), the tier battery rows (step `battery-13`), the frozen forward bank (step `frozen-bank`)
and the fidelity trust rows (step `trust`).

The step `frames` runs `run_ml.py --frames-only`. It writes the train, test and held-out frames and
stops. Without that switch, `run_ml.py` goes on to the six-class model comparison of an earlier
analysis (see [Side analyses](#side-analyses)). The listing gives run times where they were recorded.
For the other steps, no run time was recorded.

### The step delay

The step `delay` runs `matlab -batch r17_delay`. `matlab/r17_delay.m` reads its stage from the
environment variable `DELAY_STAGE` (line 40). If it is empty, the script runs the stage `recon` (line
41), which writes only `r17_delay_recon.json`. Only the stage `main` writes
`results/tables/r17/r17_delay_summary.csv` and `r17_delay_trace.csv` (in `main_traces`). To run that
stage, use `matlab -batch "setenv('DELAY_STAGE','main'); r17_delay"` in `matlab/`.

### What was run for this release

For this release, `check_paper_values.py` and `run_all.py --list` were run on this tree. The two
figure scripts were run on a copy of it. Their figure files were byte-equal to the figures of the
paper, and `fig_vector.py`, run on their PDFs, gave the paper's two SVGs byte for byte. The step
`device-fidelity` (`python/device_probes_fidelity.py`) was also run once on a copy of this tree. It
gave the printed numbers of `results/device_probes/device_probes_fidelity.json`. The MATLAB chain and
the other training and evaluation steps were run while the results were computed, and not again on
this tree. The order of the steps follows the files that each script reads and writes. This order
was checked by reading the scripts. The chain was not run end to end in this order.

### Steps that resume

Many MATLAB steps resume from the files they find. They skip the rows, rays or scenarios that their
output already holds. For example, `cp-relabel` skips the rows of
`results/tables/r16/r16_cp_relabel.csv`, and `device-probes` skips every scenario whose file is in
`results/device_probes/scenarios/`. To compute such a result again, delete the step's shipped output
first.

### Run times

The full chain takes many hours. The full-model probes took three to five hours (four workers), and
the model suite three to four hours. The run time of the bank was not recorded. The device-data
probes (12,772 full-model solves) took 55 minutes on one 16-core cluster node, with the 16 shards in
parallel, one per core. The 30% labels, with the 60% check before them, took 20 minutes. The steps
`device-probes` and `cp30-labels` run the shards one after another. On a laptop,
`device_probes_fidelity.py` took about four minutes and `cp30_retrain.py` 12 to 14 minutes.

### Seeds

The seeds are set in the code. The master seed is 42, and the model seeds are 0 to 4. The frozen
forward bank was drawn under its own master seed, 20260812 (`FREEZE_SEED` in `matlab/r18_freeze.m`).
The cross-check of `python/r21_design_baseline.py` draws its folds with the seed 20260921
(`NB_SEED`).

## Which script produces what

In this table, "asserted" means that `check_paper_values.py` checks the number.

| In the paper | Script and the file it writes |
|---|---|
| Table II, rows M, M+D, M+D+P and M+D+P+I: RMSE, R^2, stable-side R^2 and AUC | `python/e1_margin_suite.py` writes `results/final/tables/e1_model_suite.csv` and the per-seed rows `results/final/tables/e1_model_suite_raw.csv`. Later steps read the per-seed rows, for example `cp30-retrain` and `corruption`. `python/e1_reg_metrics.py` writes `results/final/tables/e1_reg_metrics.csv` (RMSE). |
| Table II, rows M+F and DC4, and the size of the test set | `python/r21_design_baseline.py` writes `results/tables/r21/r21_design_baseline.json`. |
| Table III: screens inside each GFM count | `matlab/e9_ma2024_index.m` computes the indices per bank row. `python/e9_ma2024_index.py` writes `results/final/e9_ma2024_index.json`. The M+F row comes from `r21_design_baseline.json`. |
| Table IV: response fidelity on the applied probes | `python/r22_faith_slack.py` writes `results/final/r22_faith_slack.json` and the per-scenario file `results/final/r22_faith_slack_rows.csv`. |
| Table V: tier census | `python/e1_tiers.py` writes `results/final/tables/e1_tiers.csv`. `python/r16_census.py` writes `results/tables/r16/r16_census.csv` (deployed rule, Wilson intervals, abstentions). `python/r19_frzc.py` writes `results/tables/r19/r19_frzc_census.csv` (frozen forward bank). |
| Fig. 8 | `python/r29f_fidelity.py` draws the per-scenario values of Table IV from `results/final/r22_faith_slack_rows.csv` and `results/final/r22_faith_slack.json`. It writes `results/figures/r29_fig8_fidelity_h160.png` (the printed height) and `r29_fig8_fidelity.png`, which are not shipped. |
| Fig. 9 | `python/r29f_glassbox.py` fits the deployed M+D model at seed 0 on the training frame (about two minutes) and draws it. It writes `results/figures/r29_fig9_glassbox.png`, which is not shipped. |
| Section I: the index of [13] on the test set (AUC 0.722, minimum over the converter buses) | `matlab/r15_jvsm.m` writes `results/tables/r15/r15_jvsm_test.csv` (one row per test row). `derived()` computes the AUC of the low index against the unstable label. |
| Section I: on the bank, low gSCR ranks collapse risk at AUC 0.244 | `python/r15_battery.py` writes `results/tables/r15/r15_results.json`. The key `B/gates/gSCR_bank_vs_0.244` holds the AUC of the low gSCR on the bank. The name of the key gives the value it was checked against. |
| Section II-A: R^2 of the damping ratio, 0.014 from M and 0.629 with M+D (five-seed means) | `python/e1_margin_suite.py` writes `results/final/tables/e1_zeta_sidequest.csv`. |
| Section II-A: the +1 s^-1 cap saturates the training target on the rows above it, 13.7% of the bank (1,000 of 7,324). The smallest stable-side margin is -4.7. | Computed from `data/dataset.csv` (column `sigma_dom`) by `derived()` in `check_paper_values.py`. |
| Section II-A: refitted on the uncapped target, AUC 0.933 +/- 0.002 | `python/r15_battery.py` writes `results/tables/r15/r15_results.json` and the per-seed rows `results/tables/r15/r15_cap_ablation.csv`. The AUC is `AUC_mean` of block `L`. The +/- is `AUC_sd`, the population standard deviation over the five seeds (`tbl.AUC.std(ddof=0)` in `python/r15_battery.py`). |
| Section III-A: 80% of a row's total absolute contribution takes a median of 28 of the 88 additive terms | `python/r15_battery.py` writes `results/tables/r15/r15_results.json` (block `H`, `EXP-K`: `n_terms` 88, `k80_median` 28.0). |
| Section II-C: on one of the two bank rays, the point beyond the collapse point diverges only at 17.1 s | `matlab/r20_straddle2.m` and `matlab/r20_straddle2b.m` write `results/tables/r20/r20_straddle2.csv` (rays 8 and 11). Ray 11, case `collapse30`, has `outcome` collapse and `tdiv` 17.123 s. |
| Section II-B: 8,500 draws, 7,324 converged (86.2%), 748 power-flow non-convergences, 428 reference-bus losses, 37% of the failures against 23% of the converged draws in one-GFM configurations | `results/tables/r15/r15_failure_ledger.csv` (one row per failed draw) and `data/dataset.csv`. The writer of the ledger is not in this repository. `derived()` computes the counts and shares (8,500 = 7,324 converged rows + 1,176 ledger rows). |
| Section II-B: the label audit, N_mis = 0 over the 7,324 rows | Computed from `data/dataset.csv` (columns `class`, `zeta_min`, `sigma_dom`) by `derived()`. `matlab/audit_rows.m` is the audit of the bank generator. |
| Section II-C: k* = 1.9730 on the design ray | `matlab/fs3_boundary.m` writes `results/feasibility/fs3_boundary.csv`. The eigenvalues along the ray are in `results/final/tables/e3c_eig_align.csv` (`matlab/e3c_eig_align.m`). |
| Section II-C: the crossing pair turns real within 0.007 of k* | `matlab/q3_beyond_fold.m` writes `results/final/tables/q3_continuation.csv`. `derived()` takes the first step with a real unstable mode (k = 1.980, `f_dom` = 0) minus k* of `fs3_boundary.csv`. |
| Section III-A: 5,626 multi-GFM rows, Spearman +0.60 of H-bar with the worst-case erosion | `matlab/q1_hbar_drift.m` and `matlab/q1_hbar_drift_b.m` write `results/final/tables/q1_hbar_drift.csv` and `q1_hbar_drift_b.csv`. `derived()` keeps the rows with `trace_ok` = 1 and two or more GFM plants, one row per id. It computes the Spearman correlation of `Hbar_row` with `ero10 / lam_min_row`. |
| Section III-A: the 64 features of the monitor, twelve of them zero throughout the training pool | `python/r13_battery.py` writes `results/tables/r13/r13_results.json`. `task5/n_main_terms` is 64. `task10/implied_structural_zero_features` holds the twelve names: Hv and Dv of slots 1, 2, 3, 5, 7 and 8. |
| Section II-C: the constant-power ray of the load-model bracket (1.973 to 0.964). Section IV-D: k* = 1.415 with 30% of every inductive load as smoothed constant power. | `matlab/q7_cpl_ray.m` writes `results/final/tables/q7_cpl_ray.csv` (column `kstar`). The rows are alpha = 0, and alpha = 0.30 and 0.60 with tau = 0.020. The three are asserted. |
| Section II-C: 3,310 of 7,324 labels flip (45.2%) | `matlab/r16_cp_relabel.m` writes `results/tables/r16/r16_cp_relabel.csv` (column `flip_bin`). |
| Section III-A: GRIP against the smallest eigenvalue (+0.019, DeLong p = 2.7e-13) | `python/r20_cheap.py` writes `results/tables/r20/r20_cheap.json` (block `dlg`). |
| Section III-B and Fig. 6: tier edges -3.631, -1.870 and the candidate -2.933 | `python/e1_tiers.py` writes `results/final/tables/e1_tier_edges.csv`. |
| Section IV-A: ensembles, XGBoost 0.0246 (DeLong p 1.6e-7) and FTT-reg 0.0294 | `python/e8_paired_test.py` writes `results/final/tables/e8_paired_frontier.csv`. `python/e2r17_faith.py` writes `results/tables/r17/e2r17_fttp_paired.csv`. |
| Section IV-A: resampling the 77 configuration cells of the test frame, not its rows, widens AMBER's AUC interval to 0.88 to 0.96 | `python/r20_cheap.py` writes `results/tables/r20/r20_cheap.json` (block `clus`: `n_clusters` 77, `ci_lo` 0.878, `ci_hi` 0.958). |
| Section IV-A: across ten resplits, 0.0199 (sd 0.0038) and 0.0145 (sd 0.0086) | Computed from `results/tables/r15/r15_repeated_splits_raw.csv` (`python/r15_battery.py`). `python/r19_fttr.py` writes `results/tables/r19/r19_fttr_summary.csv` and `r19_fttr_raw.csv`. |
| Section IV-A: the line-impedance scale added (+0.003 AUC, DeLong p = 0.25) | `python/r21_design_baseline.py` writes `results/tables/r21/r21_design_baseline.json` (`item8/paired_MDk_vs_MD`: `dAUC` of the ensembles 0.00337 and `delong_p`). |
| Section IV-A: the flags arm's stable-side R^2 gap, +0.15 [+0.10, +0.20], and its 58 Secure test rows against 198 | `python/r19_ssci.py` writes `results/tables/r19/r19_ssci_gap.csv`. `python/r17_mfl.py` writes `results/tables/r17/r17_mfl_tiers.csv` (frame `test`, tier 0). The 198 is in `results/final/tables/e1_tiers.csv`. |
| Section IV-A: the threshold of [9] against this paper's admittance (median cosine 0.27) | `matlab/e9_ma2024_threshold.m` computes the rows. `python/e9_ma2024_threshold.py` writes `results/final/e9_ma2024_threshold.json`. |
| Section IV-A: the AUC differences at the EMS configuration, XGBoost minus AMBER and FTT-reg minus AMBER (0.0237 and 0.0289) | `python/r18_battery.py` writes `results/tables/r18/r18_pidx.csv`. `python/r18_fttp_pidx.py` writes `r18_fttp_pidx.csv`. |
| Section IV-A: reactive channel at its training medians (AUC 0.421, recall 28.6%) | `python/r15_battery.py` writes `results/tables/r15/r15_dropout_ablation.csv` (group `Qinv`). |
| Section IV-A: AMBER on the frozen forward bank: AUC 0.930, and its seed-0 alarm there, 94.6% recall at 36.0% false alarms | `python/r18_freeze_score.py` writes `results/tables/r18/r18_freeze_score.csv`. The 0.930 is the mean of `AUC_freeze` over the five M+D `ebm` rows, computed by `derived()`. The row `paired_seed_mean` holds 0.9303. The seed-0 row holds `recall_freeze` and `FA_freeze`. |
| Section IV-A: noisy inputs at the frozen threshold (recall 96.9%, false alarms 52.6%) | `python/r15_battery.py` writes `results/tables/r15/r15_noise_alarm_rows.csv`. `derived()` takes the means of `recall` and `FA` over the five `ebm` draws. |
| Section IV-B: the probe-feasible interior (313 attempted, 73 and 4 removed, 236 kept, 25 unstable-side) | `python/r21_faith_md_side.py` writes `results/tables/r21/r21_faith_md_side.json` (block `results/ledger`). The copy here holds that block only. |
| Section IV-B: displayed ordering against the sensitivities (median 0.09; 0.01) | `python/r19_disp.py` writes `results/tables/r19/r19_disp_slack.json`. |
| Section IV-B: device-data probes, each parameter moved by 2% of its sampled range. They rank above chance (p < 0.001): median 0.322 [0.279, 0.364], and 0.469 at 10%. 65.7% of the scenarios fail their own null. | `matlab/device_probes_shard.m` writes the full-model solves, one file per scenario, into `results/device_probes/scenarios/`. It works with `device_probe_set.m` and `device_probe_solve.m`, and `device_probes_selfcheck.m` runs first. `python/device_probes_fidelity.py` writes `results/device_probes/device_probes_fidelity.json`. Under its key `results`, the entry `s0.02` holds `rho_median`, `ci_lo`, `ci_hi`, `n_failing_own_null` of `n_scenarios` (155 of 236), and `pooled_null`, `p_value` (1/1,001). The entry `s0.10` holds the median at 10%. The same script writes the per-scenario values of both scales. They are in `results/device_probes/device_probes_rows_s0.02.csv` and `device_probes_rows_s0.10.csv` (column `rho_ebm`). The own null per scenario is in `results/device_probes/device_probes_null_s0.02.csv` and `device_probes_null_s0.10.csv` (`null_q95`, `below_null_q95`). From these files, `check_paper_values.py` recomputes the two medians and the 155 failing scenarios. |
| Section IV-B: per-plant sign consistency of the reactive sensitivities (0.61 to 0.85) | `python/e2_faithfulness_production.py` writes `results/final/tables/e2_sign_ceiling.csv` (rows of kind `Q`, column `consistency`: lowest and highest plant). |
| Section IV-B: explicit per-slot type indicators (AUC 0.931 against 0.932, importance rank 1 to 26) | `python/e8_paired_test.py` writes `results/final/tables/e8_encoding_ablation.csv` (five-seed means of `AUC` and `AUC_base`) and `results/final/tables/e8_ablation_importances.csv` (`fpll_10` is 26th by importance). |
| Section IV-B: the deletion test (drops +0.005, +0.026, +0.097 against -0.003, -0.004, +0.003) | `python/r16_roar.py` writes `results/tables/r16/r16_roar.csv`: the five-seed means of the importance order, and the seed-0 random orders, against the baseline. Not asserted. |
| Section IV-C: the 10^-2 candidate edge collapses onto the 10^-3 edge in a quarter of the bootstrap resamples | `python/e1_tiers.py` writes `results/final/tables/e1_tier_bootstrap.csv`: three tiers in a share 0.2415 of the resamples. The text gives this share as a word, and it is not asserted. |
| Section IV-C: the frozen bank's one Secure collapse (+33.6 s^-1, prediction -3.8) | `python/r19_frzc.py` writes `results/tables/r19/r19_frzc_secure.csv` (row `seedmean`, id 5588: `sigma_dom` 33.594 and `pred`). `python/r20_cheap.py` writes the same rate in `results/tables/r20/r20_cheap.json` (`cap/id5588_sigma`). |
| Section IV-C: the alarm band alone (82.0% of points covered, 95.3% recall, 22.7% false alarms) | `python/e1_tiers.py` writes `results/final/tables/e1_abstention_curve.csv` (row w = 0.30). |
| Section IV-C: alarm precision 39% (216 of 557) | `results/tables/r13/r13_results.json` (`task2/a_alarmed_unstable` 216). 557 = 216 + the false-alarm rate of `results/final/tables/e1_alarms.csv` times the 1,077 stable test rows. Both are asserted by `derived()`. |
| Section IV-C: the screen's first 557 points, its alarms, hold 216 of the 231 unstable test points. A random order's first 557 would hold about 98 on average. The 482 XGBoost alarms hold 217. | `results/final/tables/e1_alarms.csv` (rows M+D `ebm` and `xgb`, seed 0: `test_recall`, `test_FA`), with the test counts of `results/final/tables/e1_tiers.csv` (1,308 rows, 231 unstable), by `derived()`. 98 = 557 x 231 / 1,308. 217 = the recall x 231. 482 = 217 + the false-alarm rate x 1,077. 557 and 216 come from the same operation. |
| Section IV-C: false alarms by distance to the boundary (87%, 4.7%; 17.7% unstable). The routed occupancy: 0.92 on the test frame (also in the abstract), 0.94 on the frozen forward bank, 0.86 to 0.88 at the mixtures. | `python/r17_econ.py` writes `results/tables/r17/r17_econ_curve.csv` and `r17_econ_scenarios.csv` (column `alpha`: row `test_frame` 0.9159 and the two mixtures). `python/r19_frzc.py` writes `results/tables/r19/r19_frzc_census.csv`: the row `routed` of the convention `seedmean` (the five-seed mean prediction) holds 0.9392. |
| Section IV-C: recourse on 90 verified-unsafe rows (87%, 70%, 245 MW or 4.1% of the system load, 75 evaluations, 52.35 s per case) | `matlab/r17_rec.m` writes `results/tables/r17/r17_rec.csv`: the rows with `sigma0_chk` > 0, with medians as in `derived()`. The load of each scenario is `Ptot` of `data/dataset.csv`. |
| Section IV-C: 4,800 assumed cases per window (200 x 24) ordered in about 13 ms after 6.9 CPU-min of power flows, against 75 CPU-min of full-model analysis | `derived()` takes 4,800 times the medians of `results/final/tables/e6_timing_python.csv` (`python/e6_timing_py.py`) and `results/final/tables/e6_timing_matlab.csv` (`matlab/e6_timing.m`). |
| Section IV-D: constant-power labels at 60% (30%). The retrained monitor's AUC is 0.976 (0.957) +/- 0.001, and its recalibrated alarm gives 96.4% (92.6%) recall at 14.4% (16.4%) false alarms. The constant-impedance monitor's AUC is 0.66 (0.79) at 53% (71%) recall. | At 60%, `matlab/r16_cp_relabel.m` labels the bank under 60% smoothed constant-power loads (`ALPHA = 0.6`). `python/r16_cp_retrain.py` writes `results/tables/r16/r16_cp_retrain.csv` and the per-seed `r16_cp_retrain_raw.csv`. At 30%, `matlab/cp_relabel_shard.m` labels the bank under 30% in shards. After `matlab/cp60_relabel_check.m` has checked the 60% labels, `matlab/cp30_relabel_merge.m` merges the shards into `results/cp30/cp30_relabel.csv`. `python/cp30_retrain.py` writes `results/cp30/cp30_retrain.csv` and `cp30_retrain_raw.csv`. |
| Section IV-D: the 1.2 p.u. current limit | `matlab/r16_limiter_straddle.m` writes `results/tables/r16/r16_limiter_straddle.csv`. |
| Section IV-D: union-trained monitor (within 0.004 to 0.008 AUC) | `python/r19_lmu.py` writes `results/tables/r19/r19_lmu.csv` and `r19_lmu_alarm.csv`. |

### Other numbers of the text

The other result numbers of the text are values in the files of the table above, or in these:

- the alarm rates of `results/final/tables/e1_alarms.csv`, `e1_ood.csv` and the noise scores of
  `e1_noise.csv` (written by `python/e1_margin_suite.py`)
- the blocks of `results/tables/r13/r13_results.json` and `results/tables/r15/r15_results.json`
  (written by `python/r13_battery.py` and `python/r15_battery.py`), and of
  `results/tables/r20/r20_cheap.json`
- the reduced model's k* in `results/final/tables/e3b_gfmfull.csv`
- the straddle of `results/final/e4_straddle.mat`
- the two bank rays of `results/tables/r17/r17_rays.csv` and `results/tables/r20/r20_straddle2.csv`
- the delay study of `results/tables/r17/r17_delay_summary.csv` (see [The step delay](#the-step-delay))
- the flags arm's frames in `results/tables/r15/r15_mf_strata.csv`
- the wrong flag of `results/tables/r19/r19_flagw.csv`
- the stale channel of `results/tables/r19/r19_qstale.csv`
- the resplit spreads of `results/tables/r15/r15_repeated_splits_summary.csv`
- the frozen bank's scores in `results/tables/r18/r18_freeze_score.csv`
- the operating points of `results/tables/r17/r17_opci_cells.csv`

The remaining numbers are parameter values that are code (`matlab/+params/`, and the sampled ranges
in `matlab/build_bank.m`), or values cited from other papers.

Section IV-D prints the constant-power AUCs as 0.976 (0.957 at 30%) +/- 0.001. The +/- is the sample
standard deviation (ddof 1) of the five seeds in `results/tables/r16/r16_cp_retrain_raw.csv`
(0.00056) and `results/cp30/cp30_retrain_raw.csv` (0.00078). The summary files `r16_cp_retrain.csv`
and `cp30_retrain.csv` hold the population values (0.000499 and 0.000701).

`results/final/tables/e1_alarms.csv` holds the alarms of seed 0. Its writer,
`python/e1_margin_suite.py`, calibrates each threshold on the seed-0 out-of-fold predictions and
applies it to the seed-0 test predictions. The file has no seed column.

## Side analyses

A few scripts also compute results that the paper does not report. These parts ship because a step
behind a printed result needs them:

- the Kolmogorov-Arnold network (KAN) arm: `KANRegressor` and task `T` of `python/r15_battery.py`,
  with `python/vendored_efficient_kan.py`. `python/r16_table3_ci.py`, `python/r19_ssci.py` and
  `python/r20_frzs.py` also refit this arm.
- the fidelity rows of that arm from an earlier run (`results/tables/r16/r16_kan_faith_rows.csv`).
  `python/r22_faith_slack.py` compares with them in a side block of its record.
- the six-class model comparison of an earlier analysis. `python/run_ml.py` runs it when it is
  called without `--frames-only`.

The device-data probes of Section IV-B are the steps `device-probes` and `device-fidelity`.

### AMBER-SP

AMBER-SP is a penalized spline GAM (model key `sp`, label AMBER-SP). `python/e1_margin_suite.py`
fits it beside the deployed arm. The paper does not report it. The paper's AMBER is the unmodified
explainable boosting machine (model key `ebm`). The rows of AMBER-SP are in:

- `results/final/tables/e1_model_suite.csv`, `e1_model_suite_raw.csv`, `e1_reg_metrics.csv`,
  `e1_alarms.csv`, `e1_noise.csv`, `e1_ood.csv` and `e6_timing_python.csv`
- `results/tables/r15/r15_repeated_splits_raw.csv`, `r15_repeated_splits_summary.csv` and
  `r15_results.json`
- `results/tables/r17/r17_opci_cells.csv` and `e2r17_faith_rows_fine_md.csv` (the `*_sp` columns)
- the per-scenario rows `results/final/r22_faith_slack_rows.csv` and
  `results/reference/committed/r22_faith_slack_rows.csv` (the `*_sp_*` columns)
- the gate records `results/final/r22_faith_slack.json` and three files of `results/reference/`

Its seed-0 alarm on M+D gives 96.5% recall at 31.4% false alarms (`results/final/tables/e1_alarms.csv`,
row M+D `sp`). It is among the "other arms" that Section IV-A says dominate the deployed pair (93.5%
at 31.7%) at the point estimates.

### Other side analyses

Other side analyses ship with their steps. The paper prints no number from them:

- the reactive-channel bias and common-mode noise of `python/r16_corruption.py`
  (`results/tables/r16/r16_corruption.csv`)
- the 12-case recourse run on surrogate candidates of `matlab/q9_recourse_physics.m`
  (`results/final/tables/q9_recourse_physics.csv` and `q9_recourse_physics2.csv`). The paper's
  recourse census is `results/tables/r17/r17_rec.csv`.
- the frozen-bank scores of the other arms in `results/tables/r20/r20_frzs_status.json`. The paper
  prints AMBER's 0.930 from `r18_freeze_score.csv`.
- the blocks `EXP-L` (reliability), `EXP-M` (nearest-neighbor leakage audit) and `EXP-C`
  (zero-crossing bootstrap) of `results/tables/r15/r15_results.json`

The block `EXP-K` of that file is not a side analysis. It holds the 28 of 88 terms of Section III-A,
which the table above lists.

### Scripts named but not shipped

Some scripts name earlier scripts that are not part of this repository, in comments and labels:
`e5_inversion.py`, `e2_blackbox_contrast.py`, `e1_shape_fix.py`, `fs2_sens_sigma.m`, `fs4_emt.m`,
`r16_kan_faith.py`, `e2_recourse_margin.py` and `r2_regen_thresholds.m`. The last one is the script
that wrote the bank's threshold columns.

## Numbers printed in the paper without a result file here

These numbers of the paper have no result file in this repository, or could not be confirmed by
reading one:

| Where | Number | Why |
|---|---|---|
| Section II-B | load scale in [0.6, 1.25] (realized cap 1.2) | The generator draws the load scale in [0.6, 1.2] in the main round and in [0.9, 1.2] in the top-up round (`matlab/build_bank.m` lines 60 and 56). The paper prints this as the realized cap 1.2. No file here holds the upper end 1.25 of the printed range. |
| Section III-D | the 28 paired tests of Sections IV-A and IV-B | No file lists or counts these tests. `check_paper_values.py` uses 28 only as the factor of the Bonferroni correction. |
| Section IV-C | timings on a 4-core mobile CPU | The hardware string of `results/final/tables/e6_timing_python.csv` names the processor family and Python 3.13.9, not the number of cores. |

## Layout

```
run_all.py             runs the steps in order, then check_paper_values.py
check_paper_values.py  compares the result files with the printed numbers
matlab/                model chain and bank generator
  +dev/                device models: grid-following converter (gfl_rhs.m)
                       and grid-forming virtual synchronous machine
                       (gfm_vsm_rhs.m), 14 states each (state_map.m)
  +params/             parameter values (converter.m, sys_const.m)
  +net/ +full/ +reduced/  network, full-order and reduced models
  +bank/ +checks/      bank sampling and model checks
  build_case.m ...     one scenario; build_bank.m and assemble_bank.m
                       build data/dataset.csv
python/                training, evaluation and result scripts
data/dataset.csv       the scenario bank, 7324 rows (schema in data/schema.md)
results/               the result files behind Tables II to V, and
                       results/bank/bank_spec.mat, the sampling plan of the
                       bank's main round
  device_probes/       the device-data probes: the full-model solves per
                       scenario (scenarios/) and their fidelity record
  cp30/                the 30% constant-power labels and the retrained
                       monitor on them
  tables/r13/ ... r21/ result files of the computation runs of that number
  svg/                 Figs. 8 and 9 as vector SVG (the step fig-vector)
docs/DETAILS.md        this page
```

The script names and the folders under `results/tables/` begin with prefixes: e1 to e9, fs1 to fs3, q1
to q9 and r13 to r29, with f for a figure module. These prefixes are run numbers of the computation,
and they carry no meaning.

`matlab/+checks/smib.m` and `smib_rhs.m` model one converter behind a line to an infinite bus. They
are a check of the device model, not a test system of the paper. No step calls them.

The device equations and parameter values are code. `+dev/state_map.m` lists the 14 states of each
device: `k = dev.state_map('GFL')` (or `'VSM'`) returns the index of each state, with their names in
`k.names`. `P = params.sys_const()` holds the system constants (60 Hz, the 100 MVA base, the
tolerances and the master seed 42). `c = params.converter(type, Sn, P, tune)` holds the parameters
of one converter on its own base.

## Data

`data/dataset.csv` has one row per converged scenario of the bank. A row holds the sampled design
parameters, the power-flow measurements, the device data, the strength features and the labels
(dominant-mode decay rate, damping and the collapse label). [`data/schema.md`](../data/schema.md)
describes each column. It also describes the columns of the files of the device-data probes and of
the 30% labels. Plant index i = 1 to 10 corresponds to buses 30 to 39. The column `PWSGSI` is the GRIP
index of the paper (lambda_min x Hbar).

The suite files (`results/final/tables/e1_model_suite.csv`, `e1_reg_metrics.csv`, `e1_noise.csv` and
`e1_ood.csv`) have rows `legacy32`. These rows are a fifth feature set of the suite: the P, Q and Yeff
columns and the two totals, 32 features (`CONFIGS` in `python/fs1_margin_feasibility.py`). The
paper's tables do not show them, and `check_paper_values.py` does not read them. They hold the Yeff
columns, a model-side quantity (`data/schema.md`: "model-side quantity; the ablation target"). By
design, the deployed monitor uses no model-side quantity (Section III-A).

The name `results/tables/T7_plants.csv` comes from an earlier table numbering. The paper has Tables
I to V.

### Labels inside the result files

- `task1` to `task10` in `results/tables/r13/r13_results.json` are the computation blocks of
  `python/r13_battery.py`.
- The letters `A` to `T` in `results/tables/r15/r15_results.json`, each with an `EXP-*` name, are the
  computation blocks of `python/r15_battery.py`. The exception is the block `R`, the index of [13]
  from `matlab/r15_jvsm.m`, which no script here writes.
- `item8` in `results/tables/r21/r21_design_baseline.json` is a block of
  `python/r21_design_baseline.py`.
- A gate label begins with the gate or block of the script that writes the record: G and a number,
  or a letter.
- In a gate label, "published" or "stored" names the reference value that the gate compares with.
  This value comes from a stored result file or an earlier record. "earlier record" names a value
  recorded by an earlier run. Neither is necessarily a number of this paper.

## Shipped files that no step writes

`results/` holds our run of the result files. The files below have no writer step here: they are
inputs of the scripts, and the failure ledger of the bank. Each note says which script reads the file.

- `results/tables/r19/r19_disp_summary.csv`: summary of an earlier run of `r19_disp.py`, whose gates
  compare with it. No step writes it.
- `results/tables/r19/r19_disp_status.json`: status record of the same earlier run, written by no
  step. `r19_disp.py` compares with it.
- `results/tables/r19/r19_disp_rows.csv`: per-scenario rows of the same earlier run. `r19_disp.py`
  and `r22_faith_slack.py` compare with them, and no step writes them.
- `results/tables/r16/r16_kan_faith_rows.csv`: fidelity rows of an earlier run, written by no step.
  `r22_faith_slack.py` compares with them.
- `results/final/tables/e1_grip_shape.csv`: the fitted shape of the GRIP term (40 KB). Its writer is
  not in this repository. `r13_battery.py` reads it (shape statistics).
- `results/final/tables/e2_shap_global.csv`: the global TreeSHAP importances of an earlier run (1.4
  KB). Its writer is not in this repository. `e2_shap_global_fix.py` (step `shap-order`) and
  `r13_battery.py` read it.
- `results/tables/r18/r18_plant_sn.csv`: the ten plant slots (bus, rating Sn in MVA, base dispatch in
  MW). No script here writes it. `r18_battery.py` reads it.
- `results/final/e2_recourse_candidates.csv`: the recourse candidates of an earlier run (3 KB). Its
  writer is not in this repository. `q9_recourse_physics.m` (step `recourse`) reads it.
- `results/tables/r15/r15_failure_ledger.csv`: the failure ledger of the bank, one row per failed draw
  (1,176). Its columns are the round (`round`: `bank` or `bank_topup`), the id, the two sampled scale factors, the
  GFM count and the failure reason. Its writer is not in this repository, and no script here reads
  it. `check_paper_values.py` counts it (Section II-B).

## Shipped inputs that a step also writes

Five more shipped inputs are also written by a step. They ship so that the later steps, Fig. 9 and the
device-data probes can run without first running the bank, battery, probe and fidelity steps.

- `results/bank/bank_spec.mat`: the sampling plan of the bank's main round. `build_bank.m` (step
  `bank-main`) writes it. `r18_freeze.m`, `run_ml.py` and `r19_disp.py` read it.
- `results/tables/r18/r18_gal.csv`: the gallery cases. `r18_battery.py` (step `battery-18`) writes
  it. `r29f_glassbox.py` (Fig. 9) and `r21_faith_md_side.py` read it.
- `results/final/e2_probe_features.csv`: the full-model responses to the injection probes (12,833
  solves: 41 per scenario for 313 scenarios). `e2_sens_production.m` (step `probes`, three to five
  hours on four workers) writes it. `e2_faithfulness_production.py`, `e2r17_faith.py`, `r19_taf.py`,
  `r20_cheap.py`, `r21_faith_md_side.py`, `r22_faith_slack.py`, `device_probes_fidelity.py` (its gate)
  and `device_probes_selfcheck.m` read it.
- `results/tables/r17/e2r17_faith_rows_fine_md.csv`: the fidelity rows of the deployed arm on the
  injection probes (236 scenarios, column `rho_ebm`). `e2r17_faith.py` (step `faith-md`) writes it.
  `r17_devnbf.py`, `r19_disp.py`, `r21_faith_md_side.py`, `r22_faith_slack.py`,
  `device_probes_fidelity.py` and `device_probes_shard.m` read it. Its ids are the scenarios of the
  device-data probes.
- `results/tables/T7_plants.csv`: the ten plant slots with their ratings (`Sn_MVA`). `write_tables.m`
  (step `plants`) writes it. `r13_battery.py`, `r19_disp.py`, `r21_faith_md_side.py`,
  `r22_faith_slack.py` and `device_probes_fidelity.py` read it.

## Reference files

`results/reference/` holds number files of earlier runs. Four scripts compare their results with
them.

- `results/reference/committed/e9_ma2024_index.json`: an earlier result of `e9_ma2024_index.py`,
  which compares its result with it.
- `results/reference/committed/e9_ma2024_index_rows.csv`: the rows of that earlier result.
  `e9_ma2024_index.py` compares with them.
- `results/reference/committed/e9_ma2024_threshold.json`: an earlier result of
  `e9_ma2024_threshold.py`, which compares with it.
- `results/reference/committed/r22_faith_slack.json`: an earlier result of `r22_faith_slack.py`,
  which compares its results with it.
- `results/reference/committed/r22_faith_slack_rows.csv`: the per-scenario rows of that earlier
  result. `r22_faith_slack.py` compares with them.
- `results/reference/common_base_preview_out.json`: an earlier run of the common-power-base readings.
  `r22_faith_slack.py` compares with it.
- `results/reference/common_base_robustness_out.json`: an earlier `robustness` run of the same
  readings. `r22_faith_slack.py` compares with it.
- `results/reference/ma2024_preview_auc.json`: earlier AUCs of the index of [9].
  `e9_ma2024_index.py` compares with them.
- `results/reference/r21f_fig_dims.json`: the bar values of the worked cases, recorded by an earlier
  run (this block only). `r29f_glassbox.py` compares Fig. 9 with them (its self-check).
- `results/reference/rho_slack_verify_out.json`: an earlier verification run of the readings with the
  slack plant's pairs. `r22_faith_slack.py` compares with it.

## Notes on the result records

- The block `xled` of `results/tables/r20/r20_cheap.json` is an older record. Its split of the 77
  removed probe rows (70 and 7) is a residual. The split that the paper prints (73 and 4) is the
  ledger of `results/tables/r21/r21_faith_md_side.json`.
- Text labels of the JSON result files name the files of this repository (for example
  `results/reference/`). Every number is as computed.
- The result records keep the time stamps of their runs (fields such as `started`).
- The folder `results/reference/committed/` holds results as they were committed to the project's
  version history before the last run of their scripts. These scripts compare with them.
- The `code_id` fields of `results/final/e9_ma2024_threshold.json` (b53a488ca9fa9d36) and
  `results/final/e9_ma2024_index.json` (caa17c871ad2f35d) name the MATLAB scripts as they ran. Each is
  the first 16 hex digits of the SHA-256 of `matlab/e9_ma2024_threshold.m` or
  `matlab/e9_ma2024_index.m`, with carriage returns removed (`code_hash` in each script). The copies
  here differ from the scripts that ran in the wording of their comments. In
  `e9_ma2024_threshold.m`, they also differ in the folder names under `results/tables/` in five paths
  and one comment. For the copies here, `code_hash` gives 61b6a3f8c58b3846 and c434dd3d31fc105c. The
  computation is the same.
- Some gate texts of `python/r22_faith_slack.py` are code strings that go into its record. Read two
  of them with care. The label `e2r17_faith.py line 149 (fin.sum() < 8)` names the test
  `fin.sum() < 8`, which is at line 146 of `python/e2r17_faith.py`. The gate on the majority-sign
  baseline of Table IV's scenarios (all pairs, 0.705) gives "the printed value" as its source, but
  the paper does not print 0.705. It is the value of an earlier run, in
  `results/reference/committed/r22_faith_slack.json`.
- Figures are written to `results/figures/`, which is not shipped. The shipped vector copies of Figs. 8
  and 9 are in `results/svg/`.

## Citation

```bibtex
@unpublished{poursaeed2026explainable,
  author = {Poursaeed, Amir Hossein and Namdari, Farhad and
            Shen, Shuhang and Crossley, Peter A.},
  title  = {Intrinsically Explainable, Model-Verified Monitoring of the Small-Signal Collapse Margin in 100% Inverter-Based Power Systems},
  note   = {Submitted to the IEEE Transactions on Industry Applications},
  year   = {2026}
}
```
