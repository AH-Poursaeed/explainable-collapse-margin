# data/dataset.csv schema

One row per converged scenario of the bank. build_bank.m draws the bank with the master seed 42. The
spec and the raw draws are in results/bank/bank_spec.mat. The failures are logged in the chunks and
summarized by assemble_bank.m. They are never silently dropped. Plant index i = 1..10 corresponds to
buses 30..39, in order. Converter-base pu is per unit on the plant rating Sn_i (the rating rule of
matlab/build_case.m). The system base is 100 MVA.

The bank was drawn in two sampling rounds: the main round and the top-up round (column round).

## Identity and design parameters
| column | meaning |
|---|---|
| id | scenario index into the LHS draw: 1 to 6000 in the main round, 10001 to 12499 in the top-up round |
| round | sampling round: 1 = main round, 2 = top-up round (`1 + strcmp(spec.mode, 'topup')` in build_bank.m) |
| cfgbuses | buses of the GFM (VSM) plants, for example "[38 39]" |
| nGFM | number of GFM plants |
| outage | branch row removed (0 = full topology). The pool is in bank_spec. The last four entries of the pool are the OOD-reserved topologies. |
| kL | load scale in [0.6, 1.2] |
| k | line-impedance scale in [0.6, 1.9] in the main round, up to 2.3 in the top-up round |
| kdisp_i | per-plant dispatch multiplier before renormalization, in [0.75, 1.25] |
| fpll_i | GFL PLL bandwidth [Hz] (0 for GFM plants) |
| Hv_i, Dv_i | VSM virtual inertia [s] and damping [pu] (0 for GFL plants) |
| kbw | current-loop bandwidth factor (800 Hz nominal) |

## Ground truth

The ground truth comes from the full nonlinear model, linearized at the equilibrium. The equilibrium
is found to the residual gate 1e-9 (column residual_pu).

| column | meaning |
|---|---|
| zeta_min | minimum damping ratio over the sub-synchronous band (\|Im lambda\| <= 2 pi 55), without the reference mode. It is forced negative if any out-of-band mode is unstable (oob_flag). |
| sigma_dom | max Re over the same mode set |
| f_dom | frequency of the dominant mode [Hz] |
| class | damping-band label: 0 Unstable (zeta <= 0), 1 Very Weak (<1%), 2 Weak (<3%), 3 Moderate (<5%), 4 Strong (<10%), 5 Very Strong. Checked row by row: class 0 coincides with sigma_dom >= 0. |
| abstain | 1 if zeta_min is within 0.0015 of a band edge (reject-option band) |
| oob_flag | 1 if an out-of-band instability forced the Unstable label (reported) |
| residual_pu | initialization residual, in pu-mismatch form (gate < 1e-9, `P.tol.resid` of matlab/+params/sys_const.m) |
| nzero | number of structural reference modes found (gate: exactly 1) |
| oob_stable, trust_dom | modal gates (see full.modal) |

## Reduced-chain indices
| column | meaning |
|---|---|
| lam_min | lambda_min(A) |
| H, Hbar | participation entropy and its normalized form |
| PWSGSI | lambda_min x Hbar, the GRIP index of the paper |
| C_SGSI | analytic threshold: the largest real root over the interior and static candidates |
| C_interior, C_dc | the two root families separately: the oscillatory sweep crossing, and the static root at w_c = 0 |
| thr_rule | the family that gives the bound ('interior' or 'dc') |
| im_resid | imaginary residual at the selected root (a heterogeneity diagnostic) |
| w_c | crossing frequency [rad/s] (0 for the static root) |
| gSCR | lambda_min with B_in = 0. Under homogeneous GFL dynamics, this is the gSCR of [8], [9] of the paper. |
| SCR_min | minimum over the buses of driving-point susceptance / S_b |
| lam_gap | lambda_2 - lambda_min (simplicity margin) |
| p_part_i | participation p_i of phi_min |
| port_resid | maximum standalone-device initialization residual across the 10 port extractions |

## Features (classifier inputs)
| column | meaning |
|---|---|
| Pinv_i, Qinv_i | plant P and Q, converter-base pu (the n_f core) |
| Yeff_i | sigma_max(Y_i) at the phasor point, converter base. For GFL it is the terminal admittance, for GFM it is 1/(w0 Leff) (model-side quantity; the ablation target). |
| Yhat_i | measurable surrogate: the apparent admittance \|I_i\|/\|V_i\| from the terminal fundamental phasors, converter-base pu |
| Vbus_i | plant terminal voltage magnitude [pu] (candidate feature set) |
| Ptot, Qtot | total generation, system-base pu |

SI is a normalized strength index of an earlier analysis. The paper does not use it, and it is not
stored here. Its normalization constant (95th percentile of PWSGSI) is defined on the training split
only. Python computes the constant (`common.compute_si`), together with the per-family empirical
thresholds, and saves it in results/ml_export/si_norm.json. The step `frames` writes that file, and it is not
shipped.

## Device-data probes: results/device_probes/scenarios/scenario_<id>.csv

One file per scenario of the injection probes (236 scenarios, written by
matlab/device_probes_shard.m). One row per full-order solve. The first is the base solve (probe =
base). Then come the solves for every device parameter of the scenario's model: Hv_i and Dv_i of each
VSM plant, fpll_i of each GFL plant, and kbw. Each parameter is moved by +-0.02 and +-0.10 of its
sampled range W (Hv 8 s, Dv 160 pu, fpll 35.6 Hz, kbw 0.7).

| column | meaning |
|---|---|
| id | scenario id (as in data/dataset.csv) |
| probe | the moved parameter (a dataset column name), or base |
| ok | 1 if the solve converged |
| sigma_dom, zeta_min, lam_min, Hbar, PWSGSI, Ptot, Qtot, Pinv_i, Qinv_i, Vbus_i, Yeff_i, Yhat_i | as in data/dataset.csv, for this solve |
| resid | infinity norm of the model's right-hand side after the one polish step |
| scale | s, the step as a fraction of the sampled range (0 for base) |
| sgn | +1 or -1 (0 for base) |
| delta | the change of the parameter in its own unit, sgn x scale x W |
| base_value | the parameter's value in the bank row (NaN for base) |

### results/device_probes/device_probes_fidelity.json

python/device_probes_fidelity.py writes this file. Its blocks are:

- results: one entry per scale (s0.02, s0.10). Each entry holds n_scenarios, rho_median, rho_mean,
  ci_lo and ci_hi (bootstrap 95% interval of the median), and share_clearing_own_null. It also holds
  n_failing_own_null, pooled_null (observed_median, null_q95, p_value, ...), sign_agree_w and the two
  majority-sign baselines.
- census: probe pairs, exact-zero responses, probe points outside the sampled range, and moves of
  the measurable block
- the secondary blocks display_secondary, without_slack_plant and main_effect_slope_only

## Constant-power labels at 30%: results/cp30/

cp30_relabel.csv (written by matlab/cp30_relabel_merge.m) has one row per bank row, with the columns
of results/tables/r16/r16_cp_relabel.csv:

- id
- sigma_cz, zeta_cz and class_cz: the stored constant-impedance labels
- sigma_cp, zeta_cp, fdom_cp, class_cp, oob_cp and trust_cp: 30% of every inductive load as smoothed
  constant power, tau 20 ms
- resid_pu
- flip_bin: 1 if the collapse label changes
- dclass and fail

cp30_retrain.csv (written by python/cp30_retrain.py) has one row, with the columns of
r16_cp_retrain.csv:

- retrain_AUC_mean and retrain_AUC_sd (population sd over the five seeds)
- retrain_recall and retrain_FA (the recalibrated alarm)
- frozen_cz_AUC_vs_cp and frozen_cz_recall_vs_cp (the constant-impedance monitor on the 30% labels)
- the tier edges and census, n_train, n_test

cp30_retrain_raw.csv holds the columns `seed`, `AUC` and `R2` per retrain seed.
