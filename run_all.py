"""Runs the computations behind the paper's results in order, then
check_paper_values.py. MATLAB steps run in matlab/ through `matlab -batch`
(MATLAB and MATPOWER on the path); Python steps run in python/.

    python run_all.py --list        the steps, in order
    python run_all.py STEP          one step
    python run_all.py --from STEP   that step and every later one
    python run_all.py --all         every step, then the check

Every step writes into results/ (or data/ for the bank), over the copies
there, so `git diff` shows what changed. Many MATLAB steps resume: they
skip what their output file or folder already holds (see docs/DETAILS.md,
section "Steps that resume").
"""
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (step, folder, command, what it writes)
STEPS = [
    ('n1-pool', 'matlab', 'matlab -batch run_phase1_n1', 'the N-1 outage pool of the bank'),
    ('bank-main', 'matlab', 'matlab -batch build_bank', 'the main round of the bank (6000 draws) and results/bank/bank_spec.mat'),
    ('bank-topup', 'matlab', 'matlab -batch "build_bank(2500, \'bank_topup\', \'topup\')"', 'the boundary-targeted top-up round'),
    ('bank', 'matlab', 'matlab -batch assemble_bank', 'data/dataset.csv'),
    ('frames', 'python', 'python run_ml.py --frames-only', 'the train and test frames (results/ml_export/) [written by a rerun, not shipped]'),
    ('plants', 'matlab', 'matlab -batch write_tables', 'results/tables/T7_plants.csv'),
    ('probes', 'matlab', 'matlab -batch e2_sens_production', 'the full-model probe responses: results/final/e2_probe_features.csv and the coarse e2_probe_features_coarse.csv (written by a rerun, not shipped) [run time: three to five hours on four workers]'),
    ('probes-nbf', 'matlab', 'matlab -batch r17_nbf', 'the near-boundary probe responses'),
    ('frozen-bank', 'matlab', 'matlab -batch r18_freeze', 'the frozen forward bank'),
    ('index-rows', 'matlab', 'matlab -batch "e9_ma2024_index(\'../results/e9_work/index\')"', 'the strength indices per bank row'),
    ('suite', 'python', 'python e1_margin_suite.py', 'results/final/tables/e1_model_suite.csv (Table II) [run time: three to four hours]'),
    ('tiers', 'python', 'python e1_tiers.py', 'results/final/tables/e1_tiers.csv (Table V)'),
    ('reg-metrics', 'python', 'python e1_reg_metrics.py', 'results/final/tables/e1_reg_metrics.csv (Table II)'),
    ('paired', 'python', 'python e8_paired_test.py', 'the paired test and the test predictions'),
    ('faith', 'python', 'python e2_faithfulness_production.py', 'the response pairs and fidelity rows'),
    ('battery-13', 'python', 'python r13_battery.py', 'tier battery rows, type-flag frame'),
    ('battery-15', 'python', 'python r15_battery.py', 'flags arm, same-population metrics, fidelity tail rows'),
    ('flags-ladder', 'python', 'python r17_mfl.py', 'results/tables/r17/r17_mfl_tiers.csv'),
    ('table-ci', 'python', 'python r16_table3_ci.py', 'bootstrap intervals of the suite'),
    ('census', 'python', 'python r16_census.py', 'results/tables/r16/r16_census.csv (Table V)'),
    ('faith-md', 'python', 'python e2r17_faith.py', 'fidelity rows of the deployed arm'),
    ('faith-ci', 'python', 'python r17_faith_ci.py', 'bootstrap intervals of the fidelity medians'),
    ('faith-nbf', 'python', 'python r17_devnbf.py', 'near-boundary fidelity rows'),
    ('op-ci', 'python', 'python r17_opci.py', 'operating-point intervals'),
    ('frozen-score', 'python', 'python r18_freeze_score.py', 'predictions on the frozen forward bank'),
    ('frozen-census', 'python', 'python r19_frzc.py', 'results/tables/r19/r19_frzc_census.csv (Table V)'),
    ('stable-ci', 'python', 'python r19_ssci.py', 'stable-side intervals'),
    ('type-aware', 'python', 'python r19_taf.py', 'fidelity rows of the type-aware encoding'),
    ('trust', 'python', 'python r19_trust.py', 'the fidelity trust rows'),
    ('design-baseline', 'python', 'python r21_design_baseline.py', 'results/tables/r21/r21_design_baseline.json (Tables II and III)'),
    ('faith-side', 'python', 'python r21_faith_md_side.py', 'the fidelity side readings [run time: 15 to 20 minutes]'),
    ('faith-slack', 'python', 'python r22_faith_slack.py', 'results/final/r22_faith_slack.json (Table IV)'),
    ('index', 'python', 'python e9_ma2024_index.py --merge ../results/e9_work/index/e9_rows_1_7324.csv', 'results/final/e9_ma2024_index.json (Table III)'),
    ('emt', 'matlab', 'matlab -batch e4_emt_production', 'averaged-EMT runs of the boundary verification'),
    ('emt-map', 'matlab', 'matlab -batch e4b_map_fix', 'results/final/e4_straddle.mat'),
    ('full-port', 'matlab', 'matlab -batch e3b_gfmfull_design', 'results/final/tables/e3b_gfmfull.csv'),
    ('drift-a', 'matlab', 'matlab -batch q1_hbar_drift', 'results/final/tables/q1_hbar_drift.csv'),
    ('drift-b', 'matlab', 'matlab -batch q1_hbar_drift_b', 'results/final/tables/q1_hbar_drift_b.csv'),
    ('continuation', 'matlab', 'matlab -batch q3_beyond_fold', 'results/final/tables/q3_continuation.csv'),
    ('delay', 'matlab', 'matlab -batch r17_delay', 'results/tables/r17/r17_delay_summary.csv'),
    ('recourse-rows', 'matlab', 'matlab -batch r17_rec', 'results/tables/r17/r17_rec.csv'),
    ('rays', 'matlab', 'matlab -batch r17_rays', 'results/tables/r17/r17_rays.csv and r17_rays_spec.mat'),
    ('straddle', 'matlab', 'matlab -batch r20_straddle2', 'results/tables/r20/r20_straddle2.csv'),
    ('straddle-b', 'matlab', 'matlab -batch r20_straddle2b', 'appends to r20_straddle2.csv'),
    ('timing-m', 'matlab', 'matlab -batch e6_timing', 'results/final/tables/e6_timing_matlab.csv'),
    ('timing-py', 'python', 'python e6_timing_py.py', 'results/final/tables/e6_timing_python.csv'),
    ('shap-order', 'python', 'python e2_shap_global_fix.py', 'the TreeSHAP order'),
    ('deletion', 'python', 'python r16_roar.py', 'results/tables/r16/r16_roar.csv'),
    ('flag-noise', 'python', 'python r19_flagw.py', 'results/tables/r19/r19_flagw.csv'),
    ('resplits-ftt', 'python', 'python r19_fttr.py', 'results/tables/r19/r19_fttr_summary.csv'),
    ('stale-q', 'python', 'python r19_qstale.py', 'results/tables/r19/r19_qstale.csv'),
    ('frozen-split', 'python', 'python r20_frzs.py', 'results/tables/r20/r20_frzs_status.json'),
    ('cheap', 'python', 'python r20_cheap.py', 'results/tables/r20/r20_cheap.json'),
    ('economics', 'python', 'python r17_econ.py', 'screening economics'),
    ('cpl-ray', 'matlab', 'matlab -batch q7_cpl_ray', 'results/final/tables/q7_cpl_ray.csv'),
    ('recourse', 'matlab', 'matlab -batch q9_recourse_physics', 'results/final/tables/q9_recourse_physics2.csv'),
    ('limiter', 'matlab', 'matlab -batch r16_limiter_straddle', 'results/tables/r16/r16_limiter_straddle.csv'),
    ('cp-relabel', 'matlab', 'matlab -batch r16_cp_relabel', 'results/tables/r16/r16_cp_relabel.csv'),
    ('cp-retrain', 'python', 'python r16_cp_retrain.py', 'results/tables/r16/r16_cp_retrain.csv'),
    ('threshold-rows', 'matlab', 'matlab -batch "o = \'../results/e9_work/threshold\'; e9_ma2024_threshold(o, \'test\', [], [], \'\', \'full\'); e9_ma2024_threshold(o, \'bank\', 1, 20, \'\', \'conv\'); e9_ma2024_threshold(o, \'bank\', 1, 20, \'\', \'full\'); e9_ma2024_threshold(o, \'rays\', [], [], \'\', \'full\')"', 'the threshold rows (results/e9_work/threshold/) [written by a rerun, not shipped]'),
    ('boundary', 'matlab', 'matlab -batch fs3_boundary', 'results/feasibility/fs3_boundary.csv'),
    ('eig-align', 'matlab', 'matlab -batch e3c_eig_align', 'results/final/tables/e3c_eig_align.csv'),
    ('test-ids', 'python', 'python r15_test_ids.py', 'results/tables/r15/r15_test_ids.csv [written by a rerun, not shipped]'),
    ('jvsm', 'matlab', 'matlab -batch r15_jvsm', 'results/tables/r15/r15_jvsm_test.csv'),
    ('battery-18', 'python', 'python r18_battery.py', 'results/tables/r18/r18_pidx.csv'),
    ('ftt-pidx', 'python', 'python r18_fttp_pidx.py', 'results/tables/r18/r18_fttp_pidx.csv'),
    ('corruption', 'python', 'python r16_corruption.py', 'results/tables/r16/r16_corruption.csv'),
    ('disp', 'python', 'python r19_disp.py', 'results/tables/r19/r19_disp_slack.json'),
    ('union', 'python', 'python r19_lmu.py', 'results/tables/r19/r19_lmu.csv'),
    ('threshold', 'python', 'python e9_ma2024_threshold.py --merge-test ../results/e9_work/threshold/e9thr_test_full_1_1308.csv --merge-gate ../results/e9_work/threshold/e9thr_bank_conv_1_20.csv ../results/e9_work/threshold/e9thr_bank_full_1_20.csv --rays ../results/e9_work/threshold/e9thr_rays_full.csv', 'results/final/e9_ma2024_threshold.json'),
    ('device-probes', 'matlab', 'matlab -batch "o = \'../results/device_probes\'; device_probes_selfcheck(\'..\', o, \'i+ii\'); for k = 1:16, device_probes_shard(\'..\', o, k, 16); end"', 'the full-model solves of the device-data probes, one file per scenario (results/device_probes/scenarios/; cluster-scale: 12,772 solves) [run time: 55 minutes on one 16-core cluster node with the 16 shards in parallel; this step runs them one after another]'),
    ('device-fidelity', 'python', 'python device_probes_fidelity.py', 'results/device_probes/device_probes_fidelity.json (Section IV-B) [run time: about four minutes on a laptop]'),
    ('cp30-labels', 'matlab', 'matlab -batch "o = \'../results/cp30\'; for k = 1:16, cp_relabel_shard(0.60, k, 16, \'..\', o); end; cp60_relabel_check(\'..\', o, 16); for k = 1:16, cp_relabel_shard(0.30, k, 16, \'..\', o); end; cp30_relabel_merge(\'..\', o, 16)"', 'results/cp30/cp30_relabel.csv (cluster-scale: the 60% labels again as the check, then the 30% labels, 7,324 solves each) [run time: 20 minutes on one 16-core cluster node with the shards in parallel; this step runs them one after another]'),
    ('cp30-retrain', 'python', 'python cp30_retrain.py', 'results/cp30/cp30_retrain.csv and cp30_retrain_raw.csv (Section IV-D) [run time: 12 to 14 minutes on a laptop]'),
    ('fig8', 'python', 'python r29f_fidelity.py', 'Fig. 8'),
    ('fig9', 'python', 'python r29f_glassbox.py', 'Fig. 9 [run time: about two minutes, most of it the model fit]'),
    ('fig-vector', 'python', 'python fig_vector.py', 'results/svg/r29_fig8_fidelity_h160.svg and r29_fig9_glassbox.svg (Figs. 8 and 9 as vector SVG, from the PDFs of the steps fig8 and fig9)'),
]


def run(step):
    """Python steps run with this interpreter; MATLAB steps through the shell
    (the quoted -batch statements). A failing step stops the chain with its
    name and exit code."""
    key, cwd, cmd, what = step
    print(f"== {key}: {cmd}   [in {cwd}/]", flush=True)
    if cwd == "python":
        args, shell = [sys.executable] + shlex.split(cmd)[1:], False
    else:
        args, shell = cmd, True
    rc = subprocess.run(args, cwd=ROOT / cwd, shell=shell).returncode
    if rc:
        print(f"step {key} failed with exit code {rc}; later steps not run")
        raise SystemExit(1)


def main(argv):
    a = argv or ["--list"]
    keys = [s[0] for s in STEPS]
    if a[0] == "--list":
        for k, cwd, cmd, what in STEPS:
            print(f"{k:22s} {cwd:7s} {cmd}\n{'':30s}-> {what}")
        return 0
    if a[0] == "--all":
        todo = STEPS
    elif a[0] == "--from" and len(a) > 1 and a[1] in keys:
        todo = STEPS[keys.index(a[1]):]
    elif a[0] in keys:
        todo = [STEPS[keys.index(a[0])]]
    else:
        print(f"unknown step {a[0]!r}; see python run_all.py --list")
        return 2
    for s in todo:
        run(s)
    if a[0] in ("--all", "--from"):
        return subprocess.run([sys.executable, "check_paper_values.py"], cwd=ROOT).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
