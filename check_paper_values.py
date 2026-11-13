"""Compares the result files with the numbers printed in Tables II to V of
the paper and with numbers printed in its text. One line per printed number:
the value read from (or computed from) the result file, the printed value and
the tolerance, which is half a unit of the last printed digit. Standard
library only.

    python check_paper_values.py            the files under results/
    python check_paper_values.py --quiet    only failures and the total

Exit code 1 if a check fails.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (exhibit, row | column, printed value, printed decimals, scale, file, where)
# scale 100: the paper prints the file's fraction as a percentage; scale -1:
# it prints the size of a negative difference.
# where: "row k=v, ...; column c" in a csv file, or a path of keys in a json
# file ("; count of p*" counts the list items that start with p); for the
# file "(derived)", a key of derived() below.
CHECKS = [
    ('Table II', 'M AMBER | RMSE', 1.384, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M, model=ebm; column RMSE'),
    ('Table II', 'M AMBER | R2', 0.42, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column R2'),
    ('Table II', 'M AMBER | stable R2', -0.08, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column R2_stable'),
    ('Table II', 'M AMBER | AUC [95% CI] [1]', 0.885, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column AUC'),
    ('Table II', 'M AMBER | AUC [95% CI] [2]', 0.865, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column AUC_lo'),
    ('Table II', 'M AMBER | AUC [95% CI] [3]', 0.903, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column AUC_hi'),
    ('Table II', 'M+D AMBER* | RMSE', 1.109, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D, model=ebm; column RMSE'),
    ('Table II', 'M+D AMBER* | R2', 0.628, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column R2'),
    ('Table II', 'M+D AMBER* | stable R2', 0.34, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column R2_stable'),
    ('Table II', 'M+D AMBER* | AUC [95% CI] [1]', 0.932, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column AUC'),
    ('Table II', 'M+D AMBER* | AUC [95% CI] [2]', 0.912, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column AUC_lo'),
    ('Table II', 'M+D AMBER* | AUC [95% CI] [3]', 0.948, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column AUC_hi'),
    ('Table II', 'M+D XGBoost | RMSE', 0.893, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D, model=xgb; column RMSE'),
    ('Table II', 'M+D XGBoost | R2', 0.759, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=xgb; column R2'),
    ('Table II', 'M+D XGBoost | stable R2', 0.59, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=xgb; column R2_stable'),
    ('Table II', 'M+D XGBoost | AUC [95% CI] [1]', 0.955, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=xgb; column AUC'),
    ('Table II', 'M+D XGBoost | AUC [95% CI] [2]', 0.939, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=xgb; column AUC_lo'),
    ('Table II', 'M+D XGBoost | AUC [95% CI] [3]', 0.969, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=xgb; column AUC_hi'),
    ('Table II', 'M+D FTT-reg | RMSE', 0.877, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D, model=ftt; column RMSE'),
    ('Table II', 'M+D FTT-reg | R2', 0.767, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ftt; column R2'),
    ('Table II', 'M+D FTT-reg | stable R2', 0.54, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ftt; column R2_stable'),
    ('Table II', 'M+D FTT-reg | AUC [95% CI] [1]', 0.952, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ftt; column AUC'),
    ('Table II', 'M+D FTT-reg | AUC [95% CI] [2]', 0.937, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ftt; column AUC_lo'),
    ('Table II', 'M+D FTT-reg | AUC [95% CI] [3]', 0.97, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ftt; column AUC_hi'),
    ('Table II', 'M+D linear | RMSE', 1.283, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D, model=linreg; column RMSE'),
    ('Table II', 'M+D linear | R2', 0.502, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=linreg; column R2'),
    ('Table II', 'M+D linear | stable R2', 0.06, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=linreg; column R2_stable'),
    ('Table II', 'M+D linear | AUC [95% CI] [1]', 0.911, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=linreg; column AUC'),
    ('Table II', 'M+D linear | AUC [95% CI] [2]', 0.893, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=linreg; column AUC_lo'),
    ('Table II', 'M+D linear | AUC [95% CI] [3]', 0.928, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=linreg; column AUC_hi'),
    ('Table II', 'M+F AMBER | RMSE', 1.139, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/RMSE'),
    ('Table II', 'M+F AMBER | R2', 0.608, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/R2'),
    ('Table II', 'M+F AMBER | stable R2', 0.19, 2, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/R2_stable'),
    ('Table II', 'M+F AMBER | AUC [95% CI] [1]', 0.952, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/AUC'),
    ('Table II', 'M+F AMBER | AUC [95% CI] [2]', 0.936, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/AUC_lo'),
    ('Table II', 'M+F AMBER | AUC [95% CI] [3]', 0.965, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/summary/AUC_hi'),
    ('Table II', 'M+D+P AMBER | RMSE', 1.082, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D+P, model=ebm; column RMSE'),
    ('Table II', 'M+D+P AMBER | R2', 0.646, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column R2'),
    ('Table II', 'M+D+P AMBER | stable R2', 0.39, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column R2_stable'),
    ('Table II', 'M+D+P AMBER | AUC [95% CI] [1]', 0.932, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column AUC'),
    ('Table II', 'M+D+P AMBER | AUC [95% CI] [2]', 0.917, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column AUC_lo'),
    ('Table II', 'M+D+P AMBER | AUC [95% CI] [3]', 0.95, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column AUC_hi'),
    ('Table II', 'M+D+P+I AMBER | RMSE', 1.028, 3, 1, 'results/final/tables/e1_reg_metrics.csv', 'row config=M+D+P+I, model=ebm; column RMSE'),
    ('Table II', 'M+D+P+I AMBER | R2', 0.68, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column R2'),
    ('Table II', 'M+D+P+I AMBER | stable R2', 0.4, 2, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column R2_stable'),
    ('Table II', 'M+D+P+I AMBER | AUC [95% CI] [1]', 0.951, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column AUC'),
    ('Table II', 'M+D+P+I AMBER | AUC [95% CI] [2]', 0.936, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column AUC_lo'),
    ('Table II', 'M+D+P+I AMBER | AUC [95% CI] [3]', 0.964, 3, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column AUC_hi'),
    ('Table II', 'DC4 AMBER | RMSE', 1.269, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/RMSE'),
    ('Table II', 'DC4 AMBER | R2', 0.513, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/R2'),
    ('Table II', 'DC4 AMBER | stable R2', 0.07, 2, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/R2_stable'),
    ('Table II', 'DC4 AMBER | AUC [95% CI] [1]', 0.92, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/AUC'),
    ('Table II', 'DC4 AMBER | AUC [95% CI] [2]', 0.901, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/AUC_lo'),
    ('Table II', 'DC4 AMBER | AUC [95% CI] [3]', 0.937, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/DC4/ebm/summary/AUC_hi'),
    ('Table II', 'caption | n', 1308.0, 0, 1, 'results/tables/r21/r21_design_baseline.json', 'data_facts/sizes/test'),
    ('Table II', 'note | inputs M', 32.0, 0, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M, model=ebm; column n_feats'),
    ('Table II', 'note | inputs M+D', 64.0, 0, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D, model=ebm; column n_feats'),
    ('Table II', 'note | inputs M+D+P', 74.0, 0, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P, model=ebm; column n_feats'),
    ('Table II', 'note | inputs M+D+P+I', 77.0, 0, 1, 'results/final/tables/e1_model_suite.csv', 'row config=M+D+P+I, model=ebm; column n_feats'),
    ('Table II', 'note | inputs M+F', 42.0, 0, 1, 'results/tables/r21/r21_design_baseline.json', 'arms/M+F/ebm/n_feats'),
    ('Table II', 'note | type flags', 10.0, 0, 1, 'results/tables/r21/r21_design_baseline.json', 'definitions/features/M+F; count of is_gfm_slot_*'),
    ('Table III', 'header | all [1]', 1308.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/n'),
    ('Table III', 'header | all [2]', 231.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/n_unstable'),
    ('Table III', 'header | 1 GFM [1]', 363.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/n'),
    ('Table III', 'header | 1 GFM [2]', 147.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/n_unstable'),
    ('Table III', 'header | 2 GFM [1]', 558.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/n'),
    ('Table III', 'header | 2 GFM [2]', 81.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/n_unstable'),
    ('Table III', 'header | 4 GFM [1]', 171.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/n'),
    ('Table III', 'header | 4 GFM [2]', 3.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/n_unstable'),
    ('Table III', 'M+D AMBER* (ensemble) | all', 0.932, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/monitor_md_amber_ensemble'),
    ('Table III', 'M+D AMBER* (ensemble) | 1 GFM', 0.95, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/monitor_md_amber_ensemble'),
    ('Table III', 'M+D AMBER* (ensemble) | 2 GFM', 0.895, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/monitor_md_amber_ensemble'),
    ('Table III', 'M+D AMBER* (ensemble) | 4 GFM', 0.764, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/monitor_md_amber_ensemble'),
    ('Table III', 'M+F AMBER (ensemble) | all', 0.952, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'strata/rows/0/AUC_MF_EBM'),
    ('Table III', 'M+F AMBER (ensemble) | 1 GFM', 0.961, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'strata/rows/1/AUC_MF_EBM'),
    ('Table III', 'M+F AMBER (ensemble) | 2 GFM', 0.934, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'strata/rows/2/AUC_MF_EBM'),
    ('Table III', 'M+F AMBER (ensemble) | 4 GFM', 0.819, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'strata/rows/4/AUC_MF_EBM'),
    ('Table III', 'DC4 AMBER (ensemble) | all', 0.92, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/design_coordinates_amber_ensemble'),
    ('Table III', 'DC4 AMBER (ensemble) | 1 GFM', 0.891, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/design_coordinates_amber_ensemble'),
    ('Table III', 'DC4 AMBER (ensemble) | 2 GFM', 0.894, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/design_coordinates_amber_ensemble'),
    ('Table III', 'DC4 AMBER (ensemble) | 4 GFM', 0.623, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/design_coordinates_amber_ensemble'),
    ('Table III', 'low λ_min | all', 0.839, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/lam_min'),
    ('Table III', 'low λ_min | 1 GFM', 0.576, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/lam_min'),
    ('Table III', 'low λ_min | 2 GFM', 0.808, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/lam_min'),
    ('Table III', 'low λ_min | 4 GFM', 0.931, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/lam_min'),
    ('Table III', 'low GRIP | all', 0.859, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/GRIP'),
    ('Table III', 'low GRIP | 1 GFM', 0.601, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/GRIP'),
    ('Table III', 'low GRIP | 2 GFM', 0.858, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/GRIP'),
    ('Table III', 'low GRIP | 4 GFM', 0.927, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/GRIP'),
    ('Table III', 'index of [9] as published | all', 0.856, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/ma2024_as_published'),
    ('Table III', 'index of [9] as published | 1 GFM', 0.624, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/ma2024_as_published'),
    ('Table III', 'index of [9] as published | 2 GFM', 0.858, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/ma2024_as_published'),
    ('Table III', 'index of [9] as published | 4 GFM', 0.925, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/ma2024_as_published'),
    ('Table III', 'index of [9] without shunts | all', 0.861, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/all_rows/ma2024_as_published_no_shunts'),
    ('Table III', 'index of [9] without shunts | 1 GFM', 0.654, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/ma2024_as_published_no_shunts'),
    ('Table III', 'index of [9] without shunts | 2 GFM', 0.843, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_2/ma2024_as_published_no_shunts'),
    ('Table III', 'index of [9] without shunts | 4 GFM', 0.927, 3, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_4/ma2024_as_published_no_shunts'),
    ('Table III', 'note | nonpositive', 83.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/nonpositive_as_published/test_frame/nGFM_1/n_rows_index_nonpositive'),
    ('Table III', 'note | nonpositive stable', 53.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/nonpositive_as_published/test_frame/nGFM_1/n_stable_among_them'),
    ('Table III', 'note | 1-GFM rows', 363.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_1/n'),
    ('Table III', 'note | 3-GFM rows', 216.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_3/n'),
    ('Table III', 'note | 3-GFM unstable', 0.0, 0, 1, 'results/final/e9_ma2024_index.json', 'results/auc/test_frame/nGFM_3/n_unstable'),
    ('Table IV', 'header | n', 236.0, 0, 1, 'results/final/r22_faith_slack.json', 'results/amber_fine/excl/n'),
    ('Table IV', 'AMBER* | applied mean', 0.368, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_fine/excl/mean'),
    ('Table IV', 'AMBER* | applied median [CI] [1]', 0.422, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_fine/excl/median'),
    ('Table IV', 'AMBER* | applied median [CI] [2]', 0.337, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_fine/excl/ci_lo'),
    ('Table IV', 'AMBER* | applied median [CI] [3]', 0.516, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_fine/excl/ci_hi'),
    ('Table IV', 'AMBER* | w', 0.63, 2, 1, 'results/final/r22_faith_slack.json', 'sign_agreement_w/ebm/excl'),
    ('Table IV', 'XGBoost | applied mean', 0.213, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/xgb/mean'),
    ('Table IV', 'XGBoost | applied median [CI] [1]', 0.2, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/xgb/median'),
    ('Table IV', 'XGBoost | applied median [CI] [2]', 0.158, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/xgb/ci_lo'),
    ('Table IV', 'XGBoost | applied median [CI] [3]', 0.265, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/xgb/ci_hi'),
    ('Table IV', 'XGBoost | w', 0.58, 2, 1, 'results/final/r22_faith_slack.json', 'sign_agreement_w/xgb/excl'),
    ('Table IV', 'FTT-reg | applied mean', 0.371, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/ftt/mean'),
    ('Table IV', 'FTT-reg | applied median [CI] [1]', 0.438, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/ftt/median'),
    ('Table IV', 'FTT-reg | applied median [CI] [2]', 0.344, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/ftt/ci_lo'),
    ('Table IV', 'FTT-reg | applied median [CI] [3]', 0.495, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/ftt/ci_hi'),
    ('Table IV', 'FTT-reg | w', 0.65, 2, 1, 'results/final/r22_faith_slack.json', 'sign_agreement_w/ftt/excl'),
    ('Table IV', 'linear | applied mean', 0.403, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/linreg/mean'),
    ('Table IV', 'linear | applied median [CI] [1]', 0.463, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/linreg/median'),
    ('Table IV', 'linear | applied median [CI] [2]', 0.4, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/linreg/ci_lo'),
    ('Table IV', 'linear | applied median [CI] [3]', 0.548, 3, 1, 'results/final/r22_faith_slack.json', 'results/other_arms_fine_applied_probes/linreg/ci_hi'),
    ('Table IV', 'linear | w', 0.66, 2, 1, 'results/final/r22_faith_slack.json', 'sign_agreement_w/linreg/excl'),
    ('Table IV', 'header | n', 91.0, 0, 1, 'results/final/r22_faith_slack.json', 'results/amber_coarse/excl/n'),
    ('Table IV', 'AMBER* | applied mean', 0.286, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_coarse/excl/mean'),
    ('Table IV', 'AMBER* | applied median [CI] [1]', 0.383, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_coarse/excl/median'),
    ('Table IV', 'AMBER* | applied median [CI] [2]', 0.2, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_coarse/excl/ci_lo'),
    ('Table IV', 'AMBER* | applied median [CI] [3]', 0.494, 3, 1, 'results/final/r22_faith_slack.json', 'results/amber_coarse/excl/ci_hi'),
    ('Table IV', 'AMBER* | w', 0.66, 2, 1, 'results/final/r22_faith_slack.json', 'sign_agreement_w/ebm_coarse/excl'),
    ('Table IV', 'note | majority baseline', 0.73, 2, 1, 'results/final/r22_faith_slack.json', 'majority_baseline_236_diagnostic/excl/value'),
    ('Table IV', 'note | majority baseline n', 236.0, 0, 1, 'results/final/r22_faith_slack.json', 'majority_baseline_236_diagnostic/excl/n_ids'),
    ('Table V', 'Secure, test | n', 198.0, 0, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Secure; column n'),
    ('Table V', 'Secure, test | collapse rate', 0.0051, 4, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Secure; column collapse_rate'),
    ('Table V', 'Secure, test | Wilson 95% [1]', 0.0009, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Secure; column wilson_lo'),
    ('Table V', 'Secure, test | Wilson 95% [2]', 0.028, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Secure; column wilson_hi'),
    ('Table V', 'Secure, deployed rule | n', 110.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Secure; column n'),
    ('Table V', 'Secure, deployed rule | collapse rate', 0.0, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Secure; column collapse_rate'),
    ('Table V', 'Secure, deployed rule | Wilson 95% [1]', 0.0, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Secure; column wilson_lo'),
    ('Table V', 'Secure, deployed rule | Wilson 95% [2]', 0.0337, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Secure; column wilson_hi'),
    ('Table V', 'Alert, test | n', 666.0, 0, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Alert; column n'),
    ('Table V', 'Alert, test | collapse rate', 0.039, 4, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Alert; column collapse_rate'),
    ('Table V', 'Alert, test | Wilson 95% [1]', 0.0268, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Alert; column wilson_lo'),
    ('Table V', 'Alert, test | Wilson 95% [2]', 0.0566, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Alert; column wilson_hi'),
    ('Table V', 'Alert, deployed rule | n', 333.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Alert; column n'),
    ('Table V', 'Alert, deployed rule | collapse rate', 0.03, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Alert; column collapse_rate'),
    ('Table V', 'Alert, deployed rule | Wilson 95% [1]', 0.0164, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Alert; column wilson_lo'),
    ('Table V', 'Alert, deployed rule | Wilson 95% [2]', 0.0544, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Alert; column wilson_hi'),
    ('Table V', 'Emergency, test | n', 444.0, 0, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Emergency; column n'),
    ('Table V', 'Emergency, test | collapse rate', 0.4595, 4, 1, 'results/final/tables/e1_tiers.csv', 'row frame=test, tier=Emergency; column collapse_rate'),
    ('Table V', 'Emergency, test | Wilson 95% [1]', 0.4137, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Emergency; column wilson_lo'),
    ('Table V', 'Emergency, test | Wilson 95% [2]', 0.506, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=pre_rule, tier=Emergency; column wilson_hi'),
    ('Table V', 'Emergency, deployed rule | n', 444.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Emergency; column n'),
    ('Table V', 'Emergency, deployed rule | collapse rate', 0.4595, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Emergency; column collapse_rate'),
    ('Table V', 'Emergency, deployed rule | Wilson 95% [1]', 0.4137, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Emergency; column wilson_lo'),
    ('Table V', 'Emergency, deployed rule | Wilson 95% [2]', 0.506, 4, 1, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Emergency; column wilson_hi'),
    ('Table V', 'note | guard band n', 185.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=abstained, tier=sa_band; column n'),
    ('Table V', 'note | guard band share', 14.1, 1, 100, 'results/tables/r16/r16_census.csv', 'row block=abstained, tier=sa_band; column frac_of_test'),
    ('Table V', 'note | guard band collapses', 1.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=abstained, tier=sa_band; column n_collapse'),
    ('Table V', 'note | deployed Secure Wilson upper', 3.4, 1, 100, 'results/tables/r16/r16_census.csv', 'row block=deployed, tier=Secure; column wilson_hi'),
    ('Table V', 'note | frozen bank upper', 0.56, 2, 100, 'results/tables/r19/r19_frzc_census.csv', 'row convention=seedmean, row=tier_Secure; column wilson_upper_95'),
    ('Table V', 'note | frozen bank n', 803.0, 0, 1, 'results/tables/r19/r19_frzc_census.csv', 'row convention=seedmean, row=tier_Secure; column n'),
    ('Table V', 'note | frozen collapses', 1.0, 0, 1, 'results/tables/r19/r19_frzc_census.csv', 'row convention=seedmean, row=tier_Secure; column n_collapse'),
    ('Table V', 'note | abstained in total', 421.0, 0, 1, 'results/tables/r16/r16_census.csv', 'row block=abstained, tier=total; column n'),
    # numbers printed in the text; "(derived)": computed from
    # shipped files by derived() below
    ('Sec. II-A', 'rows above the +1 s^-1 cap of the training target, % of the bank', 13.7, 1, 100, '(derived)', 'bank:above_cap'),
    ('Sec. II-A', 'smallest stable-side margin', -4.7, 1, 1, '(derived)', 'bank:min_stable'),
    ('Sec. II-A', 'damping-ratio R^2 from M, five-seed mean', 0.014, 3, 1, '(derived)', 'zeta:M'),
    ('Sec. II-A', 'damping-ratio R^2 with M+D, five-seed mean', 0.629, 3, 1, '(derived)', 'zeta:M+D'),
    ('Sec. II-B', 'converged rows of the bank', 7324, 0, 1, '(derived)', 'bank:rows'),
    ('Sec. II-B', 'unstable rows of the bank', 1120, 0, 1, '(derived)', 'bank:unstable'),
    ('Sec. II-C', 'labels that flip under constant power', 3310, 0, 1, '(derived)', 'flips:count'),
    ('Sec. II-C', 'labels that flip, %', 45.2, 1, 100, '(derived)', 'flips:share'),
    ('Sec. III-A', 'GRIP minus lam_min, AUC', 0.019, 3, 1, 'results/tables/r20/r20_cheap.json', 'dlg/dauc'),
    ('Sec. III-A', 'GRIP minus lam_min, DeLong p', 2.7e-13, 14, 1, 'results/tables/r20/r20_cheap.json', 'dlg/delong_p'),
    ('Sec. III-B', 'tier edge at risk 1e-3', -3.631, 3, 1, 'results/final/tables/e1_tier_edges.csv', 'row risk_level=0.001; column edge_mhat'),
    ('Sec. III-B', 'tier edge at risk 1e-1', -1.87, 3, 1, 'results/final/tables/e1_tier_edges.csv', 'row risk_level=0.1; column edge_mhat'),
    ('Fig. 6', 'candidate edge at risk 1e-2', -2.933, 3, 1, 'results/final/tables/e1_tier_edges.csv', 'row risk_level=0.01; column edge_mhat'),
    ('Sec. III-D', 'development pool (train + test)', 4358, 0, 1, '(derived)', 'pool'),
    ('Sec. III-D', 'Bonferroni p, XGBoost vs AMBER (28 x p)', 4.4e-06, 7, 1, '(derived)', 'bonferroni'),
    ('Sec. III-D', 'same-seed FTT-reg leads with p < 0.05', 6, 0, 1, 'results/tables/r19/r19_fttr_summary.csv', 'row convention=lead_sameseed; column n_p_lt_05'),
    ('Sec. III-D', 'same-seed FTT-reg leads with p < 0.05/28', 2, 0, 1, '(derived)', 'fttr:below_bonf'),
    ('Sec. IV-A', 'XGBoost minus AMBER, ensembles', 0.0246, 4, -1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column dAUC'),
    ('Sec. IV-A', 'XGBoost ensemble AUC', 0.957, 3, 1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column AUC_xgb'),
    ('Sec. IV-A', 'AMBER ensemble AUC', 0.932, 3, 1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column AUC_ebm'),
    ('Sec. IV-A', 'AMBER minus XGBoost, 95% CI low', -0.035, 3, 1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column dAUC_lo'),
    ('Sec. IV-A', 'AMBER minus XGBoost, 95% CI high', -0.016, 3, 1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column dAUC_hi'),
    ('Sec. IV-A', 'AMBER minus XGBoost, DeLong p', 1.6e-07, 8, 1, 'results/final/tables/e8_paired_frontier.csv', 'row pairing=seed-mean; column delong_p'),
    ('Sec. IV-A', 'XGBoost minus AMBER over ten resplits, mean', 0.0199, 4, 1, '(derived)', 'resplit:mean'),
    ('Sec. IV-A', 'XGBoost minus AMBER over ten resplits, sd', 0.0038, 4, 1, '(derived)', 'resplit:sd'),
    ('Sec. IV-A', 'resplits with XGBoost ahead', 10, 0, 1, '(derived)', 'resplit:positive'),
    ('Sec. IV-A', 'FTT-reg minus AMBER, ensembles', 0.0294, 4, -1, 'results/tables/r17/e2r17_fttp_paired.csv', 'row pairing=seed-mean; column dAUC'),
    ('Sec. IV-A', 'FTT-reg same-seed lead, mean', 0.0145, 4, 1, 'results/tables/r19/r19_fttr_summary.csv', 'row convention=lead_sameseed; column mean'),
    ('Sec. IV-A', 'FTT-reg same-seed lead, sd', 0.0086, 4, 1, 'results/tables/r19/r19_fttr_summary.csv', 'row convention=lead_sameseed; column sd_ddof1'),
    ('Sec. IV-A', 'EMS configuration, XGBoost lead', 0.0237, 4, -1, 'results/tables/r18/r18_pidx.csv', 'row pairing=seed-mean; column dAUC'),
    ('Sec. IV-A', 'EMS configuration, FTT-reg lead', 0.0289, 4, -1, 'results/tables/r18/r18_fttp_pidx.csv', 'row pairing=seed-mean; column dAUC'),
    ('Sec. IV-A', 'reactive channel at medians, AUC', 0.421, 3, 1, 'results/tables/r15/r15_dropout_ablation.csv', 'row group=Qinv; column AUC'),
    ('Sec. IV-A', 'reactive channel at medians, recall %', 28.6, 1, 100, 'results/tables/r15/r15_dropout_ablation.csv', 'row group=Qinv; column recall'),
    ('Sec. IV-A', 'median cosine at zero frequency', 0.27, 2, 1, 'results/final/e9_ma2024_threshold.json', 'results/device_check/cosine_s0/q50'),
    ('Sec. IV-B', 'rows attempted', 313, 0, 1, 'results/tables/r21/r21_faith_md_side.json', 'results/ledger/attempted'),
    ('Sec. IV-B', 'removed by the cap-region rule', 73, 0, 1, 'results/tables/r21/r21_faith_md_side.json', 'results/ledger/removed_cap_region_total'),
    ('Sec. IV-B', 'lost by the eight-probe entry rule', 4, 0, 1, 'results/tables/r21/r21_faith_md_side.json', 'results/ledger/lost_probe_nonconvergence'),
    ('Sec. IV-B', 'scenarios kept', 236, 0, 1, 'results/tables/r21/r21_faith_md_side.json', 'results/ledger/kept'),
    ('Sec. IV-B', 'unstable-side scenarios kept', 25, 0, 1, 'results/tables/r21/r21_faith_md_side.json', 'results/ledger/unstable_side_kept_below_cap'),
    ('Sec. IV-B', 'displayed ordering vs sensitivities, median', 0.09, 2, 1, 'results/tables/r19/r19_disp_slack.json', 'results/displayed_vs_analytic/excl/median'),
    ('Sec. IV-B', 'same, slack pairs inside', 0.01, 2, 1, 'results/tables/r19/r19_disp_slack.json', 'results/displayed_vs_analytic/all/median'),
    ('Sec. IV-C', 'unstable test points, %', 17.7, 1, 100, '(derived)', 'econ:unstable (m >= 0)|share_of_test'),
    ('Sec. IV-C', 'false alarms within 0.5 of the boundary, %', 87, 0, 100, '(derived)', 'econ:(-0.5, 0.0]|FA_rate'),
    ('Sec. IV-C', 'false alarms beyond 4, %', 4.7, 1, 100, '(derived)', 'econ:(-inf, -4.0]|FA_rate'),
    ('Sec. IV-C', 'recourse rows (verified unsafe)', 90, 0, 1, '(derived)', 'rec:n'),
    ('Sec. IV-C', 'recourse stabilizes, %', 87, 0, 100, '(derived)', 'rec:stabilised'),
    ('Sec. IV-C', 'recourse clears the alarm, %', 70, 0, 100, '(derived)', 'rec:clears'),
    ('Sec. IV-C', 'recourse median MW (stabilized rows)', 245, 0, 1, '(derived)', 'rec:cost_mw'),
    ('Sec. IV-C', 'recourse median full-model evaluations', 75, 0, 1, '(derived)', 'rec:evals'),
    ('Sec. IV-C', 'recourse median s per case', 52.35, 2, 1, '(derived)', 'rec:wall_s'),
    ('Sec. IV-C', 'routed occupancy, mixture with 90% deep', 0.88, 2, 1, 'results/tables/r17/r17_econ_scenarios.csv', 'row scenario=utility_mix_A; column alpha'),
    ('Sec. IV-C', 'routed occupancy, mixture with 99% deep', 0.86, 2, 1, 'results/tables/r17/r17_econ_scenarios.csv', 'row scenario=quiet_fleet_B; column alpha'),
    ('Sec. IV-D', 'constant-power monitor, AUC', 0.976, 3, 1, 'results/tables/r16/r16_cp_retrain.csv', 'row n_test=1308; column retrain_AUC_mean'),
    ('Sec. I', 'index of [13], AUC (min over buses, test set)', 0.722, 3, 1, '(derived)', 'jvsm:auc'),
    ('Sec. I', 'index of [13], test rows', 1308, 0, 1, '(derived)', 'jvsm:n'),
    ('Sec. II-B', 'draws of the bank (converged + ledgered failures)', 8500, 0, 1, '(derived)', 'ledger:draws'),
    ('Sec. II-B', 'converged draws, %', 86.2, 1, 100, '(derived)', 'ledger:converged'),
    ('Sec. II-B', 'power-flow non-convergences', 748, 0, 1, '(derived)', 'ledger:powerflow'),
    ('Sec. II-B', 'reference-bus losses', 428, 0, 1, '(derived)', 'ledger:refbus'),
    ('Sec. II-B', 'failed draws in one-GFM configurations, %', 37, 0, 100, '(derived)', 'ledger:onegfm'),
    ('Sec. II-B', 'converged draws in one-GFM configurations, %', 23, 0, 100, '(derived)', 'bank:onegfm'),
    ('Sec. II-B', 'label audit N_mis over the bank', 0, 0, 1, '(derived)', 'bank:nmis'),
    ('Sec. II-C', 'crossing pair real within this distance of k*', 0.007, 3, 1, '(derived)', 'cross:gap'),
    ('Sec. III-A', 'multi-GFM rows of the drift study', 5626, 0, 1, '(derived)', 'drift:n'),
    ('Sec. III-A', 'Spearman of H-bar with the worst-case erosion', 0.6, 2, 1, '(derived)', 'drift:rho'),
    ('Sec. IV-A', 'line-impedance scale added, DeLong p', 0.25, 2, 1, 'results/tables/r21/r21_design_baseline.json', 'item8/paired_MDk_vs_MD/delong_p'),
    ('Sec. IV-A', 'line-impedance scale added, AUC gain (ensembles)', 0.003, 3, 1, 'results/tables/r21/r21_design_baseline.json', 'item8/paired_MDk_vs_MD/dAUC'),
    ('Sec. III-A', 'features zero throughout the training pool', 12, 0, 1, 'results/tables/r13/r13_results.json', 'task10/implied_structural_zero_features; count of *'),
    ('Sec. IV-A', 'stable-side R^2 gap, deployed minus flags arm', 0.15, 2, 1, '(derived)', 'gap:gap_point'),
    ('Sec. IV-A', 'same gap, 95% CI low', 0.1, 2, 1, '(derived)', 'gap:gap_ci_lo'),
    ('Sec. IV-A', 'same gap, 95% CI high', 0.2, 2, 1, '(derived)', 'gap:gap_ci_hi'),
    ('Sec. IV-A', 'Secure test rows of the flags arm', 58, 0, 1, 'results/tables/r17/r17_mfl_tiers.csv', 'row frame=test, tier_index=0; column n'),
    ('Sec. IV-B', 'per-plant sign consistency (reactive), lowest', 0.61, 2, 1, '(derived)', 'signc:Q|min'),
    ('Sec. IV-B', 'per-plant sign consistency (reactive), highest', 0.85, 2, 1, '(derived)', 'signc:Q|max'),
    ('Sec. IV-B', 'type indicators, AUC (five-seed mean)', 0.931, 3, 1, '(derived)', 'enc:AUC'),
    ('Sec. IV-B', 'deployed encoding in the same run, AUC', 0.932, 3, 1, '(derived)', 'enc:AUC_base'),
    ('Sec. IV-B', 'importance rank of fpll_10 with type indicators', 26, 0, 1, '(derived)', 'rank:fpll_10'),
    ('Sec. IV-C', "frozen bank's Secure collapse, prediction (seed mean)", -3.8, 1, 1, 'results/tables/r19/r19_frzc_secure.csv', 'row convention=seedmean, id=5588; column pred'),
    ('Sec. IV-C', 'alarm band alone: points covered, %', 82.0, 1, 100, '(derived)', 'abst:0.30|coverage'),
    ('Sec. IV-C', 'alarm band alone: recall, %', 95.3, 1, 100, '(derived)', 'abst:0.30|recall_on_covered'),
    ('Sec. IV-C', 'alarm band alone: false alarms, %', 22.7, 1, 100, '(derived)', 'abst:0.30|FA_on_covered'),
    ('Sec. IV-C', 'recourse median redispatch, % of system load', 4.1, 1, 1, '(derived)', 'recload:median'),
    ('Sec. IV-C', '4,800 cases: monitor, ms', 13, 0, 1, '(derived)', 'timing:ebm_ms'),
    ('Sec. IV-C', '4,800 cases: power flow, CPU-min', 6.9, 1, 1, '(derived)', 'timing:pf_min'),
    ('Sec. IV-C', '4,800 cases: full model, CPU-min', 75, 0, 1, '(derived)', 'timing:full_min'),
    ('Sec. IV-A', 'AMBER on the frozen forward bank, five-seed mean AUC', 0.93, 3, 1, '(derived)', 'frozen:M+D|ebm|AUC_freeze'),
    ('Sec. IV-C', "a random order's first 557 points, unstable points expected", 98, 0, 1, '(derived)', 'alarm:M+D|ebm|blind'),
    ('Sec. IV-C', 'XGBoost alarm (M+D, seed 0), unstable points held', 217, 0, 1, '(derived)', 'alarm:M+D|xgb|caught'),
    ('Sec. IV-C', 'XGBoost alarm (M+D, seed 0), alarms', 482, 0, 1, '(derived)', 'alarm:M+D|xgb|alarms'),
    ('Sec. IV-D', 'design-ray k* with 30% constant power (tau 0.020)', 1.415, 3, 1, 'results/final/tables/q7_cpl_ray.csv', 'row alpha=0.30, tau=0.020; column kstar'),
    ('Sec. IV-C', "the screen's first points (its alarms)", 557, 0, 1, '(derived)', 'alarm:M+D|ebm|alarms'),
    ('Sec. IV-C', "unstable points among the screen's alarms", 216, 0, 1, '(derived)', 'alarm:M+D|ebm|caught'),
    ('Sec. IV-D', 'design-ray k* with no constant power', 1.973, 3, 1, 'results/final/tables/q7_cpl_ray.csv', 'row alpha=0.00, tau=NaN; column kstar'),
    ('Sec. IV-D', 'design-ray k* with 60% constant power (tau 0.020)', 0.964, 3, 1, 'results/final/tables/q7_cpl_ray.csv', 'row alpha=0.60, tau=0.020; column kstar'),
    ('Sec. IV-A', 'frozen forward bank, seed-0 alarm: recall, %', 94.6, 1, 100, 'results/tables/r18/r18_freeze_score.csv', 'row config=M+D, model=ebm, seed=0; column recall_freeze'),
    ('Sec. IV-A', 'frozen forward bank, seed-0 alarm: false alarms, %', 36.0, 1, 100, 'results/tables/r18/r18_freeze_score.csv', 'row config=M+D, model=ebm, seed=0; column FA_freeze'),
    ('Sec. IV-A', 'noisy inputs at the frozen threshold: recall, %', 96.9, 1, 100, '(derived)', 'noise:ebm|recall'),
    ('Sec. IV-A', 'noisy inputs at the frozen threshold: false alarms, %', 52.6, 1, 100, '(derived)', 'noise:ebm|FA'),
    ('Sec. IV-B', 'device-data probes, 2% of the range: median rank correlation', 0.322, 3, 1, 'results/device_probes/device_probes_fidelity.json', 'results/s0.02/rho_median'),
    ('Sec. IV-B', 'device-data probes, 2%: bootstrap 95% CI low', 0.279, 3, 1, 'results/device_probes/device_probes_fidelity.json', 'results/s0.02/ci_lo'),
    ('Sec. IV-B', 'device-data probes, 2%: bootstrap 95% CI high', 0.364, 3, 1, 'results/device_probes/device_probes_fidelity.json', 'results/s0.02/ci_hi'),
    ('Sec. IV-B', 'device-data probes, 10% of the range: median', 0.469, 3, 1, 'results/device_probes/device_probes_fidelity.json', 'results/s0.10/rho_median'),
    ('Sec. IV-B', 'device-data probes, 2%: scenarios failing their own null, %', 65.7, 1, 100, '(derived)', 'devprobe:s0.02|fail_share'),
    ('Sec. IV-B', 'device-data probes, 2%: pooled null, 1/p (p = 1/1,001 < 0.001)', 1001, 0, 1, '(derived)', 'devprobe:s0.02|inv_p'),
    ('Sec. IV-D', 'constant-power monitor at 30%, AUC (five-seed mean)', 0.957, 3, 1, 'results/cp30/cp30_retrain.csv', 'row n_test=1308; column retrain_AUC_mean'),
    ('Sec. IV-D', 'constant-power monitor at 30%, sd over the five seeds', 0.001, 3, 1, '(derived)', 'cpsd:results/cp30/cp30_retrain_raw.csv'),
    ('Sec. IV-D', 'constant-power monitor at 30%, recalibrated alarm: recall, %', 92.6, 1, 100, 'results/cp30/cp30_retrain.csv', 'row n_test=1308; column retrain_recall'),
    ('Sec. IV-D', 'constant-power monitor at 30%, recalibrated alarm: false alarms, %', 16.4, 1, 100, 'results/cp30/cp30_retrain.csv', 'row n_test=1308; column retrain_FA'),
    ('Sec. IV-D', 'constant-impedance monitor on the 30% labels, AUC', 0.79, 2, 1, 'results/cp30/cp30_retrain.csv', 'row n_test=1308; column frozen_cz_AUC_vs_cp'),
    ('Sec. IV-D', 'constant-impedance monitor on the 30% labels, recall, %', 71, 0, 100, 'results/cp30/cp30_retrain.csv', 'row n_test=1308; column frozen_cz_recall_vs_cp'),
    ('Sec. IV-D', 'constant-power monitor at 60%, sd over the five seeds', 0.001, 3, 1, '(derived)', 'cpsd:results/tables/r16/r16_cp_retrain_raw.csv'),
    ('Sec. IV-D', 'constant-power monitor at 60%, recalibrated alarm: recall, %', 96.4, 1, 100, 'results/tables/r16/r16_cp_retrain.csv', 'row n_test=1308; column retrain_recall'),
    ('Sec. IV-D', 'constant-power monitor at 60%, recalibrated alarm: false alarms, %', 14.4, 1, 100, 'results/tables/r16/r16_cp_retrain.csv', 'row n_test=1308; column retrain_FA'),
    ('Sec. IV-D', 'constant-impedance monitor on the 60% labels, AUC', 0.66, 2, 1, 'results/tables/r16/r16_cp_retrain.csv', 'row n_test=1308; column frozen_cz_AUC_vs_cp'),
    ('Sec. IV-D', 'constant-impedance monitor on the 60% labels, recall, %', 53, 0, 100, 'results/tables/r16/r16_cp_retrain.csv', 'row n_test=1308; column frozen_cz_recall_vs_cp'),
    ('Sec. I', 'low gSCR on the bank, AUC', 0.244, 3, 1, 'results/tables/r15/r15_results.json', 'B/gates/gSCR_bank_vs_0.244'),
    ('Sec. III-A', 'additive terms of the deployed monitor', 88, 0, 1, 'results/tables/r15/r15_results.json', 'H/n_terms'),
    ('Sec. III-A', "terms that reach 80% of a row's absolute contribution, median", 28, 0, 1, 'results/tables/r15/r15_results.json', 'H/k80_median'),
    ('Sec. II-A', 'refit on the uncapped target, AUC (five-seed mean)', 0.933, 3, 1, 'results/tables/r15/r15_results.json', 'L/AUC_mean'),
    ('Sec. II-A', 'refit on the uncapped target, sd over the five seeds', 0.002, 3, 1, 'results/tables/r15/r15_results.json', 'L/AUC_sd'),
    ('Sec. II-C', 'bank ray 11, point beyond k*: divergence time in a 30 s window, s', 17.1, 1, 1, 'results/tables/r20/r20_straddle2.csv', 'row rayid=11, case_=collapse30; column tdiv'),
    ('Sec. IV-A', 'configuration cells of the test frame', 77, 0, 1, 'results/tables/r20/r20_cheap.json', 'clus/n_clusters'),
    ('Sec. IV-A', 'AMBER AUC, configuration-cell bootstrap 95% interval, low', 0.88, 2, 1, 'results/tables/r20/r20_cheap.json', 'clus/ci_lo'),
    ('Sec. IV-A', 'AMBER AUC, configuration-cell bootstrap 95% interval, high', 0.96, 2, 1, 'results/tables/r20/r20_cheap.json', 'clus/ci_hi'),
    ('Sec. IV-C', 'routed occupancy on the test frame', 0.92, 2, 1, 'results/tables/r17/r17_econ_scenarios.csv', 'row scenario=test_frame; column alpha'),
    ('Sec. IV-C', 'routed occupancy on the frozen forward bank (five-seed mean prediction)', 0.94, 2, 1, 'results/tables/r19/r19_frzc_census.csv', 'row convention=seedmean, row=routed; column rate'),
    ('Sec. IV-C', "frozen bank's Secure collapse, true growth rate, s^-1", 33.6, 1, 1, 'results/tables/r19/r19_frzc_secure.csv', 'row convention=seedmean, id=5588; column sigma_dom'),
    ('Sec. IV-B', 'device-data probes, 2%: median from the per-scenario rows', 0.322, 3, 1, '(derived)', 'devrows:s0.02|median'),
    ('Sec. IV-B', 'device-data probes, 10%: median from the per-scenario rows', 0.469, 3, 1, '(derived)', 'devrows:s0.10|median'),
    ('Sec. IV-B', 'device-data probes, 2%: scenarios failing their own null, % (per-scenario nulls)', 65.7, 1, 100, '(derived)', 'devrows:s0.02|fail_share'),
]


def rows(rel):
    with (ROOT / rel).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _ranks(x):
    """average ranks (1-based); tied values share the mean of their ranks"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _auc(score, label):
    """probability that a positive row scores above a negative one (ties 1/2)"""
    r = _ranks(score)
    npos = sum(label)
    nneg = len(label) - npos
    return (sum(ri for ri, y in zip(r, label) if y) - npos * (npos + 1) / 2) / (npos * nneg)


def _spearman(a, b):
    """Pearson correlation of the average ranks"""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def one(rs, what):
    if len(rs) != 1:
        raise ValueError(f"{len(rs)} rows match {what}")
    return rs[0]


def _count(x, what):
    """a count recovered as a rate times a count: it must be a whole number"""
    k = round(x)
    if abs(x - k) > 1e-6:
        raise ValueError(f"{what}: {x} is not a whole number")
    return k


def derived(key):
    """values that follow from shipped files by a stated operation"""
    kind, _, arg = key.partition(":")
    if kind == "resplit":       # per split: seed-mean AUC of XGBoost minus AMBER
        by = {}
        for r in rows("results/tables/r15/r15_repeated_splits_raw.csv"):
            by.setdefault((r["split"], r["model"]), []).append(float(r["AUC"]))
        d = [st.mean(by[(s, "xgb")]) - st.mean(by[(s, "ebm")])
             for s in sorted({s for s, _ in by}, key=int)]
        return {"mean": st.mean(d), "sd": st.stdev(d), "positive": sum(x > 0 for x in d)}[arg]
    if kind == "bank":          # data/dataset.csv: unstable means sigma_dom >= 0
        d = rows("data/dataset.csv")
        s = [float(r["sigma_dom"]) for r in d]
        u = sum(v >= 0 for v in s)
        if arg == "onegfm":     # share of converged draws with one GFM plant
            return sum(r["nGFM"] == "1" for r in d) / len(d)
        if arg == "nmis":       # the label check: stored class 0 (Unstable),
            #                     zeta_min <= 0 and sigma_dom >= 0 must agree
            return (sum(abs((r["class"] == "0") - (float(r["zeta_min"]) <= 0)) for r in d)
                    + sum(abs((float(r["zeta_min"]) <= 0) - (float(r["sigma_dom"]) >= 0))
                          for r in d))
        if arg == "above_cap":  # rows above the +1 s^-1 cap of the training target
            return sum(v > 1 for v in s) / len(s)
        return {"rows": len(s), "unstable": u, "share": u / len(s),
                "min_stable": min(v for v in s if v < 0)}[arg]
    if kind == "ledger":        # one row per failed draw of the bank
        led = rows("results/tables/r15/r15_failure_ledger.csv")
        n_ok = len(rows("data/dataset.csv"))
        return {"powerflow": lambda: sum(r["fail"] == "powerflow" for r in led),
                "refbus": lambda: sum("ensure_ref_node" in r["fail"] for r in led),
                "onegfm": lambda: sum(r["nGFM"] == "1" for r in led) / len(led),
                "draws": lambda: n_ok + len(led),
                "converged": lambda: n_ok / (n_ok + len(led))}[arg]()
    if kind == "jvsm":          # the index of [13] per test row; risk = low index
        r = rows("results/tables/r15/r15_jvsm_test.csv")
        if arg == "n":
            return len(r)
        return _auc([-float(x["jvsm_min"]) for x in r], [float(x["sigma_row"]) >= 0 for x in r])
    if kind == "drift":         # both drift files, one row per id (first file
        seen = {}               # first: every id of the second file is also in
        #                         the first, with the same trace_ok, so the first
        #                         decides), traced rows with two or more GFM plants
        for rel in ("results/final/tables/q1_hbar_drift.csv",
                    "results/final/tables/q1_hbar_drift_b.csv"):
            for x in rows(rel):
                if x["id"] and x["id"] not in seen:
                    seen[x["id"]] = x
        sel = [x for x in seen.values() if x["trace_ok"] == "1" and int(x["nGFM"]) >= 2]
        if arg == "n":
            return len(sel)
        return _spearman([float(x["Hbar_row"]) for x in sel],
                         [float(x["ero10"]) / float(x["lam_min_row"]) for x in sel])
    if kind == "cross":         # first k of the continuation where the unstable
        #                         mode is real, minus k* of the same ray
        q = rows("results/final/tables/q3_continuation.csv")
        k_real = min(float(x["k"]) for x in q if x["pf_ok"] == "1"
                     and float(x["f_dom"]) == 0 and float(x["sigma_dom"]) > 0)
        b = one([x for x in rows("results/feasibility/fs3_boundary.csv")
                 if x["ray"] == "k-sweep kL=1.0"], "ray k-sweep kL=1.0")
        return k_real - float(b["kstar_full"])
    if kind == "gap":           # one row: deployed minus flags arm, stable-side R^2
        return float(one(rows("results/tables/r19/r19_ssci_gap.csv"), "gap")[arg])
    if kind == "signc":         # per-plant sign consistency of one kind (P or Q)
        k, fn = arg.split("|")
        v = [float(x["consistency"]) for x in rows("results/final/tables/e2_sign_ceiling.csv")
             if x["kind"] == k]
        return {"min": min, "max": max}[fn](v)
    if kind == "enc":           # five-seed mean of the encoding ablation
        return st.mean(float(x[arg]) for x in rows("results/final/tables/e8_encoding_ablation.csv"))
    if kind == "rank":          # 1 + number of terms with a larger importance
        r = sorted(rows("results/final/tables/e8_ablation_importances.csv"),
                   key=lambda x: -float(x["importance"]))
        return 1 + [x["term"] for x in r].index(arg)
    if kind == "abst":          # one half-width of the abstention curve
        w, col = arg.split("|")
        r = one([x for x in rows("results/final/tables/e1_abstention_curve.csv")
                 if abs(float(x["w"]) - float(w)) < 1e-9], f"w={w}")
        return float(r[col])
    if kind == "recload":       # recourse rows that stabilize: median of the
        #                         redispatch over the scenario's load (Ptot, 100 MVA pu)
        p = {r["id"]: float(r["Ptot"]) for r in rows("data/dataset.csv")}
        v = [100.0 * float(x["cost_mw"]) / (100.0 * p[x["id"]])
             for x in rows("results/tables/r17/r17_rec.csv")
             if float(x["n_evals"]) > 0 and x["stabilised"] == "1"]
        return st.median(v)
    if kind == "timing":        # the assumed 4,800 cases per window (200 x 24)
        py = {x["model"]: x for x in rows("results/final/tables/e6_timing_python.csv")}
        ml = rows("results/final/tables/e6_timing_matlab.csv")

        def stage(p):
            return float(one([x for x in ml if x["stage"].startswith(p)], p)["median_wall_s"])
        return {"ebm_ms": lambda: 4800 * float(py["ebm"]["inference_us_per_sample"]) / 1000,
                "pf_min": lambda: 4800 * stage("power flow") / 60,
                "full_min": lambda: 4800 * stage("full model") / 60}[arg]()
    if kind == "zeta":          # mean over the five seeds of one configuration
        return st.mean(float(r["R2"]) for r in rows("results/final/tables/e1_zeta_sidequest.csv")
                       if r["config"] == arg)
    if kind == "flips":         # labels that change class under constant power
        r = rows("results/tables/r16/r16_cp_relabel.csv")
        n = sum(x["flip_bin"] == "1" for x in r)
        return {"count": n, "share": n / len(r)}[arg]
    if kind == "pool":          # the rows left after the three held-out frames
        d = json.loads((ROOT / "results/tables/r21/r21_design_baseline.json")
                       .read_text(encoding="utf-8"))["data_facts"]["sizes"]
        return d["train"] + d["test"]
    if kind == "bonferroni":    # 28 paired tests
        return 28 * value("results/final/tables/e8_paired_frontier.csv",
                          "row pairing=seed-mean; column delong_p")
    if kind == "fttr":          # same-seed leads below 0.05/28
        return sum(float(r["delong_p"]) < 0.05 / 28
                   for r in rows("results/tables/r19/r19_fttr_raw.csv"))
    if kind == "rec":           # the verified-unsafe rows: base decay rate > 0
        r = [x for x in rows("results/tables/r17/r17_rec.csv") if float(x["sigma0_chk"]) > 0]
        return {"n": lambda: len(r),
                "stabilised": lambda: st.mean(float(x["stabilised"]) for x in r),
                "clears": lambda: st.mean(float(x["clears_alarm"]) for x in r),
                "cost_mw": lambda: st.median(float(x["cost_mw"]) for x in r if x["stabilised"] == "1"),
                "evals": lambda: st.median(float(x["n_evals"]) for x in r),
                "wall_s": lambda: st.median(float(x["wall_s"]) for x in r)}[arg]()
    if kind == "econ":          # one bin of the margin curve of the test set
        b, col = arg.split("|")
        r = [x for x in rows("results/tables/r17/r17_econ_curve.csv") if x["bin"] == b]
        if len(r) != 1:
            raise ValueError(f"{len(r)} rows match {b}")
        return float(r[0][col])
    if kind == "alarm":         # one test alarm (seed 0) of e1_alarms.csv over the
        #                         test counts of e1_tiers.csv: unstable points held
        #                         (recall x unstable rows), alarms (those plus the
        #                         false-alarm rate x stable rows); "blind": the
        #                         unstable points expected among as many points
        #                         taken in a random order (its expectation)
        cfg, model, what = arg.split("|")
        t = [x for x in rows("results/final/tables/e1_tiers.csv") if x["frame"] == "test"]
        n = sum(int(x["n"]) for x in t)
        n_u = _count(sum(int(x["n"]) * float(x["collapse_rate"]) for x in t), "unstable test rows")
        a = one([x for x in rows("results/final/tables/e1_alarms.csv")
                 if x["config"] == cfg and x["model"] == model], f"{cfg} {model}")
        held = _count(float(a["test_recall"]) * n_u, "alarmed unstable rows")
        alarms = held + _count(float(a["test_FA"]) * (n - n_u), "alarmed stable rows")
        return {"caught": held, "alarms": alarms, "blind": alarms * n_u / n}[what]
    if kind == "noise":         # mean over the five noise draws of one model
        model, col = arg.split("|")
        v = [float(x[col]) for x in rows("results/tables/r15/r15_noise_alarm_rows.csv")
             if x["model"] == model]
        if len(v) != 5:
            raise ValueError(f"{len(v)} noise draws of {model}, not 5")
        return st.mean(v)
    if kind == "frozen":        # mean over the five seeds, frozen forward bank
        cfg, model, col = arg.split("|")
        v = [float(x[col]) for x in rows("results/tables/r18/r18_freeze_score.csv")
             if x["config"] == cfg and x["model"] == model]
        if len(v) != 5:
            raise ValueError(f"{len(v)} seed rows of {cfg} {model}, not 5")
        return st.mean(v)
    if kind == "devprobe":      # the device-data probes at one scale: the share
        #                         of scenarios failing their own null, and 1/p of
        #                         the pooled null (1,001 when no permuted median
        #                         of 1,000 reached the observed one)
        s, what = arg.split("|")
        r = json.loads((ROOT / "results/device_probes/device_probes_fidelity.json")
                       .read_text(encoding="utf-8"))["results"][s]
        return {"fail_share": lambda: r["n_failing_own_null"] / r["n_scenarios"],
                "inv_p": lambda: 1 / r["pooled_null"]["p_value"]}[what]()
    if kind == "cpsd":          # sample sd (ddof 1) of the five retrain seeds
        v = [float(x["AUC"]) for x in rows(arg)]
        if len(v) != 5:
            raise ValueError(f"{len(v)} seeds in {arg}, not 5")
        return st.stdev(v)
    if kind == "devrows":       # the device-data probes from their per-scenario
        #                         files: the median of rho_ebm over the scenarios,
        #                         and the share of scenarios below their own
        #                         null's 95% quantile (below_null_q95 = 1)
        s, what = arg.split("|")
        if what == "median":
            v = [float(x["rho_ebm"]) for x in
                 rows(f"results/device_probes/device_probes_rows_{s}.csv")
                 if x["rho_ebm"] not in ("", "nan", "NaN")]
            return st.median(v)
        r = rows(f"results/device_probes/device_probes_null_{s}.csv")
        return sum(int(x["below_null_q95"]) for x in r) / len(r)
    raise KeyError(key)


def value(rel, where):
    p = ROOT / rel
    if rel.endswith(".json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        path, _, count = where.partition("; count of ")
        for k in path.split("/"):
            d = d[int(k)] if isinstance(d, list) else d[k]
        if count:
            return float(sum(1 for x in d if str(x).startswith(count.rstrip("*"))))
        return float(d)
    sel, col = where[len("row "):].split("; column ")
    want = dict(kv.split("=", 1) for kv in sel.split(", "))
    with p.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if all(r.get(k) == v for k, v in want.items())]
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} rows match {sel}")
    return float(rows[0][col])


def main():
    quiet = "--quiet" in sys.argv
    nfail = 0
    print(f"{'check':<62} {'file value':>12} {'printed':>9} {'tol':>7}")
    for ex, label, printed, dec, scale, rel, where in CHECKS:
        tol = 0.5 * 10 ** (-dec)
        try:
            v = (derived(where) if rel == "(derived)" else value(rel, where)) * scale
            ok = abs(v - printed) <= tol + 1e-12
            shown = f"{v:12.6g}"
        except (OSError, KeyError, IndexError, ValueError) as e:
            ok, shown = False, f"{type(e).__name__:>12}"
        nfail += not ok
        if not quiet or not ok:
            name = f"{ex:<10} {label}"[:62]
            print(f"{name:<62} {shown} {printed:9.{dec}f} {tol:7.1g}  "
                  f"{'ok' if ok else 'FAILED'}")
    print(f"\n{len(CHECKS)} checks, {nfail} failed")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
