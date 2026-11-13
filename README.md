# Intrinsically Explainable, Model-Verified Monitoring of the Small-Signal Collapse Margin in 100% Inverter-Based Power Systems

Code and data for the results of

> Amir Hossein Poursaeed, Farhad Namdari, Shuhang Shen, and Peter A. Crossley, "Intrinsically Explainable, Model-Verified Monitoring of the Small-Signal Collapse Margin in 100% Inverter-Based Power Systems," submitted to the *IEEE Transactions on Industry Applications*.

This repository holds the code and data behind the results of the paper. `matlab/` has the MATLAB
model chain and the bank generator. `data/dataset.csv` is the scenario bank of an all-inverter IEEE
39-bus system, with its columns in [data/schema.md](data/schema.md). `python/` has the Python training
and evaluation scripts. `results/` has the result files behind Tables II to V and Figs. 8 and 9.

## Requirements

- MATLAB R2026a with [MATPOWER](https://matpower.org) 8.1 on the MATLAB path, the Statistics and
  Machine Learning Toolbox and the Parallel Computing Toolbox. The `matlab` command must be on the
  system PATH, because `run_all.py` runs the MATLAB steps as `matlab -batch`.
- Python 3.13 and the packages in `requirements.txt` (`pip install -r requirements.txt`), each pinned
  to the version the results were computed with. `check_paper_values.py` needs only the standard library.

## How to run

```
python check_paper_values.py
python run_all.py --list
python run_all.py --all
```

`check_paper_values.py` compares the result files in `results/` with the numbers printed in Tables II
to V and 130 numbers printed in the text (293 checks). When all is well, it ends with the line
`293 checks, 0 failed`. `run_all.py --list` prints the 77 steps, and `run_all.py --all` runs them in
that order and then the check. To run one step, use `python run_all.py STEP`. To run a step and all
later ones, use `python run_all.py --from STEP`. The full chain takes many hours.

On Windows, add `-X utf8` to the python commands (for example `python -X utf8 check_paper_values.py`)
or set PYTHONUTF8=1. Some printed labels hold the Greek letter lambda, and Python writes redirected
output in cp1252.

## Where each result comes from

| In the paper | Step | File |
|---|---|---|
| Table II | `suite`, `reg-metrics`, `design-baseline` | `results/final/tables/e1_model_suite.csv`, `results/final/tables/e1_reg_metrics.csv`, `results/tables/r21/r21_design_baseline.json` |
| Table III | `index-rows`, `index`, `design-baseline` | `results/final/e9_ma2024_index.json` (the M+F row: `r21_design_baseline.json`) |
| Table IV | `faith-slack` | `results/final/r22_faith_slack.json`, `results/final/r22_faith_slack_rows.csv` |
| Table V | `tiers`, `census`, `frozen-census` | `results/final/tables/e1_tiers.csv`, `results/tables/r16/r16_census.csv`, `results/tables/r19/r19_frzc_census.csv` |
| Fig. 8 | `fig8`, then `fig-vector` | `results/svg/r29_fig8_fidelity_h160.svg` (the step `fig8` writes the PNG `results/figures/r29_fig8_fidelity_h160.png`, not shipped) |
| Fig. 9 | `fig9`, then `fig-vector` | `results/svg/r29_fig9_glassbox.svg` (the step `fig9` writes the PNG `results/figures/r29_fig9_glassbox.png`, not shipped) |
| Figs. 1 to 7 | none: diagrams drawn for the paper | the values they show are in the files here; see the rows below |
| Fig. 2: the tick at -4.7 s^-1 | none | `data/dataset.csv`, column `sigma_dom` (the smallest stable-side margin) |
| Fig. 3: k* = 1.9730 and the inset | none | `results/feasibility/fs3_boundary.csv`; the inset: `results/final/e4_straddle.mat` |
| Fig. 4: the three shapes | none | drawn by `thumbnails()` of `python/r24f_gallery.py` |
| Fig. 6: the tier edges | none | `results/final/tables/e1_tier_edges.csv` |

Everything else (the numbers of the text, run times, seeds, side analyses and the file layout) is in [docs/DETAILS.md](docs/DETAILS.md).

## License and citation

MIT, see [LICENSE](LICENSE). MATPOWER is distributed under its own license. To cite the paper, use
[CITATION.cff](CITATION.cff). A BibTeX entry is at the end of [docs/DETAILS.md](docs/DETAILS.md#citation).
