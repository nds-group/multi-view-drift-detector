# KALEIDO: Multi-View Drift Detection for Time Series

KALEIDO is a label-free, predictor-free online drift detector. It applies the
Kolmogorov–Smirnov test to several views of the same stream (raw, derivatives,
STL trend/seasonal, spectrum) inside an adaptive window, and declares a drift
only on a strict majority of views. 

This repository contains the detector, the baselines (ADWIN, KSWIN,
Page-Hinkley, CUSUM, EWMA, WinDrift++), the datasets, saved results, and the
notebooks that produce the paper figures.

Paper: N. Chukhno, A. Bazco-Nogueras, S. Alcalá-Marín, L. E. Chatzieleftheriou,
M. Fiore, "KALEIDO: Multi-View Drift Detection for Time Series."

## Quick start

All paths and commands are relative to the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the install with a small KALEIDO-versus-baselines run:

```bash
python results_all_in_one_code/run_examples_plot.py \
  --datasets random_easy AR4_linear_1 --ids 0
```

Datasets and saved results are checked in; nothing needs to be downloaded.

## Reproduce the paper

| Step | Command | Needed for |
|---|---|---|
| 1. KALEIDO (paper configuration) | `python results_all_in_one_code/KALEIDO_multi_instance.py --features "gradient+season"` | Fig. 8, 9 |
| 2. Baselines | `python results_all_in_one_code/Benchmarks.py` | Fig. 7, 8, 9 |
| 3. View-combination sweep | `python results_all_in_one_code/run_all_features.py` | Fig. 7 |
| 4. Example detections | `python results_all_in_one_code/run_examples_plot.py --datasets all --ids 0` | Fig. 10 |
| 5. Figures | Notebooks in `results_all_in_one_code/notebooks/analysis/` (see below) | All |

Steps 2 and 3 are full experiments, not smoke tests. Saved results for every
step are already in the repository, so step 5 can be run on its own.

> **Outputs are appended, not overwritten.** The metrics CSVs of steps 1–2 and
> the CSVs written by the dataset-generation notebooks are opened in append
> mode. Move or delete the old files before a fresh run.

### Details

- **Views.** `--features` takes `+`-separated names registered in
  `results_all_in_one_code/KALEIDO_TRANSFORMATIONS.py`, e.g.
  `"normal_data+gradient+season"`. Append `+ttest_ind` to add the t-test voter.
- **KALEIDO output:**
  `results_all_in_one_code/config_all_trans_1_data_only/<features>/final/`.
- **Baseline output:** metrics in
  `results_all_in_one_code/benchmark_results/metrics/`, per-stream plots in
  `results_all_in_one_code/benchmark_results/stable_window/`.
- **Sweep.** 511 even-sized combinations of the 10 views. Combinations whose
  final metrics CSV exists are skipped.
- **Examples.** Results are cached in
  `results_all_in_one_code/example_runs/cache/` and summarized in
  `example_runs_<features>.pkl` there. `--no-cache` recomputes; `--help` lists
  options.

## Figures

Notebooks are in `results_all_in_one_code/notebooks/analysis/` and write to
`results_all_in_one_code/figures/`. They set their own working directory.

| Paper figure | Notebook | Output |
|---|---|---|
| Fig. 1 | `drift_case_studies.ipynb` | `drift_case_studies.pdf` |
| Fig. 6 | `plot_stream_examples.ipynb` | `stream_examples.pdf` |
| Fig. 7 | `plot_feature_combinations_violins.ipynb` | `results_combinations_v3.pdf` |
| Fig. 8, 9 | `plot_results_kaleido_vs_benchmarks.ipynb` | `results_v3.pdf`, false-alarm PDFs |
| Fig. 10 | `plot_examples_of_detections.ipynb` | `examples_selected_detectors.pdf`, `examples_all_detectors_part{1,2}.pdf` |

`results_all_in_one_code/notebooks/data_preparation/stable_window_estimation.ipynb`
recomputes the window bounds (`n_min`, paper Sec. V-B) and writes
`stable_windows.csv`. Run it only to recalibrate.

## Datasets

| Paper name | Location | Regenerate with |
|---|---|---|
| Random easy / noisy | `results_all_in_one_code/datasets/synthetic_random_datasets/` | `random_timeseries_generation.ipynb` in that directory |
| AR linear / nonlinear | `results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/` | `linear_and_nonlinear_timeseries_generation.ipynb` in that directory |
| Friedman | `uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv` | `Data_Friedman_RegressionData_Abrupt.ipynb` in that directory |
| Rolling mean | `uncertainty_our_results/uncertainties_TS.npy` | — |
| Operational 4G / 5G (and trends) | `results_all_in_one_code/datasets/INFOCOM/` | Preparation notebook in that directory |

The Operational streams are derived from
[nds-group/ran-updates](https://github.com/nds-group/ran-updates). The raw
`traffic_dataset.parquet` and `info_dataset.parquet` files are not included;
to rebuild the processed files, obtain them from that source and set their
path in the preparation notebook.

## Repository layout

```
results_all_in_one_code/
  KALEIDO_multi_instance.py     KALEIDO runner
  KALEIDO_TRANSFORMATIONS.py    view definitions
  Benchmarks.py                 baseline runner
  run_all_features.py           view-combination sweep
  run_examples_plot.py          small KALEIDO-vs-baselines examples
  datasets/                     processed inputs and generation notebooks
  benchmark_results/            baseline metrics and plots
  example_runs/                 example caches and serialized runs
  notebooks/                    analysis and data-preparation notebooks
  figures/                      generated figures
```
