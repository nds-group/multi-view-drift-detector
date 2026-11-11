"""
Run example streams of each dataset through our detector (KALEIDO, the multi-instance
KALEIDO detector) and through every benchmark (ADWIN, KSWIN, PH, CUSUM, EWMA, WinDrift++ l1 / l12)
and SAVE the detections. No plotting here: the figure is made in
notebooks/analysis/plot_examples_of_detections.ipynb from the
saved results.

Put this file next to KALEIDO_multi_instance.py, Benchmarks.py, ADWIN_CLASS.py and run it
from the project root (the data paths are the same relative paths used in those two scripts):

    python run_examples_plot.py                                  # the 5 datasets of the paper figure, stream 0 of each
    python run_examples_plot.py --datasets all                   # all 12 datasets, stream 0 of each
    python run_examples_plot.py --datasets all --ids 0 3 10      # streams 0, 3 and 10 of every dataset
    python run_examples_plot.py --datasets random_easy AR4_linear_1 --ids-per-dataset 3 0   # one given id per dataset
    python run_examples_plot.py --features normal_data+gradient+season

Outputs (in --out-dir, default results_all_in_one_code/example_runs):
  cache/<dataset>_id<k>_<detector>_<features>.pkl   one pickle per (dataset, stream, detector) -> reruns are skipped
  example_runs_<features>.pkl                        everything the notebook needs: {"ts#id": {ts, id, stream, labels,
                                                     event_length, title, results: {detector: {tp, fp, fn, n_det}}}}
"""
import os
import re
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from KALEIDO_multi_instance import (KALEIDO_multi_instance, boyer_moore_majority_vote,
                                             classify_drift_points, classify_drift_points_infocom,
                                             load_variable_column_csv)
from Benchmarks import CUSUM, EWMA, WinDriftPP
from ADWIN_CLASS import ADWIN
from river import drift

# ============================================================================ configuration
DELTA, METHOD, ALTERNATIVE, DIST_CHANGE_METHOD, STAT_VALUE = 0.05, 'asymp', 'two-sided', 'ks', 0.1
OURS_LABEL = "KALEIDO"
BENCHMARKS = ['ADWIN', 'KSWIN', 'PH', 'CUSUM', 'EWMA', 'WINDRIFT l1', 'WINDRIFT l12']

INFOCOM_TS = ['infocom_4G', 'infocom_4G_trend', 'infocom_5G_add_day', 'infocom_4G_add_day',
              'infocom_5G_add_day_trend', 'infocom_4G_add_day_trend', 'infocom_5G_add_hour',
              'infocom_4G_add_hour', 'infocom_5G_add_hour_trend', 'infocom_4G_add_hour_trend']

# datasets of the paper figure, in panel order, with their panel titles
DEFAULT_DATASETS = ['infocom_4G_add_hour', 'infocom_4G_add_hour_trend', 'AR4_linear_1', 'random_noisy', 'random_easy']
ALL_DATASETS = ['random_easy', 'random_noisy', 'rolling_mean', 'friedman_5_drifts', 'friedman_2_drifts',
                'friedman_no_drifts', 'infocom_5G_add_hour', 'infocom_4G_add_hour',
                'infocom_5G_add_hour_trend', 'infocom_4G_add_hour_trend', 'AR4_linear_1', 'AR4_nonlinear_1']
DATASET_TITLES = {
    'infocom_4G_add_hour':       'Networking',
    'infocom_4G_add_hour_trend': 'Networking (trend)',
    'infocom_5G_add_hour':       'Networking 5G',
    'infocom_5G_add_hour_trend': 'Networking 5G (trend)',
    'AR4_linear_1':              'Synthetic network-like',
    'AR4_nonlinear_1':           'Synthetic nonlinear',
    'random_noisy':              'Synthetic noisy',
    'random_easy':               'Synthetic simple',
    'rolling_mean':              'Rolling mean',
    'friedman_5_drifts':         'Friedman (5 drifts)',
    'friedman_2_drifts':         'Friedman (2 drifts)',
    'friedman_no_drifts':        'Friedman (no drift)',
}

# per-dataset stable windows (same file the two training scripts read)
_stable_windows_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stable_windows.csv")
if os.path.exists(_stable_windows_path):
    _w = pd.read_csv(_stable_windows_path)
    WINDOW_LOOKUP = {r["time_series"]: (int(r["min_window_p95"]), int(r["max_window"])) for _, r in _w.iterrows()}
else:
    WINDOW_LOOKUP = {}


# ============================================================================ data
def load_dataset(timeseries):
    """Returns (data_streams, labels_all, event_length) exactly as the two training scripts build them.
    labels_all is a DataFrame of label indices per stream, or None for the datasets with fixed labels."""
    labels_all, event_length = None, None
    if timeseries == 'AR4_linear_1':
        x = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/time_series_linear_abrupt_100.csv')
        data_streams = x.values
        labels_all = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/drift_labels_linear_abrupt_100.csv')
    elif timeseries == 'AR4_nonlinear_1':
        x = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/time_series_nonlinear_abrupt_100.csv')
        data_streams = x.values
        labels_all = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/drift_labels_nonlinear_abrupt_100.csv')
    elif timeseries == 'rolling_mean':
        u = np.load('uncertainty_our_results/uncertainties_TS.npy')
        data_streams = [u] + [pd.Series(u).rolling(window=w).mean() for w in range(10, 101, 10)]
    elif timeseries in ('friedman', 'friedman_5_drifts'):
        data = pd.read_csv('uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv')
        data_streams = [data['y_1']] if timeseries == 'friedman' else [data['x1'], data['x3'], data['x5'], data['y_1'], data['y_org']]
    elif timeseries == 'friedman_2_drifts':
        data = pd.read_csv('uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv')
        data_streams = [data['x7'], data['x9'], data['x10']]
    elif timeseries == 'friedman_no_drifts':
        data = pd.read_csv('uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv')
        data_streams = [data['x2'], data['x4'], data['x6'], data['x8']]
    elif timeseries == 'random_noisy':
        x = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_random_datasets/time_series_random_noisy_100.csv')
        data_streams = x.values
        labels_all = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_random_datasets/drift_labels_random_noisy_100.csv')
    elif timeseries == 'random_easy':
        x = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_random_datasets/time_series_random_simple_100.csv')
        data_streams = x.values
        labels_all = load_variable_column_csv('results_all_in_one_code/datasets/synthetic_random_datasets/drift_labels_random_simple_100.csv')
    elif timeseries in INFOCOM_TS:
        base = 'results_all_in_one_code/datasets/INFOCOM/'
        gen, trend = ('5G' if '5G' in timeseries else '4G'), timeseries.endswith('_trend')
        gran = 'hour' if 'hour' in timeseries else 'day'
        if timeseries in ('infocom_4G', 'infocom_4G_trend'):
            data = pd.read_csv(f'{base}4G/ts_location_8102f81001516d06_trend_{trend}.csv')
            col = 'trend_sum_byte_dn_clean' if trend else 'sum_byte_dn_clean'
            data_streams = [data[col].values]
            labels_all = ('file', f'{base}4G/event_index_location_8102f81001516d06_trend_{trend}.txt')
            event_length = 7 * 24
        else:
            # NB: the training scripts read the non-trend file for infocom_4G_add_day_trend; kept identical here
            suffix = '_trend' if (trend and timeseries != 'infocom_4G_add_day_trend') else ''
            x = load_variable_column_csv(f'{base}ran_updates_event_{gen} addition_{gran}_granularity{suffix}.csv')
            data_streams = x.values
            labels_all = load_variable_column_csv(f'{base}drift_labels_ran_updates_event_{gen} addition_{gran}_granularity.csv')
            event_length = 7 * 24 if gran == 'hour' else 7
    else:
        raise ValueError(f"unknown dataset {timeseries}")
    return data_streams, labels_all, event_length


def make_labels(timeseries, n, labels_all, item):
    """0/1 label vector of length n for stream `item` (same rules as the training scripts)."""
    labels = np.zeros(n, dtype=int)
    if timeseries == 'rolling_mean':
        idx = [2000, 4500, 13000]
    elif timeseries in ('friedman', 'friedman_5_drifts'):
        idx = [2000, 3000, 5000, 7500, 16000]
    elif timeseries == 'friedman_2_drifts':
        idx = [10000, 12000]
    elif timeseries == 'friedman_no_drifts':
        idx = []
    elif isinstance(labels_all, tuple):                                   # ('file', path)
        idx = np.loadtxt(labels_all[1], delimiter=',', dtype=int, ndmin=1)
    else:
        idx = np.array(labels_all.iloc[item, :].dropna().values).astype(int)
    for i in idx:
        labels[i] = 1
    return labels


# ============================================================================ detectors
def run_ours(data_stream, labels, features, min_eval_window, eval_no_shrink_window, event_length=None):
    """The exact streaming + voting loop of KALEIDO_multi_instance.py (config_all_trans_1_data_only).
    Returns dict(tp=[(mid, start, end)], fp=[(mid, start, end)], fn=[idx], n_det=int)."""
    tokens = features.split('+')
    multi = KALEIDO_multi_instance(
        delta=DELTA, method=METHOD, alternative=ALTERNATIVE, min_eval_window=min_eval_window,
        eval_noshrink_window=eval_no_shrink_window, dist_change_method=DIST_CHANGE_METHOD, stat_value=STAT_VALUE,
        alternatives=[t for t in tokens if t != 'ttest_ind'], use_ttest='ttest_ind' in tokens)
    multi.instantiate(transformations=[['normal_data', multi.transf_normal_data]])

    vote_results, hist = [], defaultdict(list)
    vote_value = min_eval_window / 2 - 1
    for i, val in tqdm(enumerate(data_stream, start=1), total=len(data_stream), leave=False, desc=OURS_LABEL):
        drifts = multi.update(val, eval=False if (i < vote_value) else True)
        if all(d is None for d in drifts.values()):
            continue
        window_sizes = multi.get_window_sizes()
        pvals, dalts, stats = multi.get_alternatives_pvals(), multi.get_alternatives_drifts(), multi.get_alternatives_stats()
        for name, ws in window_sizes.items():
            if drifts[name] is not None:
                hist[name].append((i, int(np.ceil(i - ws / 2)), drifts[name], ws, True, pvals[name], dalts[name], stats[name]))
                if i >= min_eval_window:
                    for step in range(hist[name][-2][1] + 1, hist[name][-1][1]):
                        hist[name].insert(hist[name].index(hist[name][-1]),
                                          (hist[name][-2][0], step, hist[name][-2][2], hist[name][-2][3], False, [], [], []))
        while True:
            votes = []
            for name in hist:
                v = [t for t in hist[name] if t[1] == vote_value]
                if v:
                    votes.append(v[0][2])
            if len(votes) == len(hist):
                vote_results.append((vote_value, boyer_moore_majority_vote(votes)))
                vote_value += 1
                if vote_results[-1][1] is True:
                    vote_value += max(t[3] for v in hist.values() for t in v if t[1] == (vote_value - 1))
            else:
                break

    transf_results = defaultdict(list)
    for true_drift, _ in [v for v in vote_results if v[1]]:
        for name in hist:
            transf_results[name].append([t for t in hist[name] if t[1] == true_drift][0])

    if event_length is not None:
        TP, FP, TN, FN, _, _ = classify_drift_points_infocom(transf_results, labels, event_length)
    else:
        TP, FP, TN, FN, _, _ = classify_drift_points(transf_results, labels)

    # window extent of every voted drift: (middle, start, end) as used by the classification
    extent = {}
    for name, lst in transf_results.items():
        for t in lst:
            mid, start, end = t[1], t[0] - t[3], t[0]
            s, e = extent.get(mid, (start, end))
            extent[mid] = (min(s, start), max(e, end))
    return dict(tp=[(m, *extent[m]) for m in TP], fp=[(m, *extent[m]) for m in FP], fn=list(FN),
                n_det=len(TP) + len(FP))


def make_benchmark(name, min_eval_window):
    if name == 'ADWIN':
        return ADWIN(delta=DELTA, cut_detector='ADWIN', method=METHOD, min_window_length=int(min_eval_window / 2),
                     grace_period=min_eval_window, alternative=ALTERNATIVE, window_size_filesave=False, clock=1)
    if name == 'KSWIN':
        return drift.KSWIN(alpha=DELTA, seed=42, window_size=min_eval_window, stat_size=int(min_eval_window / 2))
    if name == 'PH':
        return drift.PageHinkley(min_instances=min_eval_window, delta=DELTA, threshold=50.0, alpha=0.9999, mode='both')
    if name == 'CUSUM':
        return CUSUM(threshold=0.5, drift_threshold=50.0, warmup=min_eval_window)
    if name == 'EWMA':
        return EWMA(lambda_=0.2, L=3.0, warmup=min_eval_window)
    m = re.fullmatch(r'WINDRIFT l(\d+)', name)
    if m:
        return WinDriftPP(block_size=int(min_eval_window / 2), win_levels=(int(m.group(1)),), alpha=DELTA,
                          mode='consecutive', cycle_len=None, seed=42)
    raise ValueError(name)


def classify_benchmark(drift_mid, drift_end, window_sizes, labels, event_length=None):
    """classify_drift_points / classify_drift_points_infocom of Benchmarks.py (without the `data_stream` global).
    Returns TP, FP, FN (middle indices / label indices) and the window extent of every detection."""
    n = len(labels)
    lab = labels.copy()
    if event_length is not None:
        for idx in np.where(lab == 1)[0]:
            lab[max(0, idx - event_length + 1): idx + 1] = 1
    TP, FP, FN = [], [], np.arange(n)[lab == 1].tolist()
    track = lab.copy()
    extent = {}
    mids = np.asarray(drift_mid)
    for pos, i in enumerate(drift_mid):
        if i < 0 or i >= n:
            continue
        end = drift_end[pos]
        start = end - window_sizes[end]
        extent[i] = (start, end)
        if lab[start:end].any() and track[start:end].any():
            TP.append(i)
            cand = start + np.where(lab[start:end] == 1)[0]
            idx_t = min(cand, key=lambda x: abs(i - x)) if len(cand) > 1 else start + np.argmax(lab[start:end] == 1)
            if idx_t in FN:
                FN.remove(idx_t)
                if event_length is not None:
                    track[np.where(lab == 1)[0]] = 0
        else:
            FP.append(i)
    if event_length is not None:
        FN = [] if TP else np.arange(n)[labels == 1].tolist()
    return TP, FP, FN, extent


def run_benchmark(name, data_stream, labels, min_eval_window, event_length=None):
    """The streaming loop of Benchmarks.py for one detector. Same return format as run_ours."""
    det = make_benchmark(name, min_eval_window)
    mids, ends, wsz = [], [], []
    for i, val in tqdm(enumerate(data_stream), total=len(data_stream), leave=False, desc=name):
        det.update(val)
        if name == 'ADWIN':
            wsz.append(int(det.width))
        elif name.startswith('WINDRIFT'):
            wsz.append(int(det.width))
        else:
            wsz.append(min_eval_window)
        if det.drift_detected:
            ends.append(i)
            mids.append(i - int(wsz[-1] / 2))
    TP, FP, FN, extent = classify_benchmark(mids, ends, wsz, labels, event_length)
    return dict(tp=[(m, *extent[m]) for m in TP], fp=[(m, *extent[m]) for m in FP], fn=list(FN),
                n_det=len(TP) + len(FP))


# ============================================================================ driver with cache
def run_all(pairs, features, cache_dir, use_cache=True, benchmarks=BENCHMARKS):
    """pairs = [(timeseries, stream id), ...]. Returns {"ts#id": dict(ts, id, stream, labels, event_length, title,
    results={detector: dict(tp=[(mid, start, end)], fp=[(mid, start, end)], fn=[idx], n_det=int)})}."""
    os.makedirs(cache_dir, exist_ok=True)
    out, loaded = {}, {}
    for ts, item in pairs:
        if ts not in loaded:
            loaded[ts] = load_dataset(ts)
        data_streams, labels_all, event_length = loaded[ts]
        if item >= len(data_streams):
            print(f"\n{ts}: only {len(data_streams)} streams, skipping id {item}")
            continue
        stream = pd.Series(data_streams[item]).dropna().values
        labels = make_labels(ts, len(stream), labels_all, item)
        min_w, max_w = WINDOW_LOOKUP.get(ts, (200, 500))
        print(f"\n{ts} (stream {item}, n={len(stream)}, windows {min_w}/{max_w}, drifts at {np.where(labels == 1)[0].tolist()})")

        results = {}
        for det in [OURS_LABEL] + list(benchmarks):
            tag = features if det == OURS_LABEL else det.replace(' ', '_')
            cache = os.path.join(cache_dir, f"{ts}_id{item}_{det.replace(' ', '_')}_{tag}.pkl")
            if use_cache and os.path.exists(cache):
                with open(cache, 'rb') as f:
                    results[det] = pickle.load(f)
            else:
                if det == OURS_LABEL:
                    results[det] = run_ours(stream, labels, features, min_w, max_w, event_length)
                else:
                    results[det] = run_benchmark(det, stream, labels, min_w, event_length)
                with open(cache, 'wb') as f:
                    pickle.dump(results[det], f)
            r = results[det]
            print(f"  {det:14s} TP={len(r['tp'])}  FP={len(r['fp'])}  FN={len(r['fn'])}")
        out[f"{ts}#{item}"] = dict(ts=ts, id=item, stream=stream, labels=labels, event_length=event_length,
                                   title=DATASET_TITLES.get(ts, ts), results=results)
    return out


# ============================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="dataset names, or 'all'")
    ap.add_argument("--ids", nargs="+", type=int, default=[0],
                    help="stream ids to run for EVERY dataset (ids a dataset does not have are skipped)")
    ap.add_argument("--ids-per-dataset", nargs="+", type=int, default=None,
                    help="instead of --ids: exactly one stream id per listed dataset, in the same order")
    ap.add_argument("--features", default="gradient+season", help="feature string of our detector (as in the driver)")
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--out-dir", default="results_all_in_one_code/example_runs")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if a cached pickle exists")
    args = ap.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    if args.ids_per_dataset is not None:
        assert len(args.ids_per_dataset) == len(datasets), "--ids-per-dataset needs one id per dataset"
        pairs = list(zip(datasets, args.ids_per_dataset))
    else:
        pairs = [(ts, i) for ts in datasets for i in args.ids]

    runs = run_all(pairs, args.features, os.path.join(args.out_dir, "cache"),
                   use_cache=not args.no_cache, benchmarks=args.benchmarks)

    # merge with what an earlier run already saved, so the file always holds every example computed so far
    out_file = os.path.join(args.out_dir, f"example_runs_{args.features}.pkl")
    if os.path.exists(out_file):
        with open(out_file, "rb") as f:
            previous = pickle.load(f)
        previous.update(runs)
        runs = previous
    with open(out_file, "wb") as f:
        pickle.dump(runs, f)
    print(f"\nsaved {len(runs)} example runs -> {out_file}   (plot them with plot_examples.ipynb)")