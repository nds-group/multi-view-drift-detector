import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from numba import njit
import sys

# Ensure the parent directory is in the path to import custom modules
sys.path.append(os.getcwd())
from ADWIN_CLASS import ADWIN
from river import drift
import math
import warnings
from itertools import product
from scipy import stats as st
from scipy.stats import wasserstein_distance

# loading min and max windows
windows_df = pd.read_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stable_windows.csv")
)  # columns: time_series, min_window_p95, max_window
window_lookup = {
    row["time_series"]: (int(row["min_window_p95"]), int(row["max_window"]))
    for _, row in windows_df.iterrows()
}
METRICS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_results", "metrics"
)


class CUSUM:
    """Two-sided CUSUM on continuous values. Matches river-style interface."""

    def __init__(self, threshold=5.0, drift_threshold=50.0, warmup=30):
        self.threshold = threshold  # slack (k), in units of running std
        self.drift_threshold = drift_threshold
        self.warmup = warmup
        self._reset()

    def _reset(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # for Welford running variance
        self.g_pos = 0.0
        self.g_neg = 0.0
        self.drift_detected = False

    def update(self, x):
        self.drift_detected = False
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if self.n < self.warmup:
            return self
        std = (self.m2 / (self.n - 1)) ** 0.5
        if std == 0:
            return self
        s = (x - self.mean) / std
        self.g_pos = max(0.0, self.g_pos + s - self.threshold)
        self.g_neg = max(0.0, self.g_neg - s - self.threshold)
        if self.g_pos > self.drift_threshold or self.g_neg > self.drift_threshold:
            self.drift_detected = True
            self.g_pos = 0.0
            self.g_neg = 0.0
        return self


class EWMA:
    """EWMA control chart for mean shifts on continuous values."""

    def __init__(self, lambda_=0.2, L=3.0, warmup=30):
        self.lambda_ = lambda_  # smoothing
        self.L = L  # control-limit width (sigmas)
        self.warmup = warmup
        self._reset()

    def _reset(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.z = None
        self.drift_detected = False

    def update(self, x):
        self.drift_detected = False
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if self.z is None:
            self.z = self.mean
        if self.n < self.warmup:
            self.z = (1 - self.lambda_) * self.z + self.lambda_ * x
            return self
        std = (self.m2 / (self.n - 1)) ** 0.5
        if std == 0:
            return self
        self.z = (1 - self.lambda_) * self.z + self.lambda_ * x
        sigma_z = std * (self.lambda_ / (2 - self.lambda_)) ** 0.5
        if abs(self.z - self.mean) > self.L * sigma_z:
            self.drift_detected = True
            self.z = self.mean  # reset chart after signal
        return self


class WinDriftPP:
    """Streaming adapter for WinDrift++ (Naqvi, Rehman & Islam, IEEE TAI 2025,
    https://github.com/Naureenaqvi/windriftplusplus). Matches river-style interface.

    The original is a batch detector over a table of `N_OBS x months`; here the stream
    is cut into consecutive blocks of `block_size` points (one block == one "month").
    Every time a block completes, a historical window WH and a new window WN are built
    from whole blocks and compared with the 5 ECDF-based tests of the paper
    (KS, Kuiper, CVM, AD, EMD). Drift is flagged when a majority (>=3/5) of the tests flag.

    mode='consecutive'   (paper's Mode I): WH = the `w` blocks before WN, WN = last `w` blocks.
                         Detects drift onset; no seasonality assumption. Use this by default.
    mode='corresponding' (paper's Mode II): WH = the same `w` blocks one cycle earlier
                         (cycle_len blocks ago), WN = last `w` blocks. Only meaningful for
                         periodic data (e.g. hourly traffic with cycle_len*block_size = 1 week).

    `win_levels` are the paper's window levels (default [1,3,6,12] there); a comparison at
    level w is done every w blocks. `width` is the number of stream points spanned by the
    last comparison (len(WH)+len(WN)), so `i - width/2` is the WH/WN boundary.
    """

    def __init__(
        self,
        block_size=100,
        win_levels=(1,),
        alpha=0.05,
        mode="consecutive",
        cycle_len=None,
        n_permutations=200,
        seed=42,
    ):
        self.block_size = int(block_size)
        self.win_levels = sorted(int(w) for w in win_levels)
        self.alpha = alpha
        self.mode = mode
        self.cycle_len = cycle_len  # in blocks, only for mode='corresponding'
        self.n_permutations = n_permutations
        self.seed = seed
        if mode == "corresponding" and not cycle_len:
            raise ValueError(
                "cycle_len (in blocks) is required for mode='corresponding'"
            )
        self._lcm = 1
        for w in self.win_levels:
            self._lcm = self._lcm * w // math.gcd(self._lcm, w)
        self._reset()

    def _reset(self):
        self.blocks = []  # completed blocks (list of np.ndarray)
        self._buf = []
        self.n = 0
        self.drift_detected = False
        self.width = 2 * self.block_size * self.win_levels[0]
        self.last_votes = None  # (votes_drift, [ks, kuiper, cvm, ad, emd])
        self.last_level = None
        self._rng = np.random.default_rng(self.seed)

    # ---- ECDF helpers (max-ECDF convention for ties, evaluated on the merged support) ----
    @staticmethod
    def _ecdf_pair(x_h, x_n):
        merged = np.sort(np.concatenate([x_h, x_n]))
        F_h = np.searchsorted(np.sort(x_h), merged, side="right") / len(x_h)
        F_n = np.searchsorted(np.sort(x_n), merged, side="right") / len(x_n)
        return merged, F_h, F_n

    def _run_tests(self, x_h, x_n):
        """Returns list of 5 drift flags [KS, Kuiper, CVM, AD, EMD] (1=drift)."""
        n, m = len(x_h), len(x_n)
        merged, F_h, F_n = self._ecdf_pair(x_h, x_n)
        diff = F_h - F_n
        abs_diff = np.abs(diff)
        flags = []
        # (3) KS: max|F_H - F_N| vs D_crit = c(alpha) * sqrt(1/n + 1/m)   (c(0.05)=1.36)
        c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2.0))
        flags.append(int(abs_diff.max() >= c_alpha * np.sqrt(1.0 / n + 1.0 / m)))
        # (4) Kuiper: D+ + D- vs V_crit (Press et al. approx., alpha=0.05)
        lam = np.sqrt(n * m / (n + m))
        v_crit = (1.747 + 0.11 / lam + 0.10 / lam**2) / lam
        flags.append(int(diff.max() + (-diff).max() >= v_crit))
        # (5) CVM: decision from scipy two-sample p-value
        flags.append(int(st.cramervonmises_2samp(x_h, x_n).pvalue < self.alpha))
        # (6) AD: decision from scipy k-sample p-value (capped at [0.001, 0.25] by scipy)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # scipy warns when p is capped to [0.001, 0.25]
                p_ad = st.anderson_ksamp([x_h, x_n]).pvalue
            flags.append(int(p_ad < self.alpha))
        except Exception:
            flags.append(0)
        # (7) EMD: Wasserstein-1 vs permutation critical value
        emd = wasserstein_distance(x_h, x_n)
        combined = np.concatenate([x_h, x_n])
        null = np.empty(self.n_permutations)
        for k in range(self.n_permutations):
            perm = self._rng.permutation(combined)
            null[k] = wasserstein_distance(perm[:n], perm[n:])
        flags.append(int(emd >= np.quantile(null, 1 - self.alpha)))
        return flags

    def _windows(self, w):
        """Block index ranges (hist, new) for window level w, or None if not available yet."""
        b = len(self.blocks)
        if b % w != 0:
            return None
        new = (b - w, b)
        if self.mode == "consecutive":
            hist = (b - 2 * w, b - w)
        else:  # corresponding
            hist = (b - w - self.cycle_len, b - self.cycle_len)
        if hist[0] < 0:
            return None
        return hist, new

    def update(self, x):
        self.drift_detected = False
        self.n += 1
        self._buf.append(float(x))
        if len(self._buf) < self.block_size:
            return self
        self.blocks.append(np.asarray(self._buf))
        self._buf = []
        for w in self.win_levels:
            rng_ = self._windows(w)
            if rng_ is None:
                continue
            (h0, h1), (n0, n1) = rng_
            x_h = np.concatenate(self.blocks[h0:h1])
            x_n = np.concatenate(self.blocks[n0:n1])
            flags = self._run_tests(x_h, x_n)
            votes = sum(flags)
            if votes > len(flags) - votes:  # majority vote -> drift
                self.drift_detected = True
                self.width = len(x_h) + len(x_n)
                self.last_votes = (votes, flags)
                self.last_level = w
                break
        # bound memory: keep only the blocks any level could still need
        keep = (
            2 * self.win_levels[-1]
            if self.mode == "consecutive"
            else self.cycle_len + self.win_levels[-1]
        )
        if len(self.blocks) > keep + self._lcm:
            # drop a multiple of lcm(win_levels) so `len(blocks) % w` stays unchanged for every level
            drop = ((len(self.blocks) - keep) // self._lcm) * self._lcm
            self.blocks = self.blocks[drop:]
        return self


# ADWIN
detectors_to_run = ("ADWIN", "KSWIN", "PH", "CUSUM", "EWMA", "WINDRIFT")
method = "asymp"
alternative = "two-sided"
clock = 1
delta = 0.05

# WinDrift++ (streaming adapter, see class WinDriftPP above)
windrift_mode = "consecutive"  # 'consecutive' (Mode I) or 'corresponding' (Mode II, needs windrift_cycle_len)
windrift_win_levels = (
    12,
)  # paper uses [1, 3, 6, 12]; level w compares w blocks vs w blocks every w blocks
windrift_cycle_len = None  # cycle length in BLOCKS for Mode II, e.g. 7*24 // block_size for weekly hourly data


# Boyer-Moore Majority Vote Algorithm
@njit
def boyer_moore_majority_vote(nums):
    candidate = None
    count = 0

    # Phase 1: Find a candidate
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1

    # Phase 2: Verify the candidate
    count = 0
    for num in nums:
        if num == candidate:
            count += 1

    return candidate if count > len(nums) // 2 else None


def generate_ar_series(alpha, sigma_sq, length=3000):
    """Generates an AR(p) time series.

    Args:
        alpha (list): The AR parameters [α1, α2, ..., αp].
        sigma_sq (float): The variance (σ²) of the white noise.
        length (int): The length of the time series.

    Returns:
        pd.Series: The generated AR time series.
    """

    p = len(alpha)  # Order of the AR model
    series = np.zeros(length)
    noise = np.random.normal(0, sigma_sq, length)  # Generate white noise

    # Generate initial values (first p values)
    for t in range(p):
        series[t] = np.random.normal(0, 1)  # Or any other initialization

    # Generate the rest of the time series
    for t in range(p, length):
        ar_sum = 0
        for i in range(p):
            ar_sum += alpha[i] * series[t - i - 1]
        series[t] = ar_sum + noise[t]

    return pd.Series(series)


def generate_ar_non_linear_series(alpha, sigma_sq, length=3000):
    """
    Generates a nonlinear AR(p) time series.

    Args:
        alpha (list): The AR parameters [α1, α2, ..., αp].
        sigma_sq (float): The variance (σ²) of the white noise.
        length (int): The length of the time series.

    Returns:
        pd.Series: The generated nonlinear AR time series.
    """

    p = len(alpha)  # Order of the AR model
    series = np.zeros(length)

    # Generate Gaussian noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, sigma_sq, length)  # Generate white noise

    # Initialize the time series
    x = np.zeros(length)

    # Compute the time series
    for t in range(2, length):
        x[t] = (
            noise[t]
            - alpha[0] * noise[t - 1]
            + alpha[1] * noise[t - 2]
            + alpha[2] * noise[t - 1] * noise[t - 2]
            - alpha[3] * noise[t - 2] ** 2
        )

    return pd.Series(x)


def classify_drift_points(
    drift_indices_middle, drift_indices_end, window_sizes, labels
):

    TP, FP, TN, FN = [], [], [], np.arange(len(labels))[labels == 1].tolist()
    TP_distances = []  # to store distances of TPs (middle point) to the nearest label
    TP_distances_end = (
        []
    )  # to store distances of TPs (end of window) to the nearest label
    n = len(labels)
    labels_track_TP = labels.copy()

    vector = np.zeros(len(data_stream), dtype=int)
    for i in drift_indices_middle:
        vector[i] = 1

    for i in range(n):
        if vector[i] == 1:
            # get max window size, check if it crosses label
            mid = np.asarray(drift_indices_middle)
            mid_pos = np.where(mid == i)[0][0]
            end = drift_indices_end[mid_pos]
            win = window_sizes[end]
            window_start = end - win
            window_end = end
            if (
                np.isin(labels[window_start:window_end], [1]).any()
                and np.isin(labels_track_TP[window_start:window_end], [1]).any()
            ):
                TP.append(i)
                inx_temp_array = (
                    window_start + np.where(labels[window_start:window_end] == 1)[0]
                )  # the window might cross more than 1 label
                if len(inx_temp_array) > 1:
                    idx_temp = min(
                        window_start
                        + np.where(labels[window_start:window_end] == 1)[0],
                        key=lambda x: abs(i - x),
                    )
                else:
                    idx_temp = window_start + np.argmax(
                        np.isin(labels[window_start:window_end], [1])
                    )
                if idx_temp in FN:  # in case we have more TPs for the same label
                    FN.remove(idx_temp)
                TP_distances.append(i - idx_temp)
                TP_distances_end.append(window_end - idx_temp)
            else:
                FP.append(i)

    TN = [idx for idx in range(n) if idx not in TP and idx not in FP and idx not in FN]

    return TP, FP, TN, FN, TP_distances, TP_distances_end


def classify_drift_points_infocom(
    drift_indices_middle, drift_indices_end, window_sizes, labels, event_length
):
    ranged_labels = labels.copy()
    ones = np.where(ranged_labels == 1)[0]

    for idx in ones:
        s = max(0, idx - event_length + 1)
        e = idx + 1
        ranged_labels[s:e] = 1

    TP, FP, TN, FN = (
        [],
        [],
        [],
        np.arange(len(ranged_labels))[ranged_labels == 1].tolist(),
    )
    TP_distances = []  # to store distances of TPs (middle point) to the nearest label
    TP_distances_end = (
        []
    )  # to store distances of TPs (end of window) to the nearest label
    n = len(ranged_labels)
    labels_track_TP = ranged_labels.copy()

    vector = np.zeros(len(data_stream), dtype=int)
    for i in drift_indices_middle:
        vector[i] = 1

    for i in range(n):
        if vector[i] == 1:
            # get max window size, check if it crosses label
            mid = np.asarray(drift_indices_middle)
            mid_pos = np.where(mid == i)[0][0]
            end = drift_indices_end[mid_pos]
            win = window_sizes[end]
            window_start = end - win
            window_end = end
            # get max window size, check if it crosses label
            if (
                np.isin(ranged_labels[window_start:window_end], [1]).any()
                and np.isin(labels_track_TP[window_start:window_end], [1]).any()
            ):
                TP.append(i)
                inx_temp_array = (
                    window_start
                    + np.where(ranged_labels[window_start:window_end] == 1)[0]
                )  # the window might cross more than 1 label
                if len(inx_temp_array) > 1:
                    idx_temp = min(
                        window_start
                        + np.where(ranged_labels[window_start:window_end] == 1)[0],
                        key=lambda x: abs(i - x),
                    )
                else:
                    idx_temp = window_start + np.argmax(
                        np.isin(ranged_labels[window_start:window_end], [1])
                    )
                if idx_temp in FN:  # in case we have more TPs for the same label
                    FN.remove(idx_temp)
                    labels_track_TP[np.where(ranged_labels == 1)[0]] = (
                        0  # once we  found true drift, remove all labels
                    )
                TP_distances.append(i - idx_temp)
                TP_distances_end.append(window_end - idx_temp)
            else:
                FP.append(i)
    if TP:
        # delete all FN
        FN = []
    else:
        # keep only the last FN out of event_length range
        FN = np.arange(len(labels))[labels == 1].tolist()

    TN = [idx for idx in range(n) if idx not in TP and idx not in FP and idx not in FN]

    return TP, FP, TN, FN, TP_distances, TP_distances_end


def evaluate_drift_performance(TP, FP, TN, FN, TP_distances, TP_distances_end):

    # Compute metrics
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    miss_detection_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
    sum_late = TP_distances[TP_distances > 0].sum()
    sum_early = TP_distances[TP_distances < 0].sum()
    sum_both = abs(sum_late) + abs(sum_early)
    # delay
    sum_late_end = TP_distances_end[TP_distances_end > 0].sum()
    sum_early_end = TP_distances_end[TP_distances_end < 0].sum()
    sum_both_end = abs(sum_late_end) + abs(sum_early_end)

    return {
        "True Positives": TP,
        "False Positives": FP,
        "True Negatives": TN,
        "False Negatives": FN,
        "False Positive Rate": false_positive_rate,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
        "Miss Detection Rate": miss_detection_rate,
        "Sum Late Detections": sum_late,
        "Sum Early Detections": sum_early,
        "Sum Both Early and Late Detections": sum_both,
        "Sum Late Detections end": sum_late_end,
        "Sum Early Detections end": sum_early_end,
        "Sum Both Early and Late Detections end": sum_both_end,
    }


def load_variable_column_csv(file_path, delimiter=","):
    # The max column count a line in the file could have
    largest_column_count = 0

    # Loop the data lines
    with open(file_path, "r") as temp_f:
        # Read the lines
        lines = temp_f.readlines()

        for l in lines:
            # Count the column count for the current line
            column_count = len(l.split(delimiter)) + 1

            # Set the new most column count
            largest_column_count = (
                column_count
                if largest_column_count < column_count
                else largest_column_count
            )

    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]

    # Read csv
    return pd.read_csv(file_path, header=None, delimiter=delimiter, names=column_names)


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    timeseries_to_run = (
                "random_easy",
                "random_noisy",
                "rolling_mean",
                "friedman_5_drifts",
                "friedman_2_drifts",
                "friedman_no_drifts",
                "infocom_5G_add_hour",
                "infocom_4G_add_hour",
                "infocom_5G_add_hour_trend",
                "infocom_4G_add_hour_trend",
                "AR4_linear_1",
                "AR4_nonlinear_1",

    )

    for cut_detector, timeseries in product(detectors_to_run, timeseries_to_run):
        # per-time-series windows, with a fallback if not in the file
        min_eval_window, eval_no_shrink_window = window_lookup.get(
            timeseries, (200, 500)
        )

        if timeseries == "AR4_linear_1":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/time_series_linear_abrupt_100.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/drift_labels_linear_abrupt_100.csv"
            )
        elif timeseries == "rolling_mean":
            uncertainties = np.load("uncertainty_our_results/uncertainties_TS.npy")
            data_streams = [
                uncertainties,
                pd.Series(uncertainties).rolling(window=10).mean(),
                pd.Series(uncertainties).rolling(window=20).mean(),
                pd.Series(uncertainties).rolling(window=30).mean(),
                pd.Series(uncertainties).rolling(window=40).mean(),
                pd.Series(uncertainties).rolling(window=50).mean(),
                pd.Series(uncertainties).rolling(window=60).mean(),
                pd.Series(uncertainties).rolling(window=70).mean(),
                pd.Series(uncertainties).rolling(window=80).mean(),
                pd.Series(uncertainties).rolling(window=90).mean(),
                pd.Series(uncertainties).rolling(window=100).mean(),
            ]
            x = data_streams
        elif timeseries == "friedman":
            data = pd.read_csv(
                "uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv"
            )
            data_streams = [data["y_1"]]
            x = data_streams
        elif timeseries == "friedman_5_drifts":
            data = pd.read_csv(
                "uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv"
            )
            data_streams = [
                data["x1"],
                data["x3"],
                data["x5"],
                data["y_1"],
                data["y_org"],
            ]
            x = data_streams
        elif timeseries == "friedman_2_drifts":
            data = pd.read_csv(
                "uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv"
            )
            data_streams = [data["x7"], data["x9"], data["x10"]]
            x = data_streams
        elif timeseries == "friedman_no_drifts":
            data = pd.read_csv(
                "uncertainty-drift-detection/01_Artificial_DataSets/friedman_with_2_abrupt_drift.csv"
            )
            data_streams = [data["x2"], data["x4"], data["x6"], data["x8"]]
            x = data_streams
        elif timeseries == "random_noisy":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_random_datasets/time_series_random_noisy_100.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_random_datasets/drift_labels_random_noisy_100.csv"
            )
        elif timeseries == "random_easy":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_random_datasets/time_series_random_simple_100.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_random_datasets/drift_labels_random_simple_100.csv"
            )
        elif timeseries == "AR4_nonlinear_1":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/time_series_nonlinear_abrupt_100.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/drift_labels_nonlinear_abrupt_100.csv"
            )

        elif timeseries == "infocom_4G":
            data = pd.read_csv(
                "results_all_in_one_code/datasets/INFOCOM/4G/ts_location_8102f81001516d06_trend_False.csv"
            )
            x = data["sum_byte_dn_clean"].values
            data_streams = [x]
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_4G_trend":
            data = pd.read_csv(
                "results_all_in_one_code/datasets/INFOCOM/4G/ts_location_8102f81001516d06_trend_True.csv"
            )
            x = data["trend_sum_byte_dn_clean"].values
            data_streams = [x]
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_5G_add_hour":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_5G addition_hour_granularity.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_5G addition_hour_granularity.csv"
            )
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_4G_add_hour":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_4G addition_hour_granularity.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_4G addition_hour_granularity.csv"
            )
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_5G_add_day":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_5G addition_day_granularity.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_5G addition_day_granularity.csv"
            )
            event_length = 7  # 7 days (1 week)

        elif timeseries == "infocom_4G_add_day":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_4G addition_day_granularity.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_4G addition_day_granularity.csv"
            )
            event_length = 7  # 7 days (1 week)

        elif timeseries == "infocom_5G_add_hour_trend":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_5G addition_hour_granularity_trend.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_5G addition_hour_granularity.csv"
            )
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_4G_add_hour_trend":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_4G addition_hour_granularity_trend.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_4G addition_hour_granularity.csv"
            )
            event_length = 7 * 24  # 168 hours (1 week)

        elif timeseries == "infocom_5G_add_day_trend":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_5G addition_day_granularity_trend.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_5G addition_day_granularity.csv"
            )
            event_length = 7  # 7 days (1 week)
        elif timeseries == "infocom_4G_add_day_trend":
            x = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_4G addition_day_granularity_trend.csv"
            )
            data_streams = x.values
            labels_all = load_variable_column_csv(
                "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_4G addition_day_granularity.csv"
            )
            event_length = 7  # 7 days (1 week)

        for item, data_stream in enumerate(data_streams):

            # rebuild detectors with this series' windows
            window_size_filesave = False
            if cut_detector == "ADWIN":
                detector = ADWIN(
                    delta=delta,
                    cut_detector=cut_detector,
                    method=method,
                    min_window_length=int(min_eval_window / 2),
                    grace_period=min_eval_window,
                    alternative=alternative,
                    window_size_filesave=window_size_filesave,
                    clock=clock,
                )
            elif cut_detector == "KSWIN":
                detector = drift.KSWIN(
                    alpha=delta,
                    seed=42,
                    window_size=min_eval_window,
                    stat_size=int(min_eval_window / 2),
                )
            elif cut_detector == "PH":
                detector = drift.PageHinkley(
                    min_instances=min_eval_window,
                    delta=delta,
                    threshold=50.0,
                    alpha=0.9999,
                    mode="both",
                )

            elif cut_detector == "CUSUM":
                detector = CUSUM(
                    threshold=0.5, drift_threshold=50.0, warmup=min_eval_window
                )
            elif cut_detector == "EWMA":
                detector = EWMA(lambda_=0.2, L=3.0, warmup=min_eval_window)
            elif cut_detector == "WINDRIFT":
                # block = half of the stable window, so level-1 compares min_eval_window/2 vs min_eval_window/2
                # (same split as KSWIN: window_size=min_eval_window, stat_size=min_eval_window/2)
                detector = WinDriftPP(
                    block_size=int(min_eval_window / 2),
                    win_levels=windrift_win_levels,
                    alpha=delta,
                    mode=windrift_mode,
                    cycle_len=windrift_cycle_len,
                    seed=42,
                )

            data_stream = pd.Series(data_stream).dropna().values

            (
                drift_indices_middle,
                drift_indices_end,
                drift_detected_middle,
                drift_detected_end,
                window_sizes,
            ) = ([], [], [], [], [])

            for i, val in tqdm(enumerate(data_stream)):
                detector.update(val)
                # record window size at this iteration
                if cut_detector == "ADWIN":
                    window_sizes.append(int((detector.width)))  # v3
                elif cut_detector in ("KSWIN", "PH", "CUSUM", "EWMA"):
                    window_sizes.append(min_eval_window)
                elif cut_detector == "WINDRIFT":
                    window_sizes.append(
                        int(detector.width)
                    )  # len(WH)+len(WN); middle = WH/WN boundary
                if detector.drift_detected:
                    drift_indices_end.append(i)
                    drift_indices_middle.append(
                        i - int(window_sizes[-1] / 2)
                    )  # save the middle index (not the last point)
                    drift_detected_end.append(val)
                    drift_detected_middle.append(
                        data_stream[i - int(window_sizes[-1] / 2)]
                    )  # save the middle index (not the last point)
            detector._reset()

            if cut_detector == "WINDRIFT":
                folder_path = f"results_all_in_one_code/benchmark_results/stable_window/{cut_detector}/level_{windrift_win_levels[0]}/{timeseries}"  # only for windrift
            else:
                folder_path = f"results_all_in_one_code/benchmark_results/stable_window/{cut_detector}/{timeseries}"

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            labels = np.zeros(len(data_stream), dtype=int)

            if timeseries == "random_noisy" or timeseries == "random_easy":
                labels_floats = np.array(labels_all.iloc[item, :].dropna().values)
                labels_int = labels_floats.astype(int)
                for i in labels_int:
                    labels[i] = 1
            elif timeseries == "AR4_linear_1":
                labels_floats = np.array(labels_all.iloc[item, :].dropna().values)
                labels_int = labels_floats.astype(int)
                for i in labels_int:
                    labels[i] = 1
            elif timeseries == "rolling_mean":
                for i in [2000, 4500, 13000]:
                    labels[i] = 1
            elif timeseries == "friedman" or timeseries == "friedman_5_drifts":
                for i in [2000, 3000, 5000, 7500, 16000]:
                    labels[i] = 1
            elif timeseries == "friedman_2_drifts":
                for i in [10000, 12000]:
                    labels[i] = 1
            elif timeseries == "friedman_no_drifts":
                for i in []:
                    labels[i] = 1
            elif timeseries == "AR4_nonlinear_1":
                labels_floats = np.array(labels_all.iloc[item, :].dropna().values)
                labels_int = labels_floats.astype(int)
                for i in labels_int:
                    labels[i] = 1
            elif timeseries == "infocom_4G":
                for i in np.loadtxt(
                    "results_all_in_one_code/datasets/INFOCOM/4G/event_index_location_8102f81001516d06_trend_False.txt",
                    delimiter=",",
                    dtype=int,
                    ndmin=1,
                ):
                    labels[i] = 1
            elif timeseries == "infocom_4G_trend":
                for i in np.loadtxt(
                    "results_all_in_one_code/datasets/INFOCOM/4G/event_index_location_8102f81001516d06_trend_True.txt",
                    delimiter=",",
                    dtype=int,
                    ndmin=1,
                ):
                    labels[i] = 1
            elif (
                timeseries == "infocom_5G_add_hour"
                or timeseries == "infocom_4G_add_hour"
                or timeseries == "infocom_5G_add_hour_trend"
                or timeseries == "infocom_4G_add_hour_trend"
            ):
                labels_floats = np.array(labels_all.iloc[item, :].dropna().values)
                labels_int = labels_floats.astype(int)
                for i in labels_int:
                    labels[i] = 1

            elif (
                timeseries == "infocom_5G_add_day"
                or timeseries == "infocom_4G_add_day"
                or timeseries == "infocom_5G_add_day_trend"
                or timeseries == "infocom_4G_add_day_trend"
            ):
                labels_floats = np.array(labels_all.iloc[item, :].dropna().values)
                labels_int = labels_floats.astype(int)
                for i in labels_int:
                    labels[i] = 1

            if timeseries in [
                "infocom_4G",
                "infocom_4G_trend",
                "infocom_5G_add_day",
                "infocom_4G_add_day",
                "infocom_5G_add_day_trend",
                "infocom_4G_add_day_trend",
                "infocom_5G_add_hour",
                "infocom_4G_add_hour",
                "infocom_5G_add_hour_trend",
                "infocom_4G_add_hour_trend",
            ]:
                TP_idx, FP_idx, TN_idx, FN_idx, TP_distances, TP_distances_end = (
                    classify_drift_points_infocom(
                        drift_indices_middle,
                        drift_indices_end,
                        window_sizes,
                        labels,
                        event_length,
                    )
                )
            else:
                TP_idx, FP_idx, TN_idx, FN_idx, TP_distances, TP_distances_end = (
                    classify_drift_points(
                        drift_indices_middle, drift_indices_end, window_sizes, labels
                    )
                )

            metrics = evaluate_drift_performance(
                len(TP_idx),
                len(FP_idx),
                len(TN_idx),
                len(FN_idx),
                np.array(TP_distances),
                np.array(TP_distances_end),
            )

            metrics_df = pd.DataFrame([metrics])
            metrics_df.insert(0, "id", [item])
            metrics_df.insert(1, "time_series", [timeseries])
            metrics_df.insert(
                len(metrics_df.columns), "drift_indices", [drift_indices_middle]
            )
            if cut_detector == "WINDRIFT":
                output_file = (
                    os.path.join(
                        METRICS_DIR,
                        f"metrics_proposal_100_benchmarks_{cut_detector}_"
                        f"stable_window_level_{windrift_win_levels[0]}.csv",
                    )
                )
            else:
                output_file = (
                    os.path.join(
                        METRICS_DIR,
                        f"metrics_proposal_100_benchmarks_{cut_detector}_stable_window.csv",
                    )
                )
            write_header = (
                not os.path.exists(output_file) or os.path.getsize(output_file) == 0
            )
            metrics_df.to_csv(
                output_file,
                mode="a",
                index=False,
                header=write_header,
            )

            category_colors = {
                "TP": "green",
                "FP": "blue",
                "FN": "darkviolet",
                "TN": "salmon",
            }
            plt.figure(figsize=(12, 5))
            plt.plot(
                data_stream,
                label=f"{cut_detector}, delta={delta}, min_window={min_eval_window}",
                alpha=0.3,
            )

            plt.scatter(
                FN_idx,
                [data_stream[i] for i in FN_idx],
                s=100,
                color=category_colors["FN"],
                label="FN",
                marker="^",
            )
            plt.scatter(
                TP_idx,
                [data_stream[i] for i in TP_idx],
                s=100,
                color=category_colors["TP"],
                label="TP",
                marker="o",
            )
            plt.scatter(
                FP_idx,
                [data_stream[i] for i in FP_idx],
                s=100,
                color=category_colors["FP"],
                label="FP",
                marker="x",
            )

            for drift_end in drift_indices_end:
                xmin = drift_end - window_sizes[drift_end]
                xmax = drift_end
                drift_idx = drift_end - int(window_sizes[drift_end] / 2)

                # Determine which category this drift belongs to
                if drift_idx in TP_idx:
                    color = category_colors["TP"]
                    plt.plot(
                        [xmin, xmax],
                        [data_stream[drift_idx], data_stream[drift_idx]],
                        "-o",
                        color=color,
                    )

            for x in np.where(labels == 1)[0]:
                plt.axvline(x=x, color="black", linestyle="--")
                if timeseries in [
                    "infocom_4G",
                    "infocom_4G_trend",
                    "infocom_5G_add_day",
                    "infocom_4G_add_day",
                    "infocom_5G_add_day_trend",
                    "infocom_4G_add_day_trend",
                    "infocom_5G_add_hour",
                    "infocom_4G_add_hour",
                    "infocom_5G_add_hour_trend",
                    "infocom_4G_add_hour_trend",
                ]:
                    plt.axvspan(
                        x, x - event_length, alpha=0.2, color="red", label="Event Week"
                    )

            for i, (x, y) in enumerate(
                zip(drift_indices_middle, drift_detected_middle)
            ):
                plt.text(
                    x,
                    y,
                    str(int(x)),
                    fontsize=12,
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

            plt.title(f"Drift Detected {len(drift_detected_middle)} times")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(
                f"{folder_path}/results_metrics_max_w{eval_no_shrink_window}_min_w{min_eval_window}_id_{item}.png"
            )
            plt.close("all")

            plt.figure(figsize=(10, 4))
            plt.plot(window_sizes, label="Window size")

            plt.scatter(
                drift_indices_end,
                [window_sizes[i] for i in drift_indices_end],
                color="red",
                s=30,
                label="Drift detected",
            )

            plt.xlabel("Time / iteration")
            plt.ylabel("ADWIN window size")
            plt.title("ADWIN window size with detected drifts")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.close("all")
