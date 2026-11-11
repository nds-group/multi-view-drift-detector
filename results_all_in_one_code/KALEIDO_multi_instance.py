import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from KALEIDO_TRANSFORMATIONS import KALEIDO

# loading min and max windows
windows_df = pd.read_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stable_windows.csv")
)  # columns: time_series, min_window_p95, max_window
window_lookup = {
    row["time_series"]: (int(row["min_window_p95"]), int(row["max_window"]))
    for _, row in windows_df.iterrows()
}


# create a class with default parameters to handle multiple KALEIDO instances with different transformations
class KALEIDO_multi_instance:
    def __init__(
        self,
        delta=0.05,
        method="auto",
        alternative="two-sided",
        window_size_filesave=False,
        eval_noshrink_window=100,
        min_eval_window=200,
        dist_change_method="ks",
        stat_value=0.0,
        alternatives=("gradient", "season"),
        use_ttest=False,
    ):
        self.delta = delta
        self.method = method
        self.alternative = alternative
        self.window_size_filesave = window_size_filesave
        self.eval_noshrink_window = eval_noshrink_window
        self.min_eval_window = min_eval_window
        self.dist_change_method = dist_change_method
        self.stat_value = stat_value
        self.alternatives = alternatives
        self.use_ttest = use_ttest
        self.instances_and_transformations = (
            dict()
        )  # dictionary to hold KALEIDO instances and their corresponding transformations

    def instantiate(self, transformations):
        """
        Initiate different KALEIDO instances with different transformations.
        transformations: list of lists, where each inner list contains:
            - name of the transformation (str)
            - transformation function (callable)
            - (optional) parameters for the transformation function
        """
        for transformation in transformations:
            self.instances_and_transformations[transformation[0]] = (
                KALEIDO(
                    delta=self.delta,
                    method=self.method,
                    alternative=self.alternative,
                    window_size_filesave=self.window_size_filesave,
                    eval_noshrink_window=self.eval_noshrink_window,
                    min_eval_window=self.min_eval_window,
                    dist_change_method=self.dist_change_method,
                    stat_value=self.stat_value,
                    alternatives=self.alternatives,
                    use_ttest=self.use_ttest,
                ),
                transformation[1],  # transformation function
                (
                    transformation[2:] if len(transformation) > 1 else None
                ),  # parameter for transformation
            )

    def update(self, value, eval=True):
        """
        Update all KALEIDO instances with the new value and return their drift detection results.
        value: new data point to be added (float)
        eval: whether to evaluate for drift (bool)
        Returns: dictionary with transformation names as keys and drift detection results as bool or None as values
        """
        drift_results = dict()

        for transformation_name, (
            kaleido,
            transformation,
            transformation_arguments,
        ) in self.instances_and_transformations.items():
            drift_results[transformation_name] = kaleido.update(
                value, transformation, transformation_arguments, eval
            )

        return drift_results

    def get_window_sizes(self):
        """
        Get the current window sizes of all KALEIDO instances.
        Returns: dictionary with transformation names as keys and their last window sizes as values
        """
        return {
            k: v[0].get_window_size()[-1]
            for k, v in self.instances_and_transformations.items()
        }

    def get_alternatives_pvals(self):
        return {
            k: v[0].get_p_values()
            for k, v in self.instances_and_transformations.items()
        }

    def get_alternatives_stats(self):
        return {
            k: v[0].get_stat_values()
            for k, v in self.instances_and_transformations.items()
        }

    def get_alternatives_drifts(self):
        return {
            k: v[0].get_decisions()
            for k, v in self.instances_and_transformations.items()
        }

    # define transformation functions
    def transf_normal_data(self, data):
        return np.array(data)

    def trans_total_variation(self, data):
        data = np.array(data, dtype=float)  # Ensure float type for division
        trans_data = np.diff(data)
        return np.hstack([trans_data[0], trans_data])

    def trans_aggregation(self, data, block_size):
        data = np.array(data)
        reshaped_data = data[: len(data) // block_size * block_size].reshape(
            -1, block_size
        )
        trans_data = reshaped_data.mean(axis=1)
        return trans_data

    def trans_variation_aggregarion(self, data, block_size):
        tpv_result = np.array(self.trans_total_variation(data))
        return self.trans_aggregation(tpv_result, block_size)

    def trans_rolling_mean(self, data, rollong_mean_window):
        data = np.array(data)
        return np.hstack(
            [
                np.full(rollong_mean_window - 1, np.nan),
                np.convolve(
                    data,
                    np.ones(rollong_mean_window) / rollong_mean_window,
                    mode="valid",
                ),
            ]
        )

    def trans_rolling_mean_variation(self, data, rolling_mean_window):
        tpv_result = np.array(self.trans_total_variation(data))
        return self.trans_rolling_mean(tpv_result, rolling_mean_window)


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


def classify_drift_points(transformation_results, labels):

    TP, FP, TN, FN = [], [], [], np.arange(len(labels))[labels == 1].tolist()
    TP_distances = []  # to store distances of TPs (middle point) to the nearest label
    TP_distances_end = (
        []
    )  # to store distances of TPs (end of window) to the nearest label
    n = len(labels)
    labels_track_TP = labels.copy()

    for i in range(n):
        drifts = []
        for transformation_name, drift_list in transformation_results.items():
            drift = [d for d in drift_list if d[1] == i]
            if drift:
                drifts.append(drift)

        if len(drifts) == len(transformation_results.keys()):
            if len(drifts) == 0:
                pass
            else:
                # get max window size, check if it crosses label
                window_start = min([d[0][0] - d[0][3] for d in drifts if d[0][2]])
                window_end = max([d[0][0] for d in drifts if d[0][2]])
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


def classify_drift_points_infocom(transformation_results, labels, event_length):
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

    for i in range(n):
        drifts = []
        for transformation_name, drift_list in transformation_results.items():
            drift = [d for d in drift_list if d[1] == i]
            if drift:
                drifts.append(drift)

        if len(drifts) == len(transformation_results.keys()):
            if len(drifts) == 0:
                pass
            else:
                # get max window size, check if it crosses label
                window_start = min([d[0][0] - d[0][3] for d in drifts if d[0][2]])
                window_end = max([d[0][0] for d in drifts if d[0][2]])
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
    method = "asymp"  ## it can be 'auto', 'exact' or  'asymp'
    alternative = "two-sided"  ## it can be 'two-sided' or 'greater' or 'less'
    delta = 0.05  ## significance level
    dist_change_method = (
        "ks"  ## it can be 'ks' or 'ttest' or 'levene' or 'anderson' or 'anderson_ksamp'
    )
    stat_value_list = [0.1]
    timeseries_list = [
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
    ]

    config_list = ["config_all_trans_1_data_only"]

    # feature string, e.g. 'gradient+season' or 'normal_data+gradient+season+ttest_ind'
    # 'ttest_ind' switches on the extra t-test voter; every other token must be a key
    # of KALEIDO.alternatives_registry()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="gradient+season")
    args = parser.parse_args()

    features = args.features
    tokens = features.split("+")
    use_ttest = "ttest_ind" in tokens
    alternatives = [t for t in tokens if t != "ttest_ind"]
    metrics_dir = f"results_all_in_one_code/{config_list[0]}/{features}/final"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = f"{metrics_dir}/metrics_proposal_100_configs_{config_list[0]}_{features}_stable_window.csv"
    header = not os.path.exists(metrics_csv)  # write header only when the file is new

    for timeseries in tqdm(timeseries_list):
        min_eval_window, eval_no_shrink_window = window_lookup.get(
            timeseries, (200, 500)
        )
        for config in config_list:
            for stat_value in stat_value_list:

                kaleido_multi = KALEIDO_multi_instance(
                    delta=delta,
                    method=method,
                    alternative=alternative,
                    min_eval_window=min_eval_window,
                    eval_noshrink_window=eval_no_shrink_window,
                    dist_change_method=dist_change_method,
                    stat_value=stat_value,
                    alternatives=alternatives,
                    use_ttest=use_ttest,
                )
                if config == "config_all_trans_1_data_only":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data]
                    ]
                elif config == "config_all_trans_1":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data],
                        ["total_variation", kaleido_multi.trans_total_variation],
                    ]
                elif config == "config_all_trans_1_10":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data],
                        ["total_variation", kaleido_multi.trans_total_variation],
                        ["rolling_mean_10", kaleido_multi.trans_rolling_mean, 10],
                        [
                            "variation_rolling_mean_10",
                            kaleido_multi.trans_rolling_mean_variation,
                            10,
                        ],
                    ]
                elif config == "config_all_trans_1_100":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data],
                        ["total_variation", kaleido_multi.trans_total_variation],
                        ["rolling_mean_100", kaleido_multi.trans_rolling_mean, 100],
                        [
                            "variation_rolling_mean_100",
                            kaleido_multi.trans_rolling_mean_variation,
                            100,
                        ],
                    ]
                elif config == "config_all_trans_1_10_100":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data],
                        ["total_variation", kaleido_multi.trans_total_variation],
                        ["rolling_mean_10", kaleido_multi.trans_rolling_mean, 10],
                        ["rolling_mean_100", kaleido_multi.trans_rolling_mean, 100],
                        [
                            "variation_rolling_mean_10",
                            kaleido_multi.trans_rolling_mean_variation,
                            10,
                        ],
                        [
                            "variation_rolling_mean_100",
                            kaleido_multi.trans_rolling_mean_variation,
                            100,
                        ],
                    ]
                elif config == "config_all_trans_1_10_20_30":
                    list_of_transformations = [
                        ["normal_data", kaleido_multi.transf_normal_data],
                        ["total_variation", kaleido_multi.trans_total_variation],
                        ["rolling_mean_10", kaleido_multi.trans_rolling_mean, 10],
                        ["rolling_mean_20", kaleido_multi.trans_rolling_mean, 20],
                        ["rolling_mean_30", kaleido_multi.trans_rolling_mean, 30],
                        [
                            "variation_rolling_mean_10",
                            kaleido_multi.trans_rolling_mean_variation,
                            10,
                        ],
                        [
                            "variation_rolling_mean_20",
                            kaleido_multi.trans_rolling_mean_variation,
                            20,
                        ],
                        [
                            "variation_rolling_mean_30",
                            kaleido_multi.trans_rolling_mean_variation,
                            30,
                        ],
                    ]

                if timeseries == "AR4_linear_1":
                    x = load_variable_column_csv(
                        "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/time_series_linear_abrupt_100.csv"
                    )
                    data_streams = x.values
                    labels_all = load_variable_column_csv(
                        "results_all_in_one_code/datasets/synthetic_linear_nonlinear_datasets/drift_labels_linear_abrupt_100.csv"
                    )
                elif timeseries == "rolling_mean":
                    uncertainties = np.load(
                        "uncertainty_our_results/uncertainties_TS.npy"
                    )
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
                    data_stream = x
                    event_length = 7 * 24  # 168 hours (1 week)

                elif timeseries == "infocom_4G_trend":
                    data = pd.read_csv(
                        "results_all_in_one_code/datasets/INFOCOM/4G/ts_location_8102f81001516d06_trend_True.csv"
                    )
                    x = data["trend_sum_byte_dn_clean"].values
                    data_stream = x
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
                        "results_all_in_one_code/datasets/INFOCOM/ran_updates_event_4G addition_day_granularity.csv"
                    )
                    data_streams = x.values
                    labels_all = load_variable_column_csv(
                        "results_all_in_one_code/datasets/INFOCOM/drift_labels_ran_updates_event_4G addition_day_granularity.csv"
                    )
                    event_length = 7  # 7 days (1 week)

                for item, data_stream in enumerate(data_streams):
                    kaleido_multi.instantiate(transformations=list_of_transformations)

                    vote_results = []
                    data_stream = pd.Series(data_stream).dropna().values
                    tuple_drifts_history = defaultdict(
                        list
                    )  # to store (index of drift detection as middle of window, drift) tuples for each transformation
                    vote_value = (
                        min_eval_window / 2 - 1
                    )  # when we stat the voting and evaluation
                    eval = True  # initially True (False will be when we start debounce period)

                    for i, val in tqdm(
                        enumerate(data_stream, start=1)
                    ):  # TODO check this start=0
                        drifts = kaleido_multi.update(
                            val, eval=False if (i < vote_value) else True
                        )  # eval is False during the debounce period (if vote values is the future)
                        if all(
                            d is None for d in drifts.values()
                        ):  # here we skip if all instances return None (like odd window size or window too small)
                            continue

                        window_sizes = kaleido_multi.get_window_sizes()

                        p_values_alternatives = kaleido_multi.get_alternatives_pvals()
                        drifts_alternatives = kaleido_multi.get_alternatives_drifts()
                        stat_values_alternatives = (
                            kaleido_multi.get_alternatives_stats()
                        )

                        for (
                            transformation_name,
                            window_size,
                        ) in window_sizes.items():  # items gives keys and values
                            if drifts[transformation_name] is not None:
                                tuple_drifts_history[transformation_name].append(
                                    (
                                        i,
                                        int(np.ceil(i - window_size / 2)),
                                        drifts[transformation_name],
                                        window_size,
                                        True,
                                        p_values_alternatives[transformation_name],
                                        drifts_alternatives[transformation_name],
                                        stat_values_alternatives[transformation_name],
                                    )
                                )  # the last value (now 5th) is true if we evaluated, False if it was filled due to window shrink or debounce period

                                # fill the Trues after drift detected and window shrank (also fill when we have debounce period)
                                # if they are more that 1 step away from each other in tuple_drifts_history, it will fill it with the second to last known value
                                if i >= min_eval_window:
                                    for step in range(
                                        tuple_drifts_history[transformation_name][-2][1]
                                        + 1,
                                        tuple_drifts_history[transformation_name][-1][
                                            1
                                        ],
                                    ):
                                        tuple_drifts_history[
                                            transformation_name
                                        ].insert(
                                            tuple_drifts_history[
                                                transformation_name
                                            ].index(
                                                tuple_drifts_history[
                                                    transformation_name
                                                ][-1]
                                            ),
                                            (
                                                tuple_drifts_history[
                                                    transformation_name
                                                ][-2][0],
                                                step,
                                                tuple_drifts_history[
                                                    transformation_name
                                                ][-2][2],
                                                tuple_drifts_history[
                                                    transformation_name
                                                ][-2][3],
                                                False,
                                                [],
                                                [],
                                                [],
                                            ),
                                        )

                        while (
                            True
                        ):  # Do as many vote_values as there are available for all transformations
                            vote_inputs = []
                            for transformation_name in tuple_drifts_history.keys():
                                vote_result = [
                                    v1
                                    for v1 in tuple_drifts_history[transformation_name]
                                    if v1[1] == vote_value
                                ]

                                if len(vote_result) > 0:
                                    vote_inputs.append(vote_result[0][2])

                            if len(vote_inputs) == len(tuple_drifts_history.keys()):
                                vote_results.append(
                                    (vote_value, boyer_moore_majority_vote(vote_inputs))
                                )
                                vote_value += 1

                                if vote_results[-1][1] == True:
                                    vote_value += max(
                                        [
                                            t[3]
                                            for v in tuple_drifts_history.values()
                                            for t in v
                                            if t[1] == (vote_value - 1)
                                        ]
                                    )

                            else:
                                break

                    true_results = [v for v in vote_results if v[1]]
                    transf_results = defaultdict(list)
                    for true_drift, _ in true_results:
                        for transformation_name in tuple_drifts_history.keys():
                            transf_results[transformation_name].append(
                                [
                                    v
                                    for v in tuple_drifts_history[transformation_name]
                                    if v[1] == true_drift
                                ][0]
                            )

                    folder_path = f"results_all_in_one_code/{config}/{features}/{timeseries}/stable_window/final/stat_{stat_value}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    drift_indices = [v[0] for v in vote_results if v[1]]
                    values = [data_stream[int(i)] for i in drift_indices]

                    labels = np.zeros(len(data_stream), dtype=int)

                    if timeseries == "random_noisy" or timeseries == "random_easy":
                        labels_floats = np.array(
                            labels_all.iloc[item, :].dropna().values
                        )
                        labels_int = labels_floats.astype(int)
                        for i in labels_int:
                            labels[i] = 1
                    elif timeseries == "AR4_linear_1":
                        labels_floats = np.array(
                            labels_all.iloc[item, :].dropna().values
                        )
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
                        # for i in [3000, 6000, 9000]:
                        labels_floats = np.array(
                            labels_all.iloc[item, :].dropna().values
                        )
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
                        labels_floats = np.array(
                            labels_all.iloc[item, :].dropna().values
                        )
                        labels_int = labels_floats.astype(int)
                        for i in labels_int:
                            labels[i] = 1

                    elif (
                        timeseries == "infocom_5G_add_day"
                        or timeseries == "infocom_4G_add_day"
                        or timeseries == "infocom_5G_add_day_trend"
                        or timeseries == "infocom_4G_add_day_trend"
                    ):
                        labels_floats = np.array(
                            labels_all.iloc[item, :].dropna().values
                        )
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
                        (
                            TP_idx,
                            FP_idx,
                            TN_idx,
                            FN_idx,
                            TP_distances,
                            TP_distances_end,
                        ) = classify_drift_points_infocom(
                            transf_results, labels, event_length
                        )
                    else:
                        (
                            TP_idx,
                            FP_idx,
                            TN_idx,
                            FN_idx,
                            TP_distances,
                            TP_distances_end,
                        ) = classify_drift_points(transf_results, labels)

                    metrics = evaluate_drift_performance(
                        len(TP_idx),
                        len(FP_idx),
                        len(TN_idx),
                        len(FN_idx),
                        np.array(TP_distances),
                        np.array(TP_distances_end),
                    )

                    # Save the metrics dictionary as a CSV file
                    metrics_df = pd.DataFrame([metrics])

                    # Save the metrics dictionary as a CSV file
                    metrics_df.insert(0, "id", [item])
                    metrics_df.insert(1, "time_series", [timeseries])
                    metrics_df.insert(2, "config", [config])
                    metrics_df.insert(3, "stat_value", [stat_value])
                    metrics_df.insert(
                        len(metrics_df.columns), "drift_indices", [drift_indices]
                    )
                    metrics_df.to_csv(metrics_csv, mode="a", index=False, header=header)

                    for transformation_name in tuple_drifts_history.keys():
                        label_by_idx = {}
                        for i in TP_idx:
                            label_by_idx[i] = "TP"
                        for i in FP_idx:
                            label_by_idx[i] = "FP"
                        for i in TN_idx:
                            label_by_idx[i] = "TN"
                        for i in FN_idx:
                            label_by_idx[i] = "FN"

                        tuple_drifts_history[transformation_name] = [
                            t + (label_by_idx.get(t[1]),)
                            for t in tuple_drifts_history[transformation_name]
                        ]

                    if header:
                        header = False


# to run it with a particular combination python KALEIDO_multi_instance.py --features normal_data+trend+ttest_ind
# otherwise run run_all_features.py