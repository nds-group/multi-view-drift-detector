"""Run KALEIDO_multi_instance.py once per feature combination.

Each run writes its own metrics CSV and results folder (same naming as before),
so the loop is resumable: finished combinations are skipped.
"""
import itertools
import os
import subprocess
import sys

POOL = ['normal_data', 'gradient', 'second_gradient', 'third_derivative',
        'fourth_derivative', 'fifth_derivative', 'tenth_derivative',
        'trend', 'season', 'fft_transform']
MIN_K, MAX_K = 1, 3          # subset sizes to run; 1..len(POOL) = all 1023 subsets
WITH_AND_WITHOUT_TTEST = False
CONFIG = 'config_all_trans_1_data_only'

runs = []
# for k in range(MIN_K, MAX_K + 1): # I interapted when groups of size of 3 were running
for k in range(2, len(POOL) + 1, 2): # only even combinations
    for combo in itertools.combinations(POOL, k):
        base = '+'.join(combo)
        runs.append(base)
        if WITH_AND_WITHOUT_TTEST:
            runs.append(base + '+ttest_ind')

print(f"{len(runs)} runs")
for n, features in enumerate(runs, 1):
    marker = f"results_all_in_one_code/{CONFIG}/{features}/final/metrics_proposal_100_configs_{CONFIG}_{features}_stable_window.csv"
    if os.path.exists(marker):
        print(f"[{n}/{len(runs)}] skip {features}")
        continue
    # --- claim this combination so other copies skip it ---
    run_dir = f"results_all_in_one_code/{CONFIG}/{features}"
    os.makedirs(run_dir, exist_ok=True)
    lock = os.path.join(run_dir, '.running')
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print(f"[{n}/{len(runs)}] busy {features}")
        continue
    # ------------------------------------------------------
    print(f"[{n}/{len(runs)}] run  {features}", flush=True)
    try:
        subprocess.run([sys.executable, 'results_all_in_one_code/KALEIDO_multi_instance.py',
                    '--features', features], check=True)
    finally:
        os.remove(lock)