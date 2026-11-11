import numpy as np
from scipy.stats import ks_2samp, ttest_ind, levene
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft

class KALEIDO:
    def __init__(self, delta=0.05, method='auto',alternative='two-sided',window_size_filesave = False,
                 eval_noshrink_window = 100, min_eval_window = 10, dist_change_method = 'ks', stat_value=0.0,
                 alternatives=('gradient', 'season'), use_ttest=False):
        """
        Initialize the KALEIDO drift detector.
        
        :param delta: The significance level for the KS test (default: 0.05), Inverse of confidence level (95% sure that the distributions are equal).
        :param method: The method used to compute the KS test (default: 'auto').
        :param alternative: The alternative hypothesis for the KS test (default: 'two-sided').
        :param window_size_filesave: The file to save the window size (default: False).
        :param eval_noshrink_window: The minimum window size to evaluate without shrinking (default: 100).
        :param min_eval_window: The minimum window size to evaluate (default: 10).
        :param dist_change_method: The method to detect distribution changes for no drift (default: 'ks').


        """
        self.delta = delta  # Significance level (threshold for drift detection)
        self.window_I = [] # Input window
        self.original_data = [] # to save the original data
        self.method = method # Method for the KS test
        self.alternative = alternative # Alternative hypothesis for the KS test
        self.window_size_filesave = window_size_filesave # File to save the window size
        self.eval_noshrink_window = eval_noshrink_window # Minimum window size to evaluate without shrinking
        self.min_eval_window = min_eval_window # Minimum window size to evaluate
        self.dist_change_method = dist_change_method # Method to detect distribution changes for no drift
        self.sm_list = [] # List to save the starting index of the window search range
        self.sM_list = [] # List to save the ending index of the window search range
        self.window_size_list = [] # List to save the window size
        self.stat_value = stat_value # Minimum statistic value to consider drift
        self.alternatives_pvals = []
        self.alternatives_results = []
        self.alternatives_stats = []
        self.use_ttest = use_ttest
        self.alternative_names = list(alternatives)
        registry = self.alternatives_registry()
        unknown = [n for n in self.alternative_names if n not in registry]
        if unknown:
            raise ValueError(f"Unknown alternatives {unknown}. Known: {sorted(registry)}")
        self.list_of_alternatives = [registry[n] for n in self.alternative_names]

    def alternatives_registry(self):
        """Map feature names (as used in the driver's feature string) to transformation callables."""
        return {
            'normal_data':       self.use_normal_data,
            'gradient':          np.gradient,
            'second_gradient':   lambda x: np.gradient(np.gradient(x)),
            'third_derivative':  lambda x: self.extract_nth_derivative(x, n=3),
            'fourth_derivative': lambda x: self.extract_nth_derivative(x, n=4),
            'fifth_derivative':  lambda x: self.extract_nth_derivative(x, n=5),
            'tenth_derivative':  lambda x: self.extract_nth_derivative(x, n=10),
            'trend':             self.extract_trend,
            'season':            self.extract_season,
            'fft_transform':     self.fft_transform,
        }

    def update(self, value, transformation=None, transformation_arguments=None, eval=True):
        """
        Update the sliding window with a new data point and check for drift.
        
        :param value: New data point from the stream.
        :return: True if drift is detected, False otherwise.
        """
        self.original_data.append(value)

        self.window_I = list(self.window_I)
        if eval or len(self.window_I) < self.min_eval_window: 
            self.window_I.append(value)
        else: # during debounce period, do not increase window size, just slide it and keep size constant
            self.window_I.append(value)
            self.window_I = self.window_I[1:]  # remove the oldest value


        if self.window_size_filesave:   
            with open(self.window_size_filesave, 'a') as f:
                f.write(str(len(self.window_I)) + '\n')

        self.window_size_list.append(len(self.window_I))

        if len(self.window_I) % 2 != 0 or not eval: # do not evaluate if window is not even or eval is False (debounce period)
            return
        elif len(self.window_I) >= self.min_eval_window: # evaluate only if the window size is even and greater than min_eval_window
            if transformation is not None: # apply transformation if any
                if transformation_arguments is not None:
                    self.window_I = transformation(self.original_data, *transformation_arguments)[-len(self.window_I):]
                else:
                    self.window_I = transformation(self.original_data)[-len(self.window_I):]
            
            mask = np.isnan(self.window_I)
            if mask.any():
                idx = np.where(~mask, np.arange(len(self.window_I)), len(self.window_I) - 1)
                idx = np.minimum.accumulate(idx[::-1])[::-1]
                self.window_I = self.window_I[idx]
                
            # here we basically need only drift, the rest can be deteled (also from the resurn of the function)
            drift, _, _, _, _, _, _, _ = self.our_algo()
            drift_detected = int(drift)
        else: # if window size is less than min_eval_window
            drift_detected = None

        return bool(drift) if drift_detected is not None else False


    def our_algo(self, p_0=None, s_m=0, s_M=None):
        """
        Function to detect drift and adapt the window size.
        
        Parameters:
        window_I (list or array): The input window of size W.
        p_0 (float or None): The reference p-value. If None, indicates the first iteration.
        s_m (int): Minimum index to search for the new window's starting point.
        s_M (int): Maximum index to search for the new window's starting point.
        delta (float): Threshold value for the KS test p-value to detect drift.
        
        Returns:
        int or list: The starting index of the new window or the new window itself.
        """

        self.s_m = s_m
        self.s_M = s_M
        self.p_0 = p_0

        # Initialize s_M if not provided
        if s_M is None:
            self.s_M = len(self.window_I)
            self.window_aux = self.window_I
        # Step 1: Set the auxiliary window based on current indices
        else:
            self.s_M = s_M
            self.window_aux = self.window_aux[self.s_m:self.s_M]
        
        window_aux = self.window_aux
        
        
        # Step 2: Divide the auxiliary window into reference and recent sub-windows
        mid_index = int( np.floor( len(window_aux) / 2 ))
        window_0 = window_aux[:mid_index]   # Reference sub-window
        window_1 = window_aux[mid_index:]   # Recent sub-window

        # Step 3: Run the KS test to compute the p-value
        results = []
        p_values = []
        stats = []

 
        for alternative_data in self.list_of_alternatives:
            alternative_window_0, alternative_window_1 = self.alternative_func(window_0, window_1, alternative_data)
            stat, p_val = ks_2samp(alternative_window_0, alternative_window_1, method=self.method, alternative=self.alternative)
            results.append(p_val < self.delta  and stat > self.stat_value)
            p_values.append(p_val)
            stats.append(stat)

        # one more alternative -- comparing the means (p < 0.05 strong evidence means differ)
        if self.use_ttest:
            stat, p_val = ttest_ind(window_0, window_1, equal_var=False, alternative=self.alternative)
            results.append(p_val < self.delta and stat > self.stat_value)
            p_values.append(p_val)
            stats.append(stat)
        
        self.alternatives_pvals = p_values
        self.alternatives_results = results
        self.alternatives_stats = stats
        drift_detected = self.boyer_moore_majority_vote(results)
       
        # Step 4: Check if drift is detected
        if drift_detected:  # Drift detected
            self._shrink_window()

            return True, np.round(np.array(window_0), 2), np.round(np.array(window_1), 2), len(window_0), len(window_1), p_values, len(window_aux), results  # Trigger drift detection
        else:
    
            # Step 5: Adapt window size 
            if len(window_aux) >= self.eval_noshrink_window:                
                window_aux, self.s_m, self.s_M = self.window_shrink(window_aux, sm=self.s_m, sM=self.s_M)
                self.window_I = window_aux
              
        self.sm_list.append(self.s_m)
        self.sM_list.append(self.s_M)          
        
        return False, np.round(np.array(window_0), 2), np.round(np.array(window_1), 2), len(window_0), len(window_1), p_values, len(window_aux), results


    def _shrink_window(self):
        """
        Shrink the window by removing old elements after drift detection.
        """
        self.window_I = self.window_I[len(self.window_I)//2:]



    def reset(self):
        """
        Reset the KALEIDO detector by clearing the sliding window.
        """
        self.window = []
        self.s_m = 0
        self.s_M = None
        self.p_0 = None
        self.window_I = []

    # we do not use this function but it can be usefull to get the window size and location for each transformation
    def get_window_size(self):
        """
        Get the current size of the sliding window.

        :return: The size of the sliding window.
        """
        return np.array(self.window_size_list)
    
    def get_p_values(self):
        """
        Get the p_values of each alternative of the sliding window.

        :return: p values.
        """
        alternatives_pvals_fun = list(self.alternatives_pvals)
        self.alternatives_pvals = []
        return alternatives_pvals_fun
    
    def get_stat_values(self):
        alternatives_stats_fun = list(self.alternatives_stats)
        self.alternatives_stats = []
        return alternatives_stats_fun
    
    def get_decisions(self):
        """
        Get the drift results of each alternative of the sliding window.

        :return: drift results.
        """
        alternatives_drifts_fun = list(self.alternatives_results)
        self.alternatives_results = []
        return alternatives_drifts_fun

    
    def get_window_location(self):
        """
        Get the current location of the sliding window.

        :return: The starting and ending indices of the sliding window.
        """
        return self.sm_list, self.sM_list



    def window_shrink(self,WI, sm=0, sM=None, Wmin = 2 ):
        """
        Shrink the window based on detecting distribution changes.

        :param WI: Input window, list of values.
        :param sm: Minimum index to search for new window's starting point.
        :param sM: Maximum index to search for new window's starting point.
        :param val_ref_dec: Reference value for distribution change detection.
        :return: New window after shrinking.
        """
        break_point = False
        
        if sM is None:
            sM = len(WI)

       
        # Step 8: Set W0 to the first half of Waux
        midpoint = int( np.floor(sm + (len(WI) - sm) / 2))
        W0 = WI[sm : midpoint]

        # Step 9: Set W1 to the second half of Waux
        W1 = WI[midpoint:]

        # Step 10: Detect if there is a distribution change
        weHaveChange_alt_list = []
        for alternative in self.list_of_alternatives:
            alternative_W0, alternative_W1 = self.alternative_func(W0, W1, alternative)
            weHaveChange_alt = self.distribution_change(alternative_W0, alternative_W1, dist_change_method = self.dist_change_method)
            weHaveChange_alt_list.append(weHaveChange_alt)
        weHaveChange = self.boyer_moore_majority_vote(weHaveChange_alt_list)

        if not weHaveChange:
            # Step 12: Shrink by moving sm closer to the present
          
            sm_new = int(sm + np.ceil((sM - sm) / 2))
            break_point = True if len(WI)-sm_new <= self.min_eval_window else False
            sm = sm_new if not break_point else sM - self.min_eval_window #sm #set to min window size otherwise
            sM = sM

        else:
            # Step 15: If change is detected, move back
            sM = sm
            sm = max(0, int(np.floor(sm / 2)))


        # Step 18: If the window is still large enough to shrink
        if sM == sm or break_point:
            self.s_m = sm 
            self.s_M = sM
            self.window_I = WI[sm:len(WI)]
     
       
        else:
            # Step 19: Recursively shrink the window further
            self.window_I, sm, sM = self.window_shrink(WI, sm, sM)            

        
        return self.window_I, sm, sM

    def distribution_change(self,W0, W1, dist_change_method):
        """
        Detects if there's a distribution change between two sub-windows without drift.

        :param W0: First sub-window.
        :param W1: Second sub-window.
        :param self.delta : Reference value for detecting the change (e.g., p-value threshold).
        :param dist_change_method: Method to detect distribution changes.
        :return: True if a significant change is detected, False otherwise.
        """
        change_has_happened = False

        ## ks_test checks if the two samples are drawn from the same distribution
        if dist_change_method == 'ks':
            stat, pvalue = ks_2samp(W0, W1,method = self.method, alternative=self.alternative)

        ## ttest checks if the means and variances are the same
        elif dist_change_method == 'ttest':
            stat, pvalue = ttest_ind(W0, W1, equal_var=False)
            # stat, pvalue = ttest_rel(W0, W1)

        ## levene checks if the variances are the same
        elif dist_change_method == 'levene':
            stat, pvalue = levene(W0, W1)

  
        ## The Ljung-Box test for autocorrelation
        elif dist_change_method == 'ljungbox':
            lb1 = acorr_ljungbox(W0,lags = len(W0)-1, return_df=False)
            lb2 = acorr_ljungbox(W1, lags = len(W1)-1, return_df=False)
            print(f'pvalue1: {lb1.lb_pvalue}, pvalue2: {lb2.lb_pvalue}')


        # Step 5: Determine if the change is significant
        if pvalue < self.delta and stat > self.stat_value:  
              change_has_happened = True
       

        # Step 6: Return whether a change has happened
        return change_has_happened    

    # Boyer-Moore Majority Vote Algorithm
    def boyer_moore_majority_vote(self, nums):
        candidate = None
        count = 0

        # Phase 1: Find a candidate
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        # Phase 2: Verify the candidate
        count = 0
        for num in nums:
            if num == candidate:
                count += 1

        return candidate if count > len(nums) // 2 else None
        

    
    # use normal data without altenatives for drift detection test
    def use_normal_data(self, data):
        return data

    # use trend for drift detection test
    def extract_trend(self, data, period=2):
        stl = STL(data, period=period)
        # stl = STL(data, period=len(data))
        res = stl.fit()
        trend = res.trend
        return trend

    # use season for drift detection test
    def extract_season(self, data, period=2):
        stl = STL(data, period=period)
        # stl = STL(data, period=len(data))
        res = stl.fit()
        season = res.seasonal
        return season
    
    # use fft
    def fft_transform(self, data):
        data = data if isinstance(data, np.ndarray) else data.values

        yf = fft(data)
        return np.abs(yf)


    # alternative function wherein we pass functions defined above
    def alternative_func(self, window_0, window_1, alternFunct):
        
        window_0 = alternFunct(window_0)
        window_1 = alternFunct(window_1)

        return window_0, window_1
    
    # general nth derivative using np.gradient (length-preserving)
    def extract_nth_derivative(self, data, n=1):
        data = data if isinstance(data, np.ndarray) else data.values
        d = data
        for _ in range(n):
            d = np.gradient(d)
        return d