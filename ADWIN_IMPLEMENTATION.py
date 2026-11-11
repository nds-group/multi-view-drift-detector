import numpy as np
from collections import deque
from math import fabs, log, pow, sqrt
from typing import Deque
from scipy.stats import ks_2samp
import itertools

class AdaptiveWindowing:
    """ The helper class for ADWIN

    Parameters
    ----------
    delta
        Confidence value.
    clock
        How often ADWIN should check for change. 1 means every new data point, default is 32. Higher
        values speed up processing, but may also lead to increased delay in change detection.
    max_buckets
        The maximum number of buckets of each size that ADWIN should keep before merging buckets
        (default is 5).
    min_window_length
        The minimum length of each subwindow (default is 5). Lower values may decrease delay in
        change detection but may also lead to more false positives.
    grace_period
        ADWIN does not perform any change detection until at least this many data points have
        arrived (default is 10).
    """
    def __init__(self, delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10,cut_detector = 'ADWIN', method = 'auto', alternative = 'two-sided', window_size_filesave = False):
        self.delta = delta
        self.bucket_deque: Deque['Bucket'] = deque([Bucket(max_size=max_buckets)])
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.n_buckets = 0
        self.grace_period = grace_period
        self.tick = 0
        self.total_width = 0
        self.n_detections = 0
        self.clock = clock
        self.max_n_buckets = 0
        self.min_window_length = min_window_length
        self.max_buckets = max_buckets
        self.cut_detector = cut_detector
        self.method = method
        self.alternative = alternative
        self.window_size_filesave = window_size_filesave

    def get_n_detections(self):
        return self.n_detections

    def get_width(self):
        return self.width

    def get_total(self):
        return self.total

    def get_variance(self):
        return self.variance

    @property
    def variance_in_window(self):
        return self.variance / self.width if self.width > 0 else 0.0

    def update(self, value: float) -> bool:
        """Update the change detector with a single data point."""
        return self._update(value)

    def _update(self, value: float) -> bool:
        # Increment window with one element
        self._insert_element(value, 0.0)
        if self.window_size_filesave:
            with open(self.window_size_filesave, 'a') as f:
                f.write(str(len(self.bucket_deque)) + '\n')
        return self._detect_change()

    def _insert_element(self, value: float, variance: float):
        bucket = self.bucket_deque[0]
        bucket.insert_data(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

        # Update width, variance and total
        self.width += 1
        incremental_variance = 0.0
        if self.width > 1:
            incremental_variance = (
                (self.width - 1)
                * (value - self.total / (self.width - 1))
                * (value - self.total / (self.width - 1))
                / self.width
            )
        self.variance += incremental_variance
        self.total += value

        self._compress_buckets()

    @staticmethod
    def _calculate_bucket_size(row: int) -> float:
        return pow(2, row)

    def _delete_element(self) -> float:
        bucket = self.bucket_deque[-1]
        n = self._calculate_bucket_size(len(self.bucket_deque) - 1)  # length of bucket
        u = bucket.get_total_at(0)  # total of bucket
        mu = u / n  # mean of bucket
        v = bucket.get_variance_at(0)  # variance of bucket

        # Update width, total, and variance
        self.width -= n
        self.total -= u
        mu_window = self.total / self.width if self.width > 0 else 0.0  # mean of the window
        incremental_variance = (
            v + n * self.width * (mu - mu_window) * (mu - mu_window)
            / (n + self.width)
        )
        self.variance -= incremental_variance

        bucket.remove()
        self.n_buckets -= 1

        if bucket.current_idx == 0:
            self.bucket_deque.pop()

        return n

    def _compress_buckets(self):
        idx = 0
        while idx < len(self.bucket_deque):
            bucket = self.bucket_deque[idx]
            if bucket.current_idx == self.max_buckets + 1:
                try:
                    next_bucket = self.bucket_deque[idx + 1]
                except IndexError:
                    self.bucket_deque.append(Bucket(max_size=self.max_buckets))
                    next_bucket = self.bucket_deque[-1]
                n1 = self._calculate_bucket_size(idx)
                n2 = self._calculate_bucket_size(idx)
                mu1 = bucket.get_total_at(0) / n1
                mu2 = bucket.get_total_at(1) / n2

                ###### we compress the first two elements of the bucket into the next 
                ##### bucket having one element keeping the same statistics, 
                # we keep the sum of the two elements and its variance
                total12 = bucket.get_total_at(0) + bucket.get_total_at(1)
                temp = n1 * n2 * (mu1 - mu2) * (mu1 - mu2) / (n1 + n2)
                v12 = bucket.get_variance_at(0) + bucket.get_variance_at(1) + temp
                next_bucket.insert_data(total12, v12)
                self.n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.max_buckets:
                    break
            idx += 1

    def _detect_change(self) -> bool:
        change_detected = False
        self.tick += 1

        if self.tick % self.clock == 0 and self.width > self.grace_period:
            reduce_width = True
            while reduce_width:
                reduce_width = False
                n0 = 0.0  # length of window 0
                n1 = self.width  # length of window 1
                u0 = 0.0  # total of window 0
                u1 = self.total  # total of window 1
                v0 = 0.0  # variance of window 0
                v1 = self.variance  # variance of window 1

                if self.cut_detector == 'ADWIN':
                    for idx in range(len(self.bucket_deque) - 1, -1, -1):
                        bucket = self.bucket_deque[idx]
                        for k in range(bucket.current_idx - 1):
                            n2 = self._calculate_bucket_size(idx)
                            u2 = bucket.get_total_at(k)

                            if n0 > 0.0:
                                mu0 = u0 / n0
                                v0 += bucket.get_variance_at(k) + n0 * n2 * (mu0 - (u2 / n2)) ** 2 / (n0 + n2)

                            if n1 > 0.0:
                                mu1 = u1 / n1
                                v1 -= bucket.get_variance_at(k) + n1 * n2 * (mu1 - (u2 / n2)) ** 2 / (n1 + n2)

                            n0 += self._calculate_bucket_size(idx)
                            n1 -= self._calculate_bucket_size(idx)
                            u0 += bucket.get_total_at(k)
                            u1 -= bucket.get_total_at(k)

                            # if n0 < self.min_window_length:
                            #     n0 = self.min_window_length
                                
                            # if n1 < self.min_window_length:
                            #     n1 = self.min_window_length


                            

                            if ( n1 >= self.min_window_length and n0 >=self.min_window_length
                                 and self._evaluate_cut(n0, n1, u0 / n0 - u1 / n1, self.delta)):
                                reduce_width = True
                                change_detected = True
                                n0 -= self._delete_element()

                                break
                elif self.cut_detector == 'KSWIN':
                    # Perform pairwise comparisons using ks_2samp
                    comparisons = []

                    # for (i, j) in itertools.combinations(range(len(self.bucket_deque)-1), 2):
                    #     stat,p_value = ks_2samp(self.bucket_deque[i].variance_array, self.bucket_deque[j].variance_array)
                    # comparisons = []
        
                    for b in range(0,len(self.bucket_deque)-1):  # Compare Bucket[i] with Bucket[i+1]
                        # print (f'len of bucket_deque: {len(self.bucket_deque)}, i: {b}')

                        stat, p_value = ks_2samp(self.bucket_deque[b].variance_array, self.bucket_deque[b+1].variance_array,method = self.method, alternative=self.alternative)
                        
                    
                        if p_value < self.delta and stat > 0.1:
                            change_detected = True
                            reduce_width = True
                            n0 -= self._delete_element()
                            # comparisons.append((f'Bucket {b} vs Bucket {b+1}', p_value))
                            # print(comparisons)
                            break
                            

                            


                        # if self._evaluate_cut_KSWIN(n0, n1, u0 / n0 - u1 / n1, self.delta):
                        #     reduce_width = True
                        #     change_detected = True
                        #     n0 -= self._delete_element()
                        #     break

        self.total_width += self.width
        if change_detected:
            self.n_detections += 1

        return change_detected

    def _evaluate_cut(self, n0: float, n1: float, delta_mean: float, delta: float) -> bool:

        delta_prime = log(2 * log(self.width) / delta)

        m_recip = ((1.0 / (n0 - self.min_window_length + 1)) + (1.0 / (n1 - self.min_window_length + 1)))
        
        epsilon = (sqrt(2 * m_recip * self.variance_in_window * delta_prime) + 2 / 3 * delta_prime * m_recip)
        return fabs(delta_mean) > epsilon
    
    def _evaluate_cut_KSWIN(self, n0: float, n1: float, delta_mean: float, delta: float) -> bool:
        delta_prime = log(2 * log(self.width) / delta)
        # m_recip = ((1.0 / (n0 - self.min_window_length + 1)) + (1.0 / (n1 - self.min_window_length + 1)))
        # epsilon = (sqrt(2 * m_recip * self.variance_in_window * delta_prime) + 2 / 3 * delta_prime * m_recip)
        # stat, pvalue = ks_2samp(n0, n1, method = 'exact', alternative='less')
        stat, pvalue = ks_2samp(n0, n1)
        if pvalue < delta:
            return True

        return False


class Bucket:
    """ A bucket class to keep statistics."""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_idx = 0
        self.total_array = np.zeros(self.max_size + 1, dtype=float)
        self.variance_array = np.zeros(self.max_size + 1, dtype=float)

    def clear_at(self, index):
        self.set_total_at(0.0, index)
        self.set_variance_at(0.0, index)

    def insert_data(self, value: float, variance: float):
        self.set_total_at(value, self.current_idx)
        self.set_variance_at(variance, self.current_idx)
        self.current_idx += 1

    def remove(self):
        self.compress(1)

    def compress(self, n_elements: int):
        window_len = len(self.total_array)
        for i in range(n_elements, window_len):
            self.total_array[i - n_elements] = self.total_array[i]
            self.variance_array[i - n_elements] = self.variance_array[i]
        for i in range(window_len - n_elements, window_len):
            self.clear_at(i)
        self.current_idx -= n_elements

    def get_total_at(self, index: int) -> float:
        return self.total_array[index]

    def get_variance_at(self, index: int) -> float:
        return self.variance_array[index]

    def set_total_at(self, value: float, index: int):
        self.total_array[index] = value

    def set_variance_at(self, value: float, index: int):
        self.variance_array[index] = value

    def get_total_array(self):
        return self.total_array
