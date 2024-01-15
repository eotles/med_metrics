"""
Unit Tests for Medical Machine Learning Metrics

This module contains unit tests for the metrics functions defined in the med_metrics package. 
It uses the unittest framework to test the functionality and correctness of various metrics.

Author: Erkin Ötleş
Email: hi@eotles.com
"""

import unittest
from med_metrics.metrics import *


class TestNNTMetrics(unittest.TestCase):
    def test_NNTvsTreated_curve(self):
        # Example test case for NNTvsTreated_curve
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.8, 0.1, 0.9, 0.4, 0.7])
        rho = 0.5
        treated, NNT, thresholds = NNTvsTreated_curve(y_true, y_score, rho)
        # Include assertions to verify the correctness of the output

    def test_average_NNTvsTreated(self):
        # Example test case for average_NNTvsTreated
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.8, 0.1, 0.9, 0.4, 0.7])
        rho = 0.5
        average_height = average_NNTvsTreated(y_true, y_score, rho)
        # Include assertions to verify the correctness of the output

if __name__ == '__main__':
    unittest.main()
