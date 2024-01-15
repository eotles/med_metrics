"""
Unit Tests for Compatibility Metrics Module

***

Functions:
***

Author: Erkin Ötleş
Email: hi@eotles.com
"""

import unittest
from med_metrics.compatiblity_metrics import *

class test_bTc_score(unittest.TestCase):
    
    def test_inconsistent_lengths(self):       
        self.assertRaises(ValueError, backwards_trust_compatibility, [0,0], [0], [0])
        self.assertRaises(ValueError, backwards_trust_compatibility, [0], [0,0], [0])
        self.assertRaises(ValueError, backwards_trust_compatibility, [0], [0], [0,0])
    
    def test_no_correct_prediction(self): 
        self.assertRaises(ValueError, backwards_trust_compatibility, [0], [1], [0])
        self.assertRaises(ValueError, backwards_trust_compatibility, [1], [0], [0])
        self.assertRaises(ValueError, backwards_trust_compatibility, [0], [1], [1])
        self.assertRaises(ValueError, backwards_trust_compatibility, [1], [0], [1])
        
    def test_calculation(self):
        for n in range(10, 100):
            y_true = np.random.choice([0,1], size=n)
            y_pred_0 = np.random.choice([0,1], size=n)
            y_pred_1 = np.random.choice([0,1], size=n)

            
            _btc = np.mean((y_pred_0==y_pred_1)[y_true==y_pred_0]) 
            self.assertEqual(_btc, backwards_trust_compatibility(y_true, y_pred_0, y_pred_1))
        
        
class test_bEc_score(unittest.TestCase):
    
    def test_inconsistent_lengths(self):       
        self.assertRaises(ValueError, backwards_error_compatibility, [0,0], [0], [0])
        self.assertRaises(ValueError, backwards_error_compatibility, [0], [0,0], [0])
        self.assertRaises(ValueError, backwards_error_compatibility, [0], [0], [0,0])
    
    def test_no_errors(self): 
        self.assertRaises(ValueError, backwards_error_compatibility, [0], [0], [0])
        self.assertRaises(ValueError, backwards_error_compatibility, [1], [1], [0])
        self.assertRaises(ValueError, backwards_error_compatibility, [0], [0], [1])
        self.assertRaises(ValueError, backwards_error_compatibility, [1], [1], [1])
        
    def test_calculation(self):
        for n in range(10, 100):
            y_true = np.random.choice([0,1], size=n)
            y_pred_0 = np.random.choice([0,1], size=n)
            y_pred_1 = np.random.choice([0,1], size=n)

            
            _bec = np.mean((y_pred_0==y_pred_1)[y_true!=y_pred_0]) 
            self.assertEqual(_bec, backwards_error_compatibility(y_true, y_pred_0, y_pred_1))



'''
For testing rank_based_compatibility
'''

from sklearn import utils

def slope_sign(score_point_i, score_point_j):
    point_diff = score_point_i - score_point_j
    sign = point_diff/abs(point_diff)
    return sign

def pr_btc_score(y_true, y_pred_0, y_pred_1):
    utils.check_consistent_length(y_true, y_pred_0, y_pred_1)
    points = [_ for _ in zip(y_true, y_pred_0, y_pred_1)]
    
    n_pairs = 0
    n_compatible_pairs = 0
    
    for idx_i, i in enumerate(points):
        for j in points[idx_i+1:]:
            if i[0]!=j[0] and i[1]!=j[1] and i[2]!=j[2]:
                true_ss = slope_sign(i[0], j[0])
                y_pred_0_ss = slope_sign(i[1], j[1])
                y_pred_1_ss = slope_sign(i[2], j[2])
                
                if true_ss==y_pred_0_ss:
                    n_pairs+=1
                    if y_pred_0_ss==y_pred_1_ss:
                        n_compatible_pairs+=1
    
    #print("{:0.3f}".format(n_compatible_pairs/n_pairs))
    #return(n_compatible_pairs, n_pairs, n_compatible_pairs/n_pairs)
    return n_compatible_pairs/n_pairs

#pr_btc_score(y_true, y_score, y_score_updated)



if __name__ == '__main__':
    unittest.main()
