"""

Functions:
- NNTvsTreated_curve: Calculates NNT vs. treated per classification threshold.
- net_benefit_curve: Calculates the net benefit across a range of thresholds.

Author: Erkin Ötleş
Email: hi@eotles.com
"""

import numpy as np
from .utils import _cm_curve, _check_min_max



def NNTvsTreated_curve(y_true, y_score, rho, pos_label=None, sample_weight=None,
                       min_treated=None, max_treated=None):
    """
    Calculate the Number Needed to Treat (NNT) vs. treated curve at various
    decision thresholds.

    This function computes NNT, an important metric in medical decision making,
    across different thresholds of a binary classifier, allowing for analysis
    within a specified range of treated patients.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels.
    y_score : ndarray of shape (n_samples,)
        Predicted probabilities or decision function outputs from a classifier.
    rho : float
        Effect size of the intervention, between 0 and 1.
    pos_label : int, float, bool, str, default=None
        The label of the positive class in binary classification.
    sample_weight : ndarray of shape (n_samples,), default=None
        Weights for samples.
    min_treated : int, default=None
        Minimum number of treated patients to consider in the curve.
    max_treated : int, default=None
        Maximum number of treated patients to consider in the curve.

    Returns:
    -------
    treated : ndarray
        Array of the number of treated patients at each threshold.
    NNT : ndarray
        Number Needed to Treat at each threshold.
    thresholds : ndarray
        Evaluated thresholds.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> rho = 0.5
    >>> NNTvsTreated_curve(y_true, y_score, rho)
    (array([0., 1., 2., 3., 4.]), array([0., 2., 2., 3., 4.]), array([ inf, 0.8 , 0.4 , 0.35, 0.1 ]))
    
    Notes
    -----
    ***

    References
    ----------
    - Any relevant literature or studies.
    """
    
    n = len(y_true)
    min_treated = min_treated or 0
    max_treated = max_treated or n
    
    _check_min_max(min_treated, 'min_treated', max_treated, 'max_treated', lb=0, ub=n)
    
    fps, tps, tns, fns, thresholds = _cm_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    
    # Calculate absolute risk reduction for each threshold
    absolute_risk_reduction = rho * tps/(tps + fps)
    # Number Needed to Treat: number of patients to treat to prevent one additional bad outcome
    NNT = 1 / (absolute_risk_reduction)
    # Total number of patients treated at each threshold
    treated = tps + fps

    # Prepending 0 to arrays to include the starting point of the curve
    NNT = np.insert(NNT, 0, 0)
    treated = np.insert(treated, 0, 0)
    thresholds = np.insert(thresholds, 0, np.inf)
    
    # Find in range (valid) elements
    valid_indices = (min_treated <= treated) & (treated <= max_treated)
    treated = treated[valid_indices]
    NNT = NNT[valid_indices]
    thresholds = thresholds[valid_indices]
    

    return treated, NNT, thresholds



def net_benefit_curve(y_true, y_score, pos_label=None, sample_weight=None,
                      min_threshold=0.0, max_threshold=1.0):
    """
    Calculate the net benefit curve of a binary classifier across a range of
    decision thresholds.

    This function assesses the net benefit, a metric balancing true positives
    and false positives, across different thresholds, allowing for analysis
    within a specified threshold range.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels.
    y_score : ndarray of shape (n_samples,)
        Predicted scores from a classifier.
    pos_label : int, float, bool, str, default=None
        The label of the positive class in binary classification.
    sample_weight : ndarray of shape (n_samples,), default=None
        Weights for samples.
    min_threshold : float, default=0.0
        Minimum threshold value to include in the analysis.
    max_threshold : float, default=1.0
        Maximum threshold value to include in the analysis.

    Returns:
    -------
    thresholds : ndarray
        Evaluated thresholds.
    net_benefit_scores : ndarray
        Net benefit scores at each threshold.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> net_benefit_curve(y_true, y_score)
    (array([0.8 , 0.4 , 0.35, 0.1 ]), array([0.25      , 0.5       , 0.36538462, 0.44444444]), array([0.8 , 0.4 , 0.35, 0.1 ]))
        
    Notes
    -----
    ***

    References
    ----------
    - Any relevant literature or studies.
    """
    
    _check_min_max(min_threshold, 'min_threshold', max_threshold, 'max_threshold', lb=0, ub=1)
    
    # Create confusion matrices at different thresholds and calculate net benefit for each
    fps, tps, tns, fns, thresholds = _cm_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    n = len(y_true)
    
    # Net benefit at each threshold
    net_benefit_scores = (tps - fps*(thresholds/(1-thresholds)))/n
    
    # Find in range (valid) elements
    valid_indices = (min_threshold <= thresholds) & (thresholds <= max_threshold)
    net_benefit_scores = net_benefit_scores[valid_indices]
    thresholds = thresholds[valid_indices]

    return thresholds, net_benefit_scores, thresholds
