"""
Curves Module
=============

Functions:
- NNTvsTreated_curve: Calculates NNT vs. treated per classification threshold.
- net_benefit_curve: Calculates the net benefit across a range of thresholds.
"""

# Author: Erkin Ötleş, hi@eotles.com

import numpy as np
from .utils import _cm_curve, _check_min_max



def NNTvsTreated_curve(
    y_true,
    y_score,
    rho,
    pos_label=None,
    sample_weight=None,
    min_treated=None,
    max_treated=None,
    *,
    warn: str = "auto"   # {"auto", "never", "always"}
):
    """
    Calculate the Number Needed to Treat (NNT) vs. treated curve at various
    decision thresholds.

    This function computes NNT, an important metric in medical decision making,
    across different thresholds of a binary classifier, allowing for analysis
    within a specified range of treated patients.
    
    Semantics:
      - treated(th) = TP + FP
      - PPV(th)     = TP / (TP + FP) for treated>0; 0 otherwise
      - ARR(th)     = rho * PPV(th)
      - NNT(th)     = 1 / ARR(th); if ARR==0 -> NNT = ∞

    Warning behavior (param `warn`):
      - "auto"  (default): suppress NumPy runtime divides, but issue a single
        warning if the in-range curve has **no finite NNT** at all (i.e., ARR==0
        everywhere across [min_treated, max_treated]).
      - "never": never warn.
      - "always": always emit a one-line summary warning.

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
    min_treated = 0 if min_treated is None else min_treated
    max_treated = n if max_treated is None else max_treated
    _check_min_max(min_treated, 'min_treated', max_treated, 'max_treated', lb=0, ub=n)

    fps, tps, tns, fns, thresholds = _cm_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    treated = tps + fps

    # Robust divisions with warnings suppressed locally
    with np.errstate(divide='ignore', invalid='ignore'):
        ppv = np.where(treated > 0, tps / treated, 0.0)
        arr = rho * ppv
        # NNT = 1/ARR; where ARR==0 -> inf (semantically correct)
        NNT = np.where(arr > 0, 1.0 / arr, np.inf)

    # Prepend anchor at treated=0 (NNT undefined -> use ∞ to avoid bias)
    treated = np.insert(treated, 0, 0.0)
    NNT = np.insert(NNT, 0, np.inf)
    thresholds = np.insert(thresholds, 0, np.inf)

    # Restrict to requested treated range
    in_range = (min_treated <= treated) & (treated <= max_treated)
    treated = treated[in_range]
    NNT = NNT[in_range]
    thresholds = thresholds[in_range]

    # Optional human-friendly warning (single line), not NumPy spam
    if warn != "never":
        # "finite NNT exists" means any ARR>0 → any NNT finite
        has_finite = np.any(np.isfinite(NNT) & (NNT > 0))
        if warn == "always" or (warn == "auto" and not has_finite):
            warnings.warn(
                "NNTvsTreated_curve: no finite NNT within the selected treated range "
                f"[{min_treated}, {max_treated}]; ARR==0 throughout. Returning ∞ values.",
                RuntimeWarning
            )

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
