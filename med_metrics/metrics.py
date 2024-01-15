"""
Metrics Module for Medical Machine Learning Evaluation

This module provides specialized metrics for evaluating machine learning models in medical contexts. 
It includes functions for calculating the Number Needed to Treat (NNT) versus the number of patients treated 
at various decision thresholds, computing the average height of the NNT vs. treated curve, and assessing 
net benefit across different thresholds.

Functions:
- average_NNTvsTreated: Computes the average height of the NNT vs. treated curve.
- net_benefit: Computes the net benefit of a binary classifier at a given threshold.
- average_net_benefit: Determines the average net benefit over all thresholds.

Author: Erkin Ötleş
Email: hi@eotles.com
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from .curves import NNTvsTreated_curve, net_benefit_curve


def average_NNTvsTreated(y_true, y_score, rho, pos_label=None, sample_weight=None,
                         min_treated=None, max_treated=None):
    """
    Computes the average height of the Number Needed to Treat (NNT) vs. treated curve for a binary classifier.

    This function calculates the NNT at various thresholds and determines the
    average value over the specified range of treated patients. The average NNT
    provides insights into the overall effectiveness of the intervention across
    different thresholds.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels.
    y_score : ndarray of shape (n_samples,)
        Predicted probabilities or decision function outputs.
    rho : float
        Effect size of the intervention, between 0 and 1.
    pos_label : int, float, bool, or str, default=None
        Label of the positive class.
    sample_weight : ndarray of shape (n_samples,), default=None
        Weights for samples.
    min_treated : int, default=None
        Minimum number of treated patients to consider in the curve.
    max_treated : int, default=None
        Maximum number of treated patients to consider in the curve.

    Returns:
    -------
    float
        Average height of the NNT vs. treated curve.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> rho = 0.5
    >>> average_NNTvsTreated(y_true, y_score, rho)
    2.25
        
    Notes
    -----
    ***

    References
    ----------
    - Any relevant literature or studies.
    """
    
    treated, NNT, _ = NNTvsTreated_curve(y_true, y_score, rho, pos_label=pos_label, sample_weight=sample_weight, min_treated=min_treated, max_treated=max_treated)

    # Trapz calculates the area under the curve, here representing the total NNT
    auc = np.trapz(NNT, treated)

    # Calculate the average height of the curve
    average_height = auc / (treated.max() - treated.min())

    return average_height


def net_benefit(y_true, y_score, decision_threshold=0.5):
    """
    Computes the net benefit of a binary classifier at a specific decision
    threshold.

    Net benefit is a metric that quantifies the trade-off between true positives
    and false positives at a given threshold. It is particularly useful in
    medical decision-making contexts.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels.
    y_score : ndarray of shape (n_samples,)
        Predicted scores from the classifier.
    decision_threshold : float, default=0.5
        Threshold for classifying an instance as positive.

    Returns:
    -------
    float
        Net benefit score at the specified decision threshold.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.2, 0.6, 0.3, 0.8]
    >>> net_benefit(y_true, y_score, 0.5)
    0.5
        
    Notes
    -----
    ***

    References
    ----------
    - Any relevant literature or studies.
    """
    
    y_score = np.array(y_score)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_score>=decision_threshold).ravel()
    n = tn+fp+fn+tp

    # Net benefit formula:
    net_benefit_score = (tp - fp*(decision_threshold/(1-decision_threshold)))/n
    
    return net_benefit_score


def average_net_benefit(y_true, y_score, pos_label=None, sample_weight=None,
                        min_threshold=0.0, max_threshold=1.0):
    """
    Calculates the average net benefit of a binary classifier across a specified
    range of thresholds.

    Net benefit is a key metric in medical decision-making, quantifying the
    trade-offs at different thresholds. This function averages the net benefit
    over the range of thresholds, giving a comprehensive view of classifier
    performance.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels.
    y_score : ndarray of shape (n_samples,)
        Predicted scores from the classifier.
    pos_label : int, float, bool, or str, default=None
        Label of the positive class.
    sample_weight : ndarray of shape (n_samples,), default=None
        Weights for samples.
    min_threshold : float, default=0.0
        Minimum threshold to consider for the calculation.
    max_threshold : float, default=1.0
        Maximum threshold to consider for the calculation.

    Returns:
    -------
    float
        The average net benefit across the evaluated thresholds.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> average_net_benefit(y_true, y_score)
    0.3898046398046398
        
    Notes
    -----
    ***

    References
    ----------
    - Any relevant literature or studies.
    """
    # Calculate the net benefit across all thresholds
    thresholds, net_benefit_scores, _ = net_benefit_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight, min_threshold=min_threshold, max_threshold=max_threshold)
    
    # Calculate the area under the curve (AUC)
    # need to reverse the orders because the thresholds (net_benefit_scores) are in descending order
    auc = np.trapz(net_benefit_scores[::-1], thresholds[::-1])
    
    # Average net benefit over the range of thresholds
    average_height = auc / (thresholds.max() - thresholds.min())
    
    return average_height
    
