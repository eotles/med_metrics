"""
Compatibility Metrics Module
============================

This module contains functions to evaluate the compatibility of predictions made by machine learning models
in medical contexts. It focuses on assessing how predictions change when models are updated or when different
models are used on the same data.

Functions:
- backwards_trust_compatibility: Evaluates trust compatibility between two model predictions, focusing on cases where the first model's predictions are correct.
- backwards_error_compatibility: Assesses error compatibility between two model predictions, focusing on instances where the first model's predictions are incorrect.
- rank_based_compatibility: Measures the consistency of ranking between original and updated model scores based on true labels.
"""

# Author: Erkin Ötleş, hi@eotles.com

import numpy as np
from .utils import check_consistent_length

def backwards_trust_compatibility(y_true, y_pred_0, y_pred_1):
    """
    Calculate Backwards Trust Compatibility (BTC) between two prediction sets.

    BTC quantifies the agreement between two models, focusing on instances where
    the first model's prediction matches the true label. It's useful for
    assessing the trustworthiness of the second model's predictions based on the
    first model's correct predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred_0 : array-like of shape (n_samples,)
        Predictions from the first model.
    y_pred_1 : array-like of shape (n_samples,)
        Predictions from the second model.

    Returns
    -------
    float
        BTC score, representing the proportion of correct predictions by the
        first model that are also correct in the second model.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred_0 = [0, 1, 0, 1]
    >>> y_pred_1 = [0, 1, 1, 0]
    >>> btc = backwards_trust_compatibility(y_true, y_pred_0, y_pred_1)
    >>> print(btc)

    Notes
    -----
    BTC is especially meaningful in scenarios where AI models are updated
    frequently, as it helps in understanding the impact of these updates on
    the established trust and reliability from the perspective of the end-user.

    References
    ----------
    - Gagan Bansal, Besmira Nushi, Ece Kamar, et al. "A Case for Backward
      Compatibility for Human-AI Teams." arXiv:1906.01148v1 [cs.HC], Jun 2019.
    """
    
    check_consistent_length(y_true, y_pred_0, y_pred_1)
    
    g = np.vstack([y_true, y_pred_0, y_pred_1]).T
    trust_candidates = g[g[:,0] == g[:,1]]
    
    if trust_candidates.shape[0] <= 0:
        raise ValueError("No y_pred_0 correct predictions present. Backwards Trust Compatibility is not defined in that case.")

    trust_matches = trust_candidates[trust_candidates[:,1] == trust_candidates[:,2]]

    btc = trust_matches.shape[0]/trust_candidates.shape[0]
    return(btc)


def backwards_error_compatibility(y_true, y_pred_0, y_pred_1):
    """
    Calculate Backwards Error Compatibility (BEC) between two prediction sets.

    BEC assesses the agreement between two models' predictions, focusing on
    instances where the first model's prediction does not match the true label.
    It measures consistency in errors between two models.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred_0 : array-like of shape (n_samples,)
        Predictions from the first model.
    y_pred_1 : array-like of shape (n_samples,)
        Predictions from the second model.

    Returns
    -------
    float
        BEC score, quantifying the proportion of incorrect predictions by the
        first model that are also incorrect in the second model.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred_0 = [1, 0, 1, 0]
    >>> y_pred_1 = [1, 0, 0, 1]
    >>> bec = backwards_error_compatibility(y_true, y_pred_0, y_pred_1)
    >>> print(bec)

    Notes
    -----
    BEC is especially meaningful in scenarios where AI models are updated
    frequently, as it helps in understanding the impact of these updates on
    the established trust and reliability from the perspective of the end-user.

    References
    ----------
    - Gagan Bansal, Besmira Nushi, Ece Kamar, et al. "A Case for Backward
      Compatibility for Human-AI Teams." arXiv:1906.01148v1 [cs.HC], Jun 2019.
    """
    
    check_consistent_length(y_true, y_pred_0, y_pred_1)
    
    g = np.vstack([y_true, y_pred_0, y_pred_1]).T
    error_candidates = g[g[:,0] != g[:,1]]
    
    if error_candidates.shape[0] <= 0:
        raise ValueError("No y_pred_0 errors present. Backwards Error Compatibility is not defined in that case.")
    
    error_matches = error_candidates[error_candidates[:,1] == error_candidates[:,2]]

    bec = error_matches.shape[0]/error_candidates.shape[0]
    return(bec)


def rank_based_compatibility(y_true, y_score_original, y_score_updated):
    """
    Evaluate the consistency in instance ranking between original and updated
    model scores using Rank-Based Compatibility (CR).

    This metric, based on Otles et al. (2023), quantifies the degree to which the ranking order of instances,
    based on their scores, is preserved when transitioning from an original
    model to an updated model. It is particularly useful in clinical settings for
    maintaining consistent risk stratification, crucial for prioritizing patient
    treatments and resource allocation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score_original : array-like of shape (n_samples,)
        Scores from the original model.
    y_score_updated : array-like of shape (n_samples,)
        Scores from the updated model.

    Returns
    -------
    float
        Compatibility score, representing the proportion of consistent rankings
        between the original and updated model scores.

    Examples
    --------
    >>> y_true = [1, 0, 1, 0]
    >>> y_score_original = [0.7, 0.3, 0.6, 0.2]
    >>> y_score_updated = [0.8, 0.1, 0.5, 0.3]
    >>> rank_based_compatibility(y_true, y_score_original, y_score_updated)
    0.75

    Notes
    -----
    CR evaluates the ordering of patient-pairs and is crucial in scenarios where
    models are used for clinical decision-making and resource allocation. It offers
    a more direct assessment of model updates in healthcare, beyond traditional
    metrics like AUROC.

    References
    ----------
    - Erkin Ötleş, Brian T. Denton, Jenna Wiens. "Updating Clinical Risk Stratification
      Models Using Rank-Based Compatibility: Approaches for Evaluating and Optimizing
      Clinician-Model Team Performance." arXiv:2308.05619v1 [stat.ML], 2023.
    """

    
    a = y_score_original
    b = y_score_updated

    def make_pair_matrix(y, p):
        pm = p[y==1] > p[y==0][:, None]
        return(pm)

    apm = make_pair_matrix(y_true, a)
    bpm = make_pair_matrix(y_true, b)
    return np.sum(apm*bpm)/np.sum(apm)
