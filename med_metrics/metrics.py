"""
Metrics Module
==============

This module provides specialized metrics for evaluating machine learning models in medical contexts. 
It includes functions for calculating the Number Needed to Treat (NNT) versus the number of patients treated 
at various decision thresholds, computing the average height of the NNT vs. treated curve, and assessing 
net benefit across different thresholds.

Functions:
- average_NNTvsTreated: Computes the average height of the NNT vs. treated curve.
- net_benefit: Computes the net benefit of a binary classifier at a given threshold.
- average_net_benefit: Determines the average net benefit over all thresholds.
"""

# Author: Erkin Ötleş, hi@eotles.com

import numpy as np
from sklearn.metrics import confusion_matrix
from .curves import NNTvsTreated_curve, net_benefit_curve


def average_NNTvsTreated(
    y_true,
    y_score,
    rho,
    pos_label=None,
    sample_weight=None,
    min_treated=None,
    max_treated=None,
    *,
    policy: str = "finite",    # {"finite", "propagate", "clip"}
    epsilon: float = 1e-12      # used only if policy == "clip"
):
    """
    Compute the average height of the NNT-vs-treated curve.

    This metric summarizes the Number Needed to Treat (NNT) across decision
    thresholds by taking the area under the NNT-vs-treated curve and dividing
    by the treated span:
        average NNT = AUC_NNT(treated) / (treated_max - treated_min).

    Definitions
    -----------
    - treated(th) = TP(th) + FP(th) : number of patients that would be treated
      at threshold th.
    - PPV(th)     = TP(th) / (TP(th) + FP(th)) for treated(th) > 0; 0 otherwise.
    - ARR(th)     = rho * PPV(th), where `rho` is the relative risk reduction
      of the intervention (0 ≤ rho ≤ 1).
    - NNT(th)     = 1 / ARR(th). If ARR(th) = 0, NNT(th) = ∞.

    Numerical policy
    ----------------
    The `policy` argument controls how zero-benefit regions (ARR = 0 → NNT = ∞)
    are handled when averaging:
      * "finite" (default): Integrate only over thresholds where NNT is finite
        (ARR > 0). If no finite region exists, returns np.inf.
      * "propagate": Integrate over the full curve including ∞/NaN. Standard
        numpy behavior applies (result may be inf or nan).
      * "clip": Floor ARR at `epsilon` (thus clipping NNT from above at 1/epsilon)
        before integrating. Ensures finiteness but introduces an explicit floor.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score : array-like of shape (n_samples,)
        Predicted scores (e.g., probabilities or decision function).
    rho : float
        Relative risk reduction (effect size) of the intervention in [0, 1].
    pos_label : int, float, bool, or str, optional
        Label of the positive class; forwarded to the internal confusion-matrix
        curve routine.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    min_treated : int, optional
        Minimum number of treated patients to include when forming the curve.
        Defaults to 0 if not provided.
    max_treated : int, optional
        Maximum number of treated patients to include when forming the curve.
        Defaults to n_samples if not provided.
    policy : {"finite", "propagate", "clip"}, default="finite"
        Handling of zero-benefit regions as described above.
    epsilon : float, default=1e-12
        ARR floor used only if `policy="clip"` (effective NNT ceiling is 1/epsilon).

    Returns
    -------
    float
        Average NNT over the selected domain.
        Special cases:
          - Returns `np.inf` if `policy="finite"` and no finite-NNT region exists.
          - Returns `np.nan` if the treated span is degenerate
            (treated_max == treated_min) or if the curve is empty.

    Notes
    -----
    - The curve includes an anchor at treated = 0. Under this implementation the
      anchor does not contribute to the integral unless it is part of the finite
      region (policy-dependent).
    - When reporting this metric, consider also reporting a separate coverage
      measure (see `NNTvsTreated_coverage`) to indicate what fraction of the
      treated range actually had finite NNT.

    Examples
    --------
    >>> y_true  = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> rho = 0.5
    >>> round(average_NNTvsTreated(y_true, y_score, rho, policy="finite"), 3)
    2.25
    >>> # Using a floor to avoid infinities:
    >>> round(average_NNTvsTreated(y_true, y_score, rho, policy="clip", epsilon=1e-6), 3)
    2.25
    """
    
    treated, NNT, _ = NNTvsTreated_curve(
        y_true, y_score, rho,
        pos_label=pos_label,
        sample_weight=sample_weight,
        min_treated=min_treated,
        max_treated=max_treated
    )

    if treated.size == 0:
        return np.nan

    span = treated.max() - treated.min()
    if span <= 0:
        # Degenerate domain: either a single threshold or no variation
        # With no treated variation, “average height” is undefined.
        return np.nan

    if policy == "finite":
        m = np.isfinite(NNT) & (NNT > 0)
        if not np.any(m):
            return np.inf
        auc = np.trapz(NNT[m], treated[m])
        return auc / (treated[m].max() - treated[m].min())

    if policy == "clip":
        # Floor ARR via epsilon by clipping NNT from above
        # NNT = 1/ARR -> clip ARR by epsilon == clip NNT by 1/epsilon
        NNT = np.minimum(NNT, 1.0 / epsilon)

    # "propagate" or post-clip: may still yield inf/nan if present
    auc = np.trapz(NNT, treated)
    return auc / span
    
    
def NNTvsTreated_coverage(
    y_true,
    y_score,
    rho,
    pos_label=None,
    sample_weight=None,
    min_treated=None,
    max_treated=None
) -> float:
    """
    Fraction of the treated span where NNT is finite (ARR > 0).

    This scalar complements `average_NNTvsTreated` by indicating how much of the
    treated range exhibits any expected benefit (finite NNT). It is defined as:

        coverage = (treated_max_finite - treated_min_finite) / (treated_max - treated_min)

    where “finite” refers to thresholds with ARR > 0 (i.e., NNT finite).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score : array-like of shape (n_samples,)
        Predicted scores (e.g., probabilities or decision function).
    rho : float
        Relative risk reduction (effect size) of the intervention in [0, 1].
    pos_label : int, float, bool, or str, optional
        Label of the positive class.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    min_treated : int, optional
        Minimum number of treated patients to include when forming the curve.
    max_treated : int, optional
        Maximum number of treated patients to include when forming the curve.

    Returns
    -------
    float
        Coverage in [0, 1].
        Special cases:
          - Returns 0.0 if the curve is empty or the treated span is degenerate.
          - Returns 0.0 if no finite-NNT region exists.

    Notes
    -----
    - This implementation measures the span between the smallest and largest
      treated values that have finite NNT and normalizes by the total treated
      span. If finite-NNT regions are *disjoint*, this “convex-hull” approach
      can over-estimate the fraction. If you need exact set-measure coverage,
      compute the normalized sum of finite sub-interval lengths (e.g., integrate
      an indicator over `treated`) instead.

    Examples
    --------
    TODO: check this
    >>> y_true  = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> rho = 0.5
    >>> 0.0 <= NNTvsTreated_coverage(y_true, y_score, rho) <= 1.0
    True
    """
    treated, NNT, _ = NNTvsTreated_curve(
        y_true, y_score, rho,
        pos_label=pos_label,
        sample_weight=sample_weight,
        min_treated=min_treated,
        max_treated=max_treated
    )

    if treated.size == 0:
        return 0.0

    span = treated.max() - treated.min()
    if span <= 0:
        return 0.0

    m = np.isfinite(NNT) & (NNT > 0)
    if not np.any(m):
        return 0.0

    covered = treated[m].max() - treated[m].min()
    return float(covered / span)


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
    
