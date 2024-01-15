"""
Utility Functions for Medical Machine Learning Metrics

This module contains utility functions used in the calculation of machine learning metrics 
for medical applications. It includes functions for generating binary classification curves, 
confusion matrix curves, and functions to handle dictionary inputs for metric and curve functions.

Functions:
- _binary_clf_curve: Calculate true and false positives per binary classification threshold.
- _cm_curve: Calculate confusion matrix per binary classification threshold.
- _get_funcs_dict: Handle dictionary inputs for metric and curve functions.
- _get_funcs_kwargs_dict: Handle dictionary inputs for metric and curve functions kwargs.

Some of this code is adapted from the scikit-learn project: 
https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/metrics/_ranking.py

Author: Erkin Ötleş
Email: hi@eotles.com
"""

import numpy as np
import types

from sklearn.utils.multiclass import type_of_target

from sklearn.utils import (
    assert_all_finite,
    #check_array,
    check_consistent_length,
    column_or_1d,
)

from sklearn.utils.extmath import stable_cumsum


def _check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, float, bool, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int, float, bool or str
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
        
    Notes
    -----
    Taken from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    if pos_label is None:
        # Compute classes only if pos_label is not specified:
        classes = np.unique(y_true)
        if classes.dtype.kind in "OUS" or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        ):
            classes_repr = ", ".join([repr(c) for c in classes.tolist()])
            raise ValueError(
                f"y_true takes value in {{{classes_repr}}} and pos_label is not "
                "specified: either make y_true take value in {0, 1} or "
                "{-1, 1} or pass pos_label explicitly."
            )
        pos_label = 1

    return pos_label
    
    

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Compute true and false positives at different thresholds for binary
    classification.

    This function sorts scores and calculates the cumulative true and false
    positives across decreasing score thresholds. It's used to generate points
    for ROC or precision-recall curves.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score : array-like of shape (n_samples,)
        Estimated probabilities or decision function outputs.
    pos_label : int, float, bool or str, default=None
        The label of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        False positives count at each threshold.
    tps : ndarray of shape (n_thresholds,)
        True positives count at each threshold.
    thresholds : ndarray of shape (n_thresholds,)
        Thresholds at which fps and tps are calculated.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    """
    
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true, input_name="y_true")
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    # Filter out zero-weighted samples, as they should not impact the result
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = _check_sample_weight(sample_weight, y_true)
        nonzero_weight_mask = sample_weight != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weight = sample_weight[nonzero_weight_mask]

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    #threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def _cm_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Generate confusion matrix elements (TP, FP, TN, FN) for varying thresholds.

    This function computes true and false positives/negatives for different score thresholds. Useful in contexts where the full confusion matrix is needed at various operational points.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score : array-like of shape (n_samples,)
        Estimated probabilities or decision function outputs.
    pos_label : int, float, bool or str, default=None
        The label of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        False positives count at each threshold.
    tps : ndarray of shape (n_thresholds,)
        True positives count at each threshold.
    tns : ndarray of shape (n_thresholds,)
        True negatives count at each threshold.
    fns : ndarray of shape (n_thresholds,)
        False negatives count at each threshold.
    thresholds : ndarray of shape (n_thresholds,)
        Thresholds at which fps, tps, tns, and fns are calculated.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_score = [0.1, 0.4, 0.35, 0.8]
    >>> fps, tps, tns, fns, thresholds = _cm_curve(y_true, y_score)
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    tns = fps[-1] - fps
    fns = tps[-1] - tps

    return fps, tps, tns, fns, thresholds



def _get_funcs_dict(funcs, funcs_name='parameter'):
    """
    Convert input functions to a standardized dictionary format.

    This utility function ensures that the input, whether a single function, list, or dictionary of functions, is transformed into a uniform dictionary format. It's primarily used to standardize metric and curve function inputs.

    Parameters
    ----------
    funcs : function or list of functions or dict
        Single function, list of functions, or dictionary of functions.
    funcs_name : str, default='parameter'
        Name to use in error messages for the functions parameter.

    Returns
    -------
    dict
        Dictionary where keys are function names and values are the corresponding functions.

    Examples
    --------
    >>> funcs = [roc_auc_score, average_precision_score]
    >>> funcs_dict = _get_funcs_dict(funcs)
    """
    # Handle dictionary inputs for metric_funcs and curve_funcs
    if funcs is None:
        funcs_dict = {}
    elif isinstance(funcs, types.FunctionType):
        funcs_dict = {funcs.__name__: funcs}
    elif isinstance(funcs, list):
        funcs_dict = {}
        for func in funcs:
            if isinstance(func, types.FunctionType):
                funcs_dict[func.__name__] = func
            else:
                raise ValueError("{} list of contains a non-function entity.".format(funcs_name))    
    elif isinstance(funcs, dict):
        funcs_dict = funcs
    else:
        raise ValueError("{} must be a single function, list of functions, or dictionary of functions.".format(funcs_name))

    return funcs_dict


def _get_funcs_kwargs_dict(funcs_dict, funcs_kwargs, funcs_kwargs_name='parameter'):
    """
    Map function-specific keyword arguments to corresponding functions.

    This utility function aligns additional keyword arguments (kwargs) to their respective functions in a standardized dictionary format. It's used to provide specific arguments to metric and curve functions.

    Parameters
    ----------
    funcs_dict : dict
        Dictionary of functions.
    funcs_kwargs : dict
        Dictionary of keyword arguments for each function.
    funcs_kwargs_name : str, default='parameter'
        Name to use in error messages for the functions kwargs parameter.

    Returns
    -------
    dict
        Dictionary where keys are function names and values are dictionaries of kwargs for each function.

    Examples
    --------
    >>> funcs_dict = {'roc_auc_score': roc_auc_score}
    >>> funcs_kwargs = {'roc_auc_score': {'average': 'macro'}}
    >>> kwargs_dict = _get_funcs_kwargs_dict(funcs_dict, funcs_kwargs)
    """
    
    funcs_kwargs_dict = {k: {} for k in funcs_dict}
    if funcs_kwargs is None:
        pass #nothing to do
    elif isinstance(funcs_kwargs, dict):
        for k, v in funcs_kwargs.items():
            if k in funcs_dict:
                if isinstance(v, dict):
                    funcs_kwargs_dict[k] = v
                else:
                    raise ValueError("{}[{}] is not kwargs (dictionary format).".format(funcs_kwargs_name, k))
            else:
                raise ValueError("{} is not a valid keyword for {}.".format(k, funcs_kwargs_name))
    else:
        raise ValueError("{} must be none or dictionary of kwargs (dictionary format).".format(funcs_kwargs_name))
        
    return funcs_kwargs_dict
    


def _check_in_range(parameter_value, lb=-float('inf'), ub=float('inf'), parameter_name='parameter'):
    """
    Check if a parameter value is within specified bounds.

    Parameters
    ----------
    parameter_value : numeric
        The value of the parameter to be checked.
    lb : numeric, default=-float('inf')
        The lower bound for the parameter value.
    ub : numeric, default=float('inf')
        The upper bound for the parameter value.
    parameter_name : str, default='parameter'
        The name of the parameter for error messages.

    Raises
    ------
    ValueError
        If parameter_value is not within the bounds (lb, ub).

    Examples
    --------
    >>> _check_in_range(5, lb=0, ub=10, parameter_name='example_param')
    """
    
    if parameter_value < lb:
        raise ValueError(f"{parameter_name} set to {parameter_value}, must be greater than {lb}")
    elif ub < parameter_value:
        raise ValueError(f"{parameter_name} set to {parameter_value}, must be less n {ub}")
        

def _check_min_max(min_parameter_value, min_parameter_name,
                  max_parameter_value, max_parameter_name,
                  lb=-float('inf'), ub=float('inf')):
    """
    Check if minimum and maximum parameter values are within specified bounds and correctly ordered.

    Parameters
    ----------
    min_parameter_value : numeric
        The minimum parameter value.
    min_parameter_name : str
        The name of the minimum parameter.
    max_parameter_value : numeric
        The maximum parameter value.
    max_parameter_name : str
        The name of the maximum parameter.
    lb : numeric, default=-float('inf')
        The lower bound for the parameter values.
    ub : numeric, default=float('inf')
        The upper bound for the parameter values.

    Raises
    ------
    ValueError
        If min_parameter_value > max_parameter_value or if values are not within the bounds (lb, ub).

    Examples
    --------
    >>> _check_min_max(0, 'min_val', 10, 'max_val', lb=0, ub=100)
    """
    
    _check_in_range(min_parameter_value, lb=lb, ub=ub, parameter_name=min_parameter_name)
    _check_in_range(max_parameter_value, lb=lb, ub=ub, parameter_name=max_parameter_name)
    
    if min_parameter_value > max_parameter_value:
        raise ValueError(f"{min_parameter_name} greater than {max_parameter_name}, {min_parameter_value}>{max_parameter_value}")
        


def _validate_ys(y_true, y_scores):
    """
    Validate and process y_true and y_scores to ensure they are in the correct format.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_scores : numpy array, list, or dict
        Predicted scores; can be a single array, a list of arrays, or a dictionary of arrays.

    Returns
    -------
    tuple
        y_true and y_scores formatted as numpy arrays.

    Raises
    ------
    ValueError
        If y_scores is not a numpy array, a list, or a dictionary.
        If lengths of y_scores elements do not match the length of y_true.

    Examples
    --------
    >>> y_true = [0, 1, 0]
    >>> y_scores = [0.2, 0.6, 0.1]
    >>> y_true, y_scores = _validate_ys(y_true, y_scores)
    """
    
    # Convert y_true to a numpy array if it's not already
    y_true = np.asarray(y_true)

    # Process y_scores to ensure it's a dictionary of numpy arrays
    if isinstance(y_scores, np.ndarray):
        y_scores = {'model_0': y_scores}
    elif isinstance(y_scores, list):
        y_scores = {'model_{}'.format(i): np.asarray(score) for i, score in enumerate(y_scores)}
    elif isinstance(y_scores, dict):
        y_scores = {key: np.asarray(score) for key, score in y_scores.items()}
    else:
        raise ValueError("y_scores must be a numpy array, a list, or a dictionary.")

    # Validate that all elements in y_scores are numpy arrays and have the same length as y_true
    for key, score in y_scores.items():
        if len(score) != len(y_true):
            raise ValueError(f"Length of score array with key '{key}' does not match length of y_true.")

    return y_true, y_scores
    
    
def _lighten_color(color, amount=0.5):
    """
    Lighten a given color by blending it with white.

    Parameters:
    - color: The original color (as an RGB tuple).
    - amount: The weight of the color against white: 0, all white, to 1, all color.

    Returns:
    - Lightened color.
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

