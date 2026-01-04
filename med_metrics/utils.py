"""
Utility Functions
=================

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
"""

# Author: Erkin Ötleş, hi@eotles.com

import numpy as np
import types
import warnings  # NEW: used for high-cardinality warnings
import hashlib   # NEW: used for stable integer hashing (deterministic RNG)

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


# =============================================================================
# NEW: Lightweight array and dict helpers (additive; do not break existing code)
# =============================================================================

def _to_np(a):
    """
    Convert input to numpy array if not already an ndarray.

    Parameters
    ----------
    a : array-like

    Returns
    -------
    ndarray
    """
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)


def _check_lengths(*arrays):
    """
    Ensure all arrays have the same length.

    Parameters
    ----------
    arrays : list of array-like

    Returns
    -------
    int
        Length of the arrays.

    Raises
    ------
    ValueError
        If lengths do not match.
    """
    lengths = [len(_to_np(a)) for a in arrays if a is not None]
    if not lengths:
        raise ValueError("No arrays provided to check lengths.")
    N = lengths[0]
    for L in lengths[1:]:
        if L != N:
            raise ValueError("Input arrays must have same length; got {}.".format(lengths))
    return N


def ensure_dict(name, value, default_key="value"):
    """
    Accept a single array or dict of arrays and return a standardized dict[str, ndarray].

    Parameters
    ----------
    name : str
        Name used in error messages.
    value : array-like or dict[str, array-like]
    default_key : str, default='value'
        Key to use when value is a single array.

    Returns
    -------
    dict
        Mapping of names to numpy arrays.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = _to_np(v)
        if len(out) == 0:
            raise ValueError("{} dict is empty.".format(name))
        _ = _check_lengths(*out.values())
        return out
    else:
        arr = _to_np(value)
        return {default_key: arr}


def dict_cartesian(labels, scores):
    """
    Build list of (label_name, model_name) pairs from two dicts.

    Parameters
    ----------
    labels : dict[str, ndarray]
    scores : dict[str, ndarray]

    Returns
    -------
    list of tuple
        (label_name, model_name) pairs.
    """
    return [(ln, mn) for ln in labels.keys() for mn in scores.keys()]


# =============================================================================
# NEW: Grouping / Binning utilities for `group_by` (additive)
# =============================================================================

def is_categorical(col, max_categorical_cardinality):
    """
    Decide if a column should be treated as categorical.

    Rules
    -----
    - If dtype is object/string/bool -> categorical.
    - If dtype is numeric -> NOT categorical (will be binned by default).
      (Users can still force categorical by passing pre-binned/labelled arrays.)
    - The cardinality heuristic only applies to non-numeric types.

    Parameters
    ----------
    col : array-like
    max_categorical_cardinality : int or None

    Returns
    -------
    bool
    """
    col_np = _to_np(col)

    # Categorical by type
    if col_np.dtype == np.bool_:
        return True
    if col_np.dtype.kind in ("U", "S", "O"):
        # Optional: warn on extremely high-cardinality categorical strings/objects
        if max_categorical_cardinality is not None:
            try:
                nunique = np.unique(col_np).size
            except Exception:
                nunique = np.unique(col_np).size
            if nunique > max_categorical_cardinality:
                warnings.warn(
                    "High-cardinality categorical column detected (unique={}, max_categorical_cardinality={}). "
                    "Consider numeric encoding/binning upstream.".format(nunique, max_categorical_cardinality),
                    RuntimeWarning
                )
        return True

    # Numeric types default to numeric handling (i.e., will be binned)
    return False



def make_bins(values, strategy="quantile", q=5, bins=None, edges=None, labels=None):
    """
    Compute bin assignments and labels for numeric arrays.

    Parameters
    ----------
    values : array-like
    strategy : {'quantile','uniform','custom'}, default='quantile'
    q : int, default=5
        Number of quantile bins (if strategy='quantile').
    bins : int, optional
        Number of equal-width bins (if strategy='uniform').
    edges : sequence of float, optional
        Bin edges (if strategy='custom').
    labels : sequence of str, optional
        Labels for bins (if strategy='custom').

    Returns
    -------
    codes : ndarray of int
        Bin indices per sample; -1 used for NaN (handled upstream).
    label_strings : list of str
        Human-readable labels for each bin.
    """
    v = _to_np(values)
    if v.dtype.kind in "fc":
        mask_nan = np.isnan(v)
    else:
        mask_nan = np.zeros_like(v, dtype=bool)

    if strategy == "custom":
        if edges is None or len(edges) < 2:
            raise ValueError("Custom strategy requires 'edges' with at least 2 values.")
        edges = np.asarray(edges, dtype=float)
        K = len(edges) - 1
        codes = np.digitize(v, edges, right=False) - 1
        # clamp out-of-range
        codes[(v < edges[0]) | (v > edges[-1])] = -1
        if labels is None:
            lbls = ["[{:.3g},{:.3g})".format(edges[i], edges[i+1]) for i in range(K)]
        else:
            if len(labels) != K:
                raise ValueError("labels length must match number of bins.")
            lbls = [str(x) for x in labels]
        codes[mask_nan] = -1
        return codes, lbls

    if strategy == "uniform":
        k = int(bins if bins is not None else 5)
        if k < 1:
            raise ValueError("uniform bins must be >=1")
        lo, hi = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.full_like(v, fill_value=0, dtype=int), ["[{:.3g},{:.3g}]".format(lo, hi)]
        edges = np.linspace(lo, hi, k + 1)
        codes = np.digitize(v, edges, right=False) - 1
        codes[codes == k] = k - 1
        lbls = ["[{:.3g},{:.3g})".format(edges[i], edges[i+1]) for i in range(k)]
        codes[mask_nan] = -1
        return codes, lbls

    if strategy == "quantile":
        k = int(q)
        if k < 1:
            raise ValueError("quantile q must be >=1")
        finite = v[~mask_nan]
        if finite.size == 0:
            return np.full_like(v, fill_value=0, dtype=int), ["[NaN]"]
        quantiles = np.linspace(0, 1, k + 1)
        edges = np.quantile(finite, quantiles)
        edges = np.unique(edges)  # ensure strictly increasing
        if edges.size == 1:
            return np.full_like(v, fill_value=0, dtype=int), ["[{:.3g}]".format(edges[0])]
        codes = np.digitize(v, edges, right=False) - 1
        K = edges.size - 1
        codes = np.clip(codes, 0, K - 1)
        lbls = ["[{:.3g},{:.3g})".format(edges[i], edges[i+1]) for i in range(K)]
        codes[mask_nan] = -1
        return codes, lbls

    raise ValueError("Unknown binning strategy: {!r}".format(strategy))


def resolve_groups(group_by,
                   group_binning=None,
                   N=None,
                   max_categorical_cardinality=100,
                   missing_group_name="Missing",
                   warn_high_cardinality=True):
    """
    Build per-group boolean index masks from a `group_by` mapping.

    Parameters
    ----------
    group_by : dict[str, array-like] or None
        Columns used to split the population.
    group_binning : dict[str, dict], optional
        Per-column binning configuration for numeric columns.
        Examples:
            {'age': {'strategy':'quantile','q':5}}
            {'age': {'strategy':'uniform','bins':10}}
            {'age': {'strategy':'custom','edges':[0,18,40,65,120], 'labels':['0-18','18-40','40-65','65+']}}
    N : int, optional
        Length of the population; if None, inferred from columns.
    max_categorical_cardinality : int or None, default=100
        If not None, columns with n_unique <= this are treated as categorical by default.
    missing_group_name : str, default='Missing'
        Label to use for missing values (NaN/None).
    warn_high_cardinality : bool, default=True
        Emit a warning when a categorical-like column seems too high-cardinality.

    Returns
    -------
    dict
        groups[group_name][level_label] -> boolean mask (length N).
    """
    if group_by is None:
        return {}

    # coerce to numpy + length checks
    cols_np = {}
    for k, v in group_by.items():
        cols_np[str(k)] = _to_np(v)
    if N is None:
        N = _check_lengths(*cols_np.values())
    else:
        _check_lengths(np.arange(N), *cols_np.values())

    resolved = {}
    for gname, col in cols_np.items():
        cfg = (group_binning or {}).get(gname, {}) if group_binning else {}
        col_np = _to_np(col)

        # Missing mask
        if col_np.dtype.kind in "fc":
            miss = np.isnan(col_np)
        else:
            # None-check is intentional for object dtype
            miss = (col_np == None)  # noqa: E711

        # Decide categorical vs numeric
        cat = is_categorical(col_np, max_categorical_cardinality)

        # Guard: very high-cardinality "categorical" columns
        if cat and max_categorical_cardinality is not None:
            try:
                nunique = np.unique(col_np[~miss]).size
            except Exception:
                nunique = np.unique(col_np).size
            frac = nunique / max(1, N)
            if nunique > max_categorical_cardinality or frac > 0.10:
                msg = ("group '{}' appears high-cardinality (unique={}, N={}, frac~{:.2f}). "
                       "Consider numeric binning via group_binning or set max_categorical_cardinality=None."
                       .format(gname, nunique, N, frac))
                if warn_high_cardinality:
                    warnings.warn(msg, RuntimeWarning)
                if col_np.dtype.kind in "ifc":
                    cat = False  # fall back to numeric handling

        levels = {}

        if cat:
            # Treat as categorical
            try:
                uniq = np.unique(col_np[~miss])
            except Exception:
                uniq = np.unique(col_np)
            for u in uniq:
                mask = (col_np == u)
                levels[str(u)] = mask
        else:
            # Numeric binning
            strategy = cfg.get("strategy", "quantile")
            if strategy == "custom":
                codes, lbls = make_bins(
                    col_np.astype(float),
                    strategy="custom",
                    edges=cfg.get("edges"),
                    labels=cfg.get("labels"),
                )
            elif strategy == "uniform":
                codes, lbls = make_bins(
                    col_np.astype(float),
                    strategy="uniform",
                    bins=int(cfg.get("bins", 5)),
                )
            else:
                # default quantile
                codes, lbls = make_bins(
                    col_np.astype(float),
                    strategy="quantile",
                    q=int(cfg.get("q", 5)),
                )
            for code, lbl in enumerate(lbls):
                mask = (codes == code)
                levels[str(lbl)] = mask

        # Missing level (if any)
        if miss.any():
            levels[str(missing_group_name)] = miss

        resolved[str(gname)] = levels

    return resolved


# =============================================================================
# NEW: Bootstrap / RNG helpers for deterministic alignment (additive)
# =============================================================================

def stable_int_hash(*parts):
    """
    Hash a sequence of parts into a stable 64-bit integer.

    Used to derive deterministic RNG seeds for (group, level) nodes so that
    bootstrap replicates are aligned across labels and models within a group.

    Parameters
    ----------
    parts : sequence
        Components to hash.

    Returns
    -------
    int
        Non-negative integer suitable for seeding np.random.default_rng.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def make_bootstrap_indices(N, B, rng, sample_size=None):
    """
    Draw bootstrap indices WITH replacement.

    Parameters
    ----------
    N : int
        Population size.
    B : int
        Number of bootstrap replicates.
    rng : numpy.random.Generator
        Random generator to use.
    sample_size : int or None, default=None
        Size of each bootstrap resample; if None, uses N.

    Returns
    -------
    ndarray of shape (B, M)
        Indices for each bootstrap replicate.
    """
    M = int(sample_size) if sample_size is not None else int(N)
    return rng.integers(low=0, high=int(N), size=(int(B), M), endpoint=False, dtype=np.int64)


# =============================================================================
# NEW: Convenience helpers for navigating results (additive)
# =============================================================================

def iter_groups(node):
    """
    Iterate groups in a result node.

    Parameters
    ----------
    node : dict
        Node with optional 'groups' key.

    Yields
    ------
    (group_name, level_name, group_node) : tuple
        Group label, level label, and the nested node.
    """
    groups = node.get("groups", {})
    for gname, levels in groups.items():
        for lvl, sub in levels.items():
            yield gname, lvl, sub


def summarize_metrics(node, ci=0.95):
    """
    Summarize point estimates and bootstrap CIs for the current node.

    Parameters
    ----------
    node : dict
        Node containing 'original_metrics' and 'bootstrap_replication_metrics'.
        Expected shape:
            original_metrics = {model: {metric: float}}
            bootstrap_replication_metrics = {model: {metric: 1D array}}
    ci : float, default=0.95
        Confidence level.

    Returns
    -------
    dict
        summary[model][metric] = {'point': float, 'ci_low': float, 'ci_high': float}
    """
    alpha = (1 - ci) / 2.0
    original = node.get('original_metrics', {}) or {}
    boots = node.get('bootstrap_replication_metrics', {}) or {}

    summary = {}
    for model, metrics in original.items():
        summary[model] = {}
        for mname, point in metrics.items():
            arr = np.asarray((boots.get(model, {}) or {}).get(mname, []), dtype=float)
            if arr.size:
                lo = float(np.quantile(arr, alpha))
                hi = float(np.quantile(arr, 1 - alpha))
            else:
                lo = np.nan
                hi = np.nan
            summary[model][mname] = {'point': float(point), 'ci_low': lo, 'ci_high': hi}
    return summary
