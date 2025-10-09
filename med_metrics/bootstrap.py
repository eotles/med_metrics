"""
Bootstrap Evaluation Module
========================================

This module contains functions for performing bootstrap evaluations of machine learning models in medical applications. 
It includes functionalities to analyze bootstrapped results, calculate confidence intervals, and plot the results of these analyses.
The module focuses on providing tools for assessing model performance through bootstrapped metrics and curves.

Some of this code is adapted from the scipy project: 
https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_resampling.py
"""

# Author: Erkin Ötleş, hi@eotles.com

from .utils import _get_funcs_dict, _get_funcs_kwargs_dict, _validate_ys, _lighten_color
import copy
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np


def _validate_category_vectors(category_vectors, n_samples):
    """Validate optional category vectors used for subgroup analyses."""

    if category_vectors is None:
        return {}

    if isinstance(category_vectors, np.ndarray):
        category_vectors = {"category_0": category_vectors}
    elif isinstance(category_vectors, list):
        category_vectors = {
            f"category_{idx}": np.asarray(values) for idx, values in enumerate(category_vectors)
        }
    elif isinstance(category_vectors, dict):
        category_vectors = {name: np.asarray(values) for name, values in category_vectors.items()}
    else:
        raise ValueError(
            "category_vectors must be None, a numpy array, a list of arrays, or a dictionary of arrays."
        )

    for name, values in category_vectors.items():
        if len(values) != n_samples:
            raise ValueError(
                f"Category '{name}' has {len(values)} samples but expected {n_samples}."
            )

    return category_vectors


def _initialize_result_bundle(metric_func_dict, curve_func_dict,
                              compatibility_metric_func_dict, y_scores,
                              y_score_key_pairs, bootstrap=False):
    """Create a container for storing metric, curve, and compatibility results."""

    metric_storage = {
        metric_name: {score_name: ([] if bootstrap else None) for score_name in y_scores}
        for metric_name in metric_func_dict
    }
    curve_storage = {
        curve_name: {score_name: ([] if bootstrap else None) for score_name in y_scores}
        for curve_name in curve_func_dict
    }
    compatibility_storage = {
        metric_name: {pair: ([] if bootstrap else None) for pair in y_score_key_pairs}
        for metric_name in compatibility_metric_func_dict
    }

    return {
        'metrics': metric_storage,
        'curves': curve_storage,
        'compatibility_metrics': compatibility_storage,
    }


def _store_value(container, key, value, bootstrap):
    """Store a single value or append to a bootstrap list."""
    if bootstrap:
        container[key].append(value)
    else:
        container[key] = value


def _evaluate_into(result_bundle, y_true, y_scores,
                   metric_func_dict, metric_kwarg_dict,
                   curve_func_dict, curve_kwarg_dict,
                   compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                   y_score_key_pairs, bootstrap=False):
    """Evaluate metrics, curves, and compatibility metrics for given data."""

    if y_true.size == 0:
        # No samples available for this subgroup; fill with neutral values.
        for metric_name, score_dict in result_bundle['metrics'].items():
            for score_name in score_dict:
                _store_value(score_dict, score_name, np.nan, bootstrap)
        for curve_name, score_dict in result_bundle['curves'].items():
            for score_name in score_dict:
                _store_value(score_dict, score_name, None, bootstrap)
        for metric_name, pair_dict in result_bundle['compatibility_metrics'].items():
            for pair in pair_dict:
                _store_value(pair_dict, pair, np.nan, bootstrap)
        return

    for metric_name, metric_func in metric_func_dict.items():
        for score_name, y_score in y_scores.items():
            try:
                value = metric_func(y_true, y_score, **metric_kwarg_dict[metric_name])
            except Exception:
                value = np.nan
            _store_value(result_bundle['metrics'][metric_name], score_name, value, bootstrap)

    for curve_name, curve_func in curve_func_dict.items():
        for score_name, y_score in y_scores.items():
            try:
                value = curve_func(y_true, y_score, **curve_kwarg_dict[curve_name])
            except Exception:
                value = None
            _store_value(result_bundle['curves'][curve_name], score_name, value, bootstrap)

    for metric_name, metric_func in compatibility_metric_func_dict.items():
        for pair in y_score_key_pairs:
            try:
                value = metric_func(
                    y_true,
                    y_scores[pair[0]],
                    y_scores[pair[1]],
                    **compatibility_metric_kwarg_dict[metric_name],
                )
            except Exception:
                value = np.nan
            _store_value(result_bundle['compatibility_metrics'][metric_name], pair, value, bootstrap)


def _finalize_bootstrap_storage(result_bundle):
    """Convert bootstrap lists to numpy arrays for easier downstream use."""

    for metric_name, score_dict in result_bundle['metrics'].items():
        for score_name, values in score_dict.items():
            result_bundle['metrics'][metric_name][score_name] = np.asarray(values)

    for curve_name, score_dict in result_bundle['curves'].items():
        for score_name, values in score_dict.items():
            # Keep list structure to preserve arbitrary curve return types
            result_bundle['curves'][curve_name][score_name] = list(values)

    for metric_name, pair_dict in result_bundle['compatibility_metrics'].items():
        for pair, values in pair_dict.items():
            result_bundle['compatibility_metrics'][metric_name][pair] = np.asarray(values)


def _resolve_outcome_container(bootstrapped_results, outcome_name=None,
                               category_name=None, category_value=None):
    """Retrieve the target container for overall or subgroup results."""

    outcomes = bootstrapped_results['outcomes']
    if outcome_name is None:
        if len(outcomes) != 1:
            raise ValueError(
                "Multiple outcomes present; specify outcome_name to disambiguate the results."
            )
        outcome_name = next(iter(outcomes))

    if outcome_name not in outcomes:
        raise KeyError(f"Outcome '{outcome_name}' not found in bootstrapped results.")

    outcome_container = outcomes[outcome_name]

    if category_name is None:
        return outcome_container['overall']

    subgroups = outcome_container.get('subgroups', {})
    if category_name not in subgroups:
        raise KeyError(f"Category '{category_name}' not found for outcome '{outcome_name}'.")

    category_groups = subgroups[category_name]
    if category_value not in category_groups:
        raise KeyError(
            f"Value '{category_value}' not found in category '{category_name}' for outcome '{outcome_name}'."
        )

    return category_groups[category_value]


def bootstrap_evaluation(y_true, y_scores,
                        metric_funcs, metric_funcs_kwargs=None,
                        curve_funcs=None, curve_funcs_kwargs=None,
                        compatibility_metric_funcs=None, compatibility_metric_funcs_kwargs=None,
                        n_bootstraps=1000, random_state=None,
                        category_vectors=None,
                       ):
    """
    Perform bootstrapping for machine learning metric and curve evaluations.

    This function generates bootstrapped samples for various metrics, curves, and compatibility measures, providing insights into model performance.

    Parameters:
    ----------
    y_true : array-like or dict of array-like
        True labels. When a dictionary is provided, each key is treated as a separate outcome and evaluated simultaneously.
    y_scores : array-like, list of arrays, or dict of arrays
        Target scores or predicted labels. If dict, keys are used as identifiers.
    metric_funcs : dict or callable
        Functions to calculate metrics. If a single function is provided, it's used for all y_scores.
    metric_funcs_kwargs : dict, optional
        Additional keyword arguments for each metric function.
    curve_funcs : dict or callable, optional
        Functions to generate curves.
    curve_funcs_kwargs : dict, optional
        Additional keyword arguments for each curve function.
    compatibility_metric_funcs : dict or callable, optional
        Functions for compatibility metrics between different y_scores.
    compatibility_metric_funcs_kwargs : dict, optional
        Additional keyword arguments for each compatibility metric function.
    n_bootstraps : int, default=1000
        Number of bootstrap iterations.
    random_state : int or RandomState, optional
        Random number generator seed for reproducibility.
    category_vectors : array-like, list, or dict, optional
        Category labels with length ``n_samples`` used for subgroup analyses. When provided, metrics are computed for each
        unique value within each category in addition to the overall results.

    Returns:
    -------
    dict
        A dictionary containing original and bootstrapped metric, curve, and compatibility results for every outcome and
        (optionally) subgroup, along with metadata about the evaluated categories and bootstrap indices.

    Examples:
    --------
    >>> y_true = [0, 1, 1, 0]
    >>> y_scores = [0.1, 0.4, 0.6, 0.2]
    >>> results = bootstrap_evaluation(y_true, y_scores, metric_funcs=my_metric_func)
    >>> print(results)
    
    Notes:
    -----
    The function is adapted from the scipy project's bootstrap function.
    
    References:
    ----------
    - https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_resampling.py
    """

    

    # Validate inputs and convert y_scores to a dictionary of numpy arrays
    y_true_dict, y_scores = _validate_ys(y_true, y_scores)

    # Determine dataset characteristics
    n_samples = len(next(iter(y_true_dict.values())))
    category_vectors = _validate_category_vectors(category_vectors, n_samples)
    category_levels = {
        name: np.unique(values) for name, values in category_vectors.items()
    }

    # Initialize random number generator and bootstrap indices
    rng = np.random.default_rng(random_state)
    bootstrap_indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))

    # Convert function lists/dicts to standardized dict format
    metric_func_dict = _get_funcs_dict(metric_funcs, 'metric_funcs')
    curve_func_dict = _get_funcs_dict(curve_funcs, 'curve_funcs')
    compatibility_metric_func_dict = _get_funcs_dict(compatibility_metric_funcs, 'compatibility_metric_funcs')

    # Prepare kwargs for each type of function
    metric_kwarg_dict = _get_funcs_kwargs_dict(metric_func_dict, metric_funcs_kwargs, 'metric_funcs_kwargs')
    curve_kwarg_dict = _get_funcs_kwargs_dict(curve_func_dict, curve_funcs_kwargs, 'curve_funcs_kwargs')
    compatibility_metric_kwarg_dict = _get_funcs_kwargs_dict(
        compatibility_metric_func_dict, compatibility_metric_funcs_kwargs, 'compatibility_metric_funcs_kwargs'
    )

    y_score_key_pairs = [pair for pair in permutations(y_scores.keys(), 2)]

    # Prepare storage for results
    outcomes_results = {}
    for outcome_name in y_true_dict:
        outcome_storage = {
            'overall': {
                'original': _initialize_result_bundle(
                    metric_func_dict, curve_func_dict, compatibility_metric_func_dict,
                    y_scores, y_score_key_pairs, bootstrap=False
                ),
                'bootstrap': _initialize_result_bundle(
                    metric_func_dict, curve_func_dict, compatibility_metric_func_dict,
                    y_scores, y_score_key_pairs, bootstrap=True
                ),
            }
        }

        if category_vectors:
            subgroup_storage = {}
            for category_name, values in category_levels.items():
                subgroup_storage[category_name] = {}
                for category_value in values:
                    subgroup_storage[category_name][category_value] = {
                        'original': _initialize_result_bundle(
                            metric_func_dict, curve_func_dict, compatibility_metric_func_dict,
                            y_scores, y_score_key_pairs, bootstrap=False
                        ),
                        'bootstrap': _initialize_result_bundle(
                            metric_func_dict, curve_func_dict, compatibility_metric_func_dict,
                            y_scores, y_score_key_pairs, bootstrap=True
                        ),
                    }
            outcome_storage['subgroups'] = subgroup_storage

        outcomes_results[outcome_name] = outcome_storage

    # Compute original metric and curve results
    for outcome_name, outcome_values in y_true_dict.items():
        outcome_storage = outcomes_results[outcome_name]
        _evaluate_into(
            outcome_storage['overall']['original'], outcome_values, y_scores,
            metric_func_dict, metric_kwarg_dict,
            curve_func_dict, curve_kwarg_dict,
            compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
            y_score_key_pairs, bootstrap=False
        )

        if category_vectors:
            for category_name, category_array in category_vectors.items():
                for category_value in category_levels[category_name]:
                    mask = category_array == category_value
                    indices = np.flatnonzero(mask)
                    subset_true = outcome_values[indices]
                    subset_scores = {score_name: score_values[indices] for score_name, score_values in y_scores.items()}
                    subgroup_bundle = outcome_storage['subgroups'][category_name][category_value]['original']
                    _evaluate_into(
                        subgroup_bundle,
                        subset_true,
                        subset_scores,
                        metric_func_dict,
                        metric_kwarg_dict,
                        curve_func_dict,
                        curve_kwarg_dict,
                        compatibility_metric_func_dict,
                        compatibility_metric_kwarg_dict,
                        y_score_key_pairs,
                        bootstrap=False,
                    )

    # Generate bootstrapped samples and calculate metrics
    for indices in bootstrap_indices:
        resampled_scores = {score_name: score_values[indices] for score_name, score_values in y_scores.items()}
        resampled_outcomes = {
            outcome_name: outcome_values[indices]
            for outcome_name, outcome_values in y_true_dict.items()
        }

        for outcome_name, resampled_true in resampled_outcomes.items():
            outcome_storage = outcomes_results[outcome_name]
            _evaluate_into(
                outcome_storage['overall']['bootstrap'], resampled_true, resampled_scores,
                metric_func_dict, metric_kwarg_dict,
                curve_func_dict, curve_kwarg_dict,
                compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                y_score_key_pairs, bootstrap=True
            )

        if category_vectors:
            for category_name, category_array in category_vectors.items():
                resampled_category = category_array[indices]
                for category_value in category_levels[category_name]:
                    group_indices = np.flatnonzero(resampled_category == category_value)
                    subset_scores = {
                        score_name: score_values[group_indices]
                        for score_name, score_values in resampled_scores.items()
                    }

                    for outcome_name, resampled_true in resampled_outcomes.items():
                        subgroup_bundle = outcomes_results[outcome_name]['subgroups'][category_name][category_value]['bootstrap']
                        subset_true = resampled_true[group_indices]
                        _evaluate_into(
                            subgroup_bundle,
                            subset_true,
                            subset_scores,
                            metric_func_dict,
                            metric_kwarg_dict,
                            curve_func_dict,
                            curve_kwarg_dict,
                            compatibility_metric_func_dict,
                            compatibility_metric_kwarg_dict,
                            y_score_key_pairs,
                            bootstrap=True,
                        )

    # Convert bootstrap lists to numpy arrays for easier downstream analysis
    for outcome_storage in outcomes_results.values():
        _finalize_bootstrap_storage(outcome_storage['overall']['bootstrap'])
        if 'subgroups' in outcome_storage:
            for category_groups in outcome_storage['subgroups'].values():
                for subgroup_bundle in category_groups.values():
                    _finalize_bootstrap_storage(subgroup_bundle['bootstrap'])

    bootstrap_replication_results = {
        'outcomes': outcomes_results,
        'category_levels': category_levels,
        'n_bootstraps': n_bootstraps,
        'bootstrap_indices': bootstrap_indices,
    }

    return bootstrap_replication_results





def analyze_bootstrap_results(bootstrapped_results, metric_func_name,
                              y_score_names=None,
                              confidence_level=0.95, alternative='two-sided',
                              method='basic',
                              outcome_name=None,
                              category_name=None,
                              category_value=None):
    """
    Analyze bootstrapped results for a specific metric, including confidence intervals and replication indices.

    This function calculates the confidence intervals for a given metric function based on bootstrapped results, and identifies
    the indices of replications within these intervals. It also provides a summary of metric and curve results within the confidence
    interval.

    Parameters:
    ----------
    bootstrapped_results : dict
        Results from ``bootstrap_evaluation``, including original and bootstrapped metrics and curves.
    metric_func_name : str
        Name of the metric function to analyze.
    y_score_names : list of str, optional
        Specific score names to analyze. If ``None``, all scores are analyzed.
    confidence_level : float, default=0.95
        Confidence level for the interval calculation.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Specifies the alternative hypothesis for interval calculation.
    method : {'percentile', 'basic'}, default='percentile'
        Method for confidence interval calculation.
    outcome_name : str, optional
        Name of the outcome to analyze. If ``None`` and only one outcome is present, it is selected automatically.
    category_name : str, optional
        Name of the category for subgroup analysis. Requires ``category_value``.
    category_value : str or numeric, optional
        Specific value within ``category_name`` to analyze.

    Returns:
    -------
    dict
        A dictionary with keys as y_score names, each containing a tuple of confidence interval, indices of replications within
        the interval, and a dictionary of corresponding bootstrapped metric and curve results.
    """

    # Calculate percentile interval
    alpha = ((1 - confidence_level) / 2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        # TODO
        raise ValueError("method='bca' not implemented")
    else:
        interval = alpha, 1 - alpha

        def percentile_func(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    container = _resolve_outcome_container(
        bootstrapped_results,
        outcome_name=outcome_name,
        category_name=category_name,
        category_value=category_value,
    )

    original_metrics = container['original']['metrics']
    bootstrap_metrics = container['bootstrap']['metrics']
    bootstrap_curves = container['bootstrap']['curves']

    if y_score_names is None:
        y_score_names = original_metrics[metric_func_name].keys()

    ci_results = {k: None for k in y_score_names}

    for y_score_key in y_score_names:
        theta_hat_b = bootstrap_metrics[metric_func_name][y_score_key]
        theta_hat = original_metrics[metric_func_name][y_score_key]

        # Calculate confidence interval of statistic
        ci_l = percentile_func(theta_hat_b, interval[0] * 100)
        ci_u = percentile_func(theta_hat_b, interval[1] * 100)
        if method == 'basic':
            ci_l, ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

        if alternative == 'less':
            ci_l = np.full_like(ci_l, -np.inf)
        elif alternative == 'greater':
            ci_u = np.full_like(ci_u, np.inf)

        ci = (ci_l, ci_u)

        is_within_bounds = (ci_l <= theta_hat_b) & (theta_hat_b <= ci_u)
        ci_replication_indices = np.where(is_within_bounds)[0]

        ci_bootstrapped_results = {
            'metrics': {},
            'curves': {}
        }

        for metric_name, replications in bootstrap_metrics.items():
            ci_bootstrapped_results['metrics'][metric_name] = replications[y_score_key][ci_replication_indices]

        for curve_name, replications in bootstrap_curves.items():
            ci_bootstrapped_results['curves'][curve_name] = [replications[y_score_key][i] for i in ci_replication_indices]

        ci_results[y_score_key] = (ci, ci_replication_indices, ci_bootstrapped_results)

    return ci_results


def summarize_bootstrap_results(bootstrapped_results, confidence_level=0.95, alternative='two-sided', method='basic', decimal_places=3,
                               outcome_name=None, category_name=None, category_value=None):
    """
    Summarizes the bootstrapped results, providing central values and confidence intervals for metrics.

    This function processes the results from ``bootstrap_evaluation`` to provide a concise summary of metrics and compatibility metrics,
    including their confidence intervals.

    Parameters:
    ----------
    bootstrapped_results : dict
        Results from ``bootstrap_evaluation``.
    confidence_level : float, default=0.95
        Confidence level for interval calculations.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Specifies the alternative hypothesis for interval calculation.
    method : {'percentile', 'basic'}, default='percentile'
        Method for confidence interval calculation.
    decimal_places : int, default=3
        Number of decimal places for rounding the results.
    outcome_name : str, optional
        Name of the outcome to summarize. If ``None`` and only one outcome is present, it is selected automatically.
    category_name : str, optional
        Name of the category for subgroup analysis. Requires ``category_value``.
    category_value : str or numeric, optional
        Specific value within ``category_name`` to summarize.

    Returns:
    -------
    tuple
        Two dictionaries containing summarized results for metrics and compatibility metrics.
    """

    # Calculate percentile interval
    alpha = ((1 - confidence_level) / 2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        # TODO
        raise ValueError("method='bca' not implemented")
    else:
        interval = alpha, 1 - alpha

        def percentile_func(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    container = _resolve_outcome_container(
        bootstrapped_results,
        outcome_name=outcome_name,
        category_name=category_name,
        category_value=category_value,
    )

    original_metrics = container['original']['metrics']
    bootstrap_metrics = container['bootstrap']['metrics']
    original_compatibility = container['original']['compatibility_metrics']
    bootstrap_compatibility = container['bootstrap']['compatibility_metrics']

    mf_summary_results = {}
    for metric_func_name, originals in original_metrics.items():
        mf_summary_results[metric_func_name] = {}

        for y_score_key, theta_hat in originals.items():
            theta_hat_b = bootstrap_metrics[metric_func_name][y_score_key]

            ci_l = percentile_func(theta_hat_b, interval[0] * 100)
            ci_u = percentile_func(theta_hat_b, interval[1] * 100)
            if method == 'basic':
                ci_l, ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

            if alternative == 'less':
                ci_l = np.full_like(ci_l, -np.inf)
            elif alternative == 'greater':
                ci_u = np.full_like(ci_u, np.inf)

            ci = (np.round(ci_l, decimal_places), np.round(ci_u, decimal_places))
            center = np.round(theta_hat, decimal_places)
            mf_summary_results[metric_func_name][y_score_key] = f"{center} {ci}"

    cmf_summary_results = {}
    for metric_func_name, originals in original_compatibility.items():
        cmf_summary_results[metric_func_name] = {}

        for y_score_pair_key, theta_hat in originals.items():
            theta_hat_b = bootstrap_compatibility[metric_func_name][y_score_pair_key]

            ci_l = percentile_func(theta_hat_b, interval[0] * 100)
            ci_u = percentile_func(theta_hat_b, interval[1] * 100)
            if method == 'basic':
                ci_l, ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

            if alternative == 'less':
                ci_l = np.full_like(ci_l, -np.inf)
            elif alternative == 'greater':
                ci_u = np.full_like(ci_u, np.inf)

            ci = (np.round(ci_l, decimal_places), np.round(ci_u, decimal_places))
            center = np.round(theta_hat, decimal_places)
            cmf_summary_results[metric_func_name][y_score_pair_key] = f"{center} {ci}"

    return mf_summary_results, cmf_summary_results

def plot_bootstrap_curve(bootstrapped_results, metric_func_name, curve_func_name,
                         y_score_names=None,
                         confidence_level=0.95, alternative='two-sided', method='basic',
                         xlabel='', ylabel='', title=None, legend_title=None, legend_title_CI_flag=True,
                         rep_line_alpha=0.01, line_alpha=1,
                         show_plot=True, figsize=(8, 8),
                         outcome_name=None, category_name=None, category_value=None):
    """
    Plots curves from bootstrapped data, highlighting the original curve and confidence intervals.

    This function visualizes the variability and uncertainty in the model's performance metrics and curves using bootstrapped data.
    It's useful for comparing different models or methodologies.

    Parameters:
    ----------
    bootstrapped_results : dict
        Results from ``bootstrap_evaluation``.
    metric_func_name : str
        The metric function name for CI analysis.
    curve_func_name : str
        The curve function to be plotted.
    y_score_names : list, optional
        Names of score arrays to consider. If ``None``, all are considered.
    confidence_level : float, default=0.95
        Confidence level for CI.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis for CI.
    method : {'percentile', 'basic', 'BCa'}, default='percentile'
        Method for CI calculation.
    xlabel, ylabel : str
        Labels for X and Y axes.
    title : str, optional
        Title of the plot.
    legend_title : str, optional
        Title for the legend.
    legend_title_CI_flag : bool, default=True
        Include CI in legend title.
    rep_line_alpha, line_alpha : float
        Alpha values for replicated and original lines.
    show_plot : bool, default=True
        Show plot if True.
    figsize : tuple, default=(8, 8)
        Size of the figure.
    outcome_name : str, optional
        Name of the outcome to plot. If ``None`` and only one outcome is present, it is selected automatically.
    category_name : str, optional
        Name of the category for subgroup analysis. Requires ``category_value``.
    category_value : str or numeric, optional
        Specific value within ``category_name`` to plot.

    Returns:
    -------
    (fig, ax)
        Matplotlib figure and axes objects.
    """

    container = _resolve_outcome_container(
        bootstrapped_results,
        outcome_name=outcome_name,
        category_name=category_name,
        category_value=category_value,
    )

    original_metrics = container['original']['metrics']
    original_curves = container['original']['curves']

    if y_score_names is None:
        y_score_names = original_metrics[metric_func_name].keys()

    title = title or curve_func_name
    legend_title = legend_title or metric_func_name
    if legend_title_CI_flag:
        confidence_level_int = int(confidence_level * 100)
        legend_title = f"{legend_title} ({confidence_level_int}% CI)"

    ci_results = analyze_bootstrap_results(
        bootstrapped_results,
        metric_func_name,
        y_score_names=y_score_names,
        confidence_level=confidence_level,
        alternative=alternative,
        method=method,
        outcome_name=outcome_name,
        category_name=category_name,
        category_value=category_value,
    )

    fig, ax = plt.subplots(figsize=figsize)

    color_map = plt.cm.get_cmap('tab10', len(y_score_names))

    for i, y_score_key in enumerate(y_score_names):
        ci, ci_replication_indices, ci_bootstrapped_results = ci_results[y_score_key]
        color = color_map(i % 10)
        light_color = _lighten_color(color, amount=0.9)

        for curve in ci_bootstrapped_results['curves'][curve_func_name]:
            if curve is None:
                continue
            x, y, _ = curve
            ax.plot(x, y, color=light_color, alpha=rep_line_alpha, zorder=1)

        center = original_metrics[metric_func_name][y_score_key]
        label = f"{y_score_key}: {center:.2f} ({ci[0]:.2f}, {ci[1]:.2f})"

        curve_data = original_curves[curve_func_name][y_score_key]
        if curve_data is not None:
            x, y, _ = curve_data
            ax.plot(x, y, color=color, alpha=line_alpha, label=label, zorder=2)

    ax.legend(title=legend_title)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_plot:
        plt.show()

    return fig, ax
__all__ = [
    'bootstrap_evaluation'
]
