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




def bootstrap_evaluation(y_true, y_scores, 
                        metric_funcs, metric_funcs_kwargs=None,
                        curve_funcs=None, curve_funcs_kwargs=None,
                         compatibility_metric_funcs=None, compatibility_metric_funcs_kwargs=None,
                        n_bootstraps=1000, random_state=None, 
                       ):
    """
    Perform bootstrapping for machine learning metric and curve evaluations.

    This function generates bootstrapped samples for various metrics, curves, and compatibility measures, providing insights into model performance.

    Parameters:
    ----------
    y_true : array-like
        True labels or binary label indicators.
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

    Returns:
    -------
    dict
        A dictionary containing original and bootstrapped metric results, curve results, and compatibility metrics.

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
    y_true, y_scores = _validate_ys(y_true, y_scores)
          
    # Initialize random number generator
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    # Convert function lists/dicts to standardized dict format
    metric_func_dict = _get_funcs_dict(metric_funcs, 'metric_funcs')
    curve_func_dict = _get_funcs_dict(curve_funcs, 'curve_funcs')
    compatibility_metric_func_dict = _get_funcs_dict(compatibility_metric_funcs, 'compatibility_metric_funcs')

   # Prepare kwargs for each type of function
    metric_kwarg_dict = _get_funcs_kwargs_dict(metric_func_dict, metric_funcs_kwargs, 'metric_funcs_kwargs')
    curve_kwarg_dict = _get_funcs_kwargs_dict(curve_func_dict, curve_funcs_kwargs, 'curve_funcs_kwargs')
    compatibility_metric_kwarg_dict = _get_funcs_kwargs_dict(compatibility_metric_func_dict, compatibility_metric_funcs_kwargs, 'compatibility_metric_funcs_kwargs')
    
    # Compute original metric and curve results
    score_storage = {key: None for key, _ in y_scores.items()}
    original_metric_results = {key: copy.deepcopy(score_storage) for key, _ in metric_func_dict.items()}
    original_curve_results = {key: copy.deepcopy(score_storage) for key, _ in curve_func_dict.items()}
    
    y_score_key_pairs = [_ for _ in permutations(y_scores.keys(), 2)]
    score_pair_storage = {key: None for key in y_score_key_pairs}
    original_compatibility_metric_results = {key: copy.deepcopy(score_pair_storage) for key,_ in compatibility_metric_func_dict.items()}

    # Get values of metrics, curves, and compatility_metrics on original data
    for y_score_key, y_score in y_scores.items():
        for metric_func_name, metric_func in metric_func_dict.items():
            original_metric_results[metric_func_name][y_score_key] = metric_func(y_true, y_score, **metric_kwarg_dict[metric_func_name])
    
        for curve_func_name, curve_func in curve_func_dict.items():
            original_curve_results[curve_func_name][y_score_key] = curve_func(y_true, y_score, **curve_kwarg_dict[curve_func_name])
            
    for cmf_name, compatibility_metric_func in compatibility_metric_func_dict.items():
         for y_score_key_pair in y_score_key_pairs:
             y_score_original = y_scores[y_score_key_pair[0]]
             y_score_updated = y_scores[y_score_key_pair[1]]
             original_compatibility_metric_results[cmf_name][y_score_key_pair] = \
                compatibility_metric_func(y_true, y_score_original, y_score_updated,
                **compatibility_metric_kwarg_dict[cmf_name])
                 

    # Initialize storage for bootstrapped results
    score_storage = {key: [] for key, _ in y_scores.items()}
    metric_results = {key: copy.deepcopy(score_storage) for key,_ in metric_func_dict.items()}
    curve_results = {key: copy.deepcopy(score_storage) for key,_ in curve_func_dict.items()}

    y_score_key_pairs = [_ for _ in permutations(y_scores.keys(), 2)]
    score_pair_storage = {key: [] for key in y_score_key_pairs}                     
    compatibility_metric_results = {key: copy.deepcopy(score_pair_storage) for key,_ in compatibility_metric_func_dict.items()}

    # Generate bootstrapped samples and calculate metrics
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        r_y_true = y_true[indices]
        r_y_scores = {k: v[indices] for k,v in y_scores.items()}

        for y_score_key, r_y_score in r_y_scores.items():
            # Process metric functions
            for mf_name, metric_func in metric_func_dict.items():
                metric_result = metric_func(r_y_true, r_y_score, **metric_kwarg_dict[mf_name])
                metric_results[mf_name][y_score_key].append(metric_result)
    
            # Process curve functions
            for cf_name, curve_func in curve_func_dict.items():
                curve_result = curve_func(r_y_true, r_y_score, **curve_kwarg_dict[cf_name])
                curve_results[cf_name][y_score_key].append(curve_result)

        for cmf_name, compatibility_metric_func in compatibility_metric_func_dict.items():
             for y_score_key_pair in y_score_key_pairs:
                 r_y_score_original = r_y_scores[y_score_key_pair[0]]
                 r_y_score_updated = r_y_scores[y_score_key_pair[1]]
                 compatibility_metric_result = compatibility_metric_func(r_y_true, r_y_score_original, r_y_score_updated, **compatibility_metric_kwarg_dict[cmf_name])
                 compatibility_metric_results[cmf_name][y_score_key_pair].append(compatibility_metric_result)


    def bootstrap_metrics_results_list_to_array(bootstrap_metrics_results):
        new_dict = {}
        for m_k, m_v in bootstrap_metrics_results.items():
            new_dict[m_k] = {}
            for s_k, s_v in m_v.items():
                new_dict[m_k][s_k] = np.array(s_v)
        return new_dict    

    # Prepare final results
    bootstrap_replication_results = {
        'original_metrics': original_metric_results,
        'original_curves': original_curve_results,
        'original_compatibility_metrics': original_compatibility_metric_results,
        'bootstrap_replication_metrics': bootstrap_metrics_results_list_to_array(metric_results),
        'bootstrap_replication_curves': curve_results,
        'bootstrap_compatibility_metrics': bootstrap_metrics_results_list_to_array(compatibility_metric_results)
        
    }

    return bootstrap_replication_results


def analyze_bootstrap_results(bootstrapped_results, metric_func_name,
                              y_score_names=None,
                              confidence_level=0.95, alternative='two-sided',
                              method='basic'):
    """
    Analyze bootstrapped results for a specific metric, including confidence intervals and replication indices.

    This function calculates the confidence intervals for a given metric function based on bootstrapped results, and identifies the indices of replications within these intervals. It also provides a summary of metric and curve results within the confidence interval.

    Parameters:
    ----------
    bootstrapped_results : dict
        Results from bootstrap_evaluation, including original and bootstrapped metrics and curves.
    metric_func_name : str
        Name of the metric function to analyze.
    y_score_names : list of str, optional
        Specific score names to analyze. If None, all scores are analyzed.
    confidence_level : float, default=0.95
        Confidence level for the interval calculation.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Specifies the alternative hypothesis for interval calculation.
    method : {'percentile', 'basic'}, default='percentile'
        Method for confidence interval calculation.

    Returns:
    -------
    dict
        A dictionary with keys as y_score names, each containing a tuple of confidence interval, indices of replications within the interval, and a dictionary of corresponding bootstrapped metric and curve results.

    Examples:
    --------
    >>> results = bootstrap_evaluation(...)
    >>> analyzed_results = analyze_bootstrap_results(results, 'my_metric')
    >>> print(analyzed_results)

    Notes:
    -----
    The method 'bca' for confidence interval calculation is not implemented yet.
    """
    
    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        # TODO
        raise ValueError(f"method='bca' not implemented")
    else:
        interval = alpha, 1-alpha

        def percentile_func(a, q):
            return np.percentile(a=a, q=q, axis=-1)
            
    if y_score_names is None:
        y_score_names = bootstrapped_results['original_metrics'][metric_func_name].keys()
    
    ci_results = {k: None for k in y_score_names}

    for y_score_key in y_score_names:
        theta_hat_b = bootstrapped_results['bootstrap_replication_metrics'][metric_func_name][y_score_key]
        theta_hat = bootstrapped_results['original_metrics'][metric_func_name][y_score_key]

        # Calculate confidence interval of statistic
        ci_l = percentile_func(theta_hat_b, interval[0]*100)
        ci_u = percentile_func(theta_hat_b, interval[1]*100)
        if method == 'basic':
            ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

        if alternative == 'less':
            ci_l = np.full_like(ci_l, -np.inf)
        elif alternative == 'greater':
            ci_u = np.full_like(ci_u, np.inf)

        # Set CI tuple
        ci = (ci_l, ci_u)

        # Find indices of replications within CI
        # Create a boolean array indicating whether each replication is within CI bounds
        is_within_bounds = (ci_l <= theta_hat_b) & (theta_hat_b <= ci_u)

        # Find indices where the condition is True
        ci_replication_indices = np.where(is_within_bounds)[0]
        
        ci_bootstrapped_results = {
            'metrics': {},
            'curves': {}
        }
        
        # Loop over metrics
        for metric_name, replications in bootstrapped_results['bootstrap_replication_metrics'].items():
            ci_bootstrapped_results['metrics'][metric_name] = replications[y_score_key][ci_replication_indices]
        
        # Loop over curves
        for curve_name, replications in bootstrapped_results['bootstrap_replication_curves'].items():
            ci_bootstrapped_results['curves'][curve_name] = [replications[y_score_key][i] for i in ci_replication_indices]
            
        ci_results[y_score_key] = (ci, ci_replication_indices, ci_bootstrapped_results)
        print()
    
    return ci_results


def summarize_bootstrap_results(bootstrapped_results, confidence_level=0.95, alternative='two-sided', method='basic', decimal_places=3):
    """
    Summarizes the bootstrapped results, providing central values and confidence intervals for metrics.

    This function processes the results from bootstrap_evaluation to provide a concise summary of metrics and compatibility metrics, including their confidence intervals.

    Parameters:
    ----------
    bootstrapped_results : dict
        Results from bootstrap_evaluation.
    confidence_level : float, default=0.95
        Confidence level for interval calculations.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Specifies the alternative hypothesis for interval calculation.
    method : {'percentile', 'basic'}, default='percentile'
        Method for confidence interval calculation.
    decimal_places : int, default=3
        Number of decimal places for rounding the results.

    Returns:
    -------
    tuple
        Two dictionaries containing summarized results for metrics and compatibility metrics.

    Examples:
    --------
    >>> results = bootstrap_evaluation(...)
    >>> summarized_results = summarize_bootstrap_results(results)
    >>> print(summarized_results)

    Notes:
    -----
    The 'bca' method for confidence interval calculation is not implemented in this function.
    """
    
    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        # TODO
        raise ValueError(f"method='bca' not implemented")
    else:
        interval = alpha, 1-alpha

        def percentile_func(a, q):
            return np.percentile(a=a, q=q, axis=-1)
    
    mf_summary_results = {}
    for metric_func_name, bsr_om_mfn in bootstrapped_results['original_metrics'].items():
        mf_summary_results[metric_func_name] = {}
        
        for y_score_key, theta_hat in bsr_om_mfn.items():
            theta_hat_b = bootstrapped_results['bootstrap_replication_metrics'][metric_func_name][y_score_key]
            
            # Calculate confidence interval of statistic
            ci_l = percentile_func(theta_hat_b, interval[0]*100)
            ci_u = percentile_func(theta_hat_b, interval[1]*100)
            if method == 'basic':
                ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

            if alternative == 'less':
                ci_l = np.full_like(ci_l, -np.inf)
            elif alternative == 'greater':
                ci_u = np.full_like(ci_u, np.inf)

            # Set CI tuple
            ci = (np.round(ci_l, decimal_places), np.round(ci_u, decimal_places))
            center = np.round(theta_hat, decimal_places)
            mf_summary_results[metric_func_name][y_score_key] = f"{center} {ci}"
            
            
    cmf_summary_results = {}
    for metric_func_name, bsr_ocm_mfn in bootstrapped_results['original_compatibility_metrics'].items():
        cmf_summary_results[metric_func_name] = {}
        
        for y_score_pair_key, theta_hat in bsr_ocm_mfn.items():
            theta_hat_b = bootstrapped_results['bootstrap_compatibility_metrics'][metric_func_name][y_score_pair_key]
            
            # Calculate confidence interval of statistic
            ci_l = percentile_func(theta_hat_b, interval[0]*100)
            ci_u = percentile_func(theta_hat_b, interval[1]*100)
            if method == 'basic':
                ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

            if alternative == 'less':
                ci_l = np.full_like(ci_l, -np.inf)
            elif alternative == 'greater':
                ci_u = np.full_like(ci_u, np.inf)

            # Set CI tuple
            ci = (np.round(ci_l, decimal_places), np.round(ci_u, decimal_places))
            center = np.round(theta_hat, decimal_places)
            cmf_summary_results[metric_func_name][y_score_pair_key] = f"{center} {ci}"
    
    return mf_summary_results, cmf_summary_results



def plot_bootstrap_curve(bootstrapped_results, metric_func_name, curve_func_name,
                         y_score_names=None,
                         confidence_level=0.95, alternative='two-sided', method='basic',
                         xlabel='', ylabel='', title=None, legend_title=None, legend_title_CI_flag=True,
                         rep_line_alpha=0.01, line_alpha=1,
                         show_plot=True, figsize=(8,8)):
    """
    Plots curves from bootstrapped data, highlighting the original curve and confidence intervals.

    This function visualizes the variability and uncertainty in the model's performance metrics and curves using bootstrapped data. It's useful for comparing different models or methodologies.

    Parameters:
    ----------
    - bootstrapped_results : dict
        Results from bootstrap_evaluation.
    - metric_func_name : str
        The metric function name for CI analysis.
    - curve_func_name : str
        The curve function to be plotted.
    - y_score_names : list, optional
        Names of score arrays to consider. If None, all are considered.
    - confidence_level : float, default=0.95
        Confidence level for CI.
    - alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis for CI.
    - method : {'percentile', 'basic', 'BCa'}, default='percentile'
        Method for CI calculation.
    - xlabel, ylabel : str
        Labels for X and Y axes.
    - title : str, optional
        Title of the plot.
    - legend_title : str, optional
        Title for the legend.
    - legend_title_CI_flag : bool, default=True
        Include CI in legend title.
    - rep_line_alpha, line_alpha : float
        Alpha values for replicated and original lines.
    - show_plot : bool, default=True
        Show plot if True.
    - figsize : tuple, default=(8,8)
        Size of the figure.

    Returns:
    -------
    - fig, ax : Matplotlib figure and axes objects

    Example Usage:
    --------------
    >>> bootstrapped_results = bootstrap_evaluation(...)
    >>> _ = plot_bootstrap_curve(bootstrapped_results, 'roc_auc_score', 'roc_curve',
                                 xlabel='False Positive Rate', ylabel='True Positive Rate'
                                 title='ROC Curve', legend_title='AUROC')
    """
    
    if y_score_names is None:
        y_score_names = bootstrapped_results['original_metrics'][metric_func_name].keys()
        
    title = title or curve_func_name
    legend_title = legend_title or metric_func_name
    if legend_title_CI_flag:
        confidence_level_int = int(confidence_level*100)
        legend_title = f"{legend_title} ({confidence_level_int}% CI)"
    
    ci_results = analyze_bootstrap_results(bootstrapped_results,
        metric_func_name, y_score_names=y_score_names, confidence_level=confidence_level,
        alternative=alternative, method=method)
        
    # Set up a square figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generating a color map
    color_map = plt.cm.get_cmap('tab10', len(y_score_names))

    for i, y_score_key in enumerate(y_score_names):
        ci, ci_replication_indices, ci_bootstrapped_results = ci_results[y_score_key]
        color = color_map(i%10)  # Get unique color for each key
        light_color = _lighten_color(color, amount=0.9)  # Lighten the color
        
        #print(y_score_key, ci, len(ci_bootstrapped_results))
        for curve in ci_bootstrapped_results['curves'][curve_func_name]:
            x, y, _ = curve
            
            ax.plot(x, y, color=light_color, alpha=rep_line_alpha, zorder=1)

        center = bootstrapped_results['original_metrics'][metric_func_name][y_score_key]
        label = f"{y_score_key}: {center:.2f} ({ci[0]:.2f}, {ci[1]:.2f})"
            
        x, y, _ = bootstrapped_results['original_curves'][curve_func_name][y_score_key]
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
