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

from .utils import (
    _get_funcs_dict, _get_funcs_kwargs_dict, _validate_ys, _lighten_color,
    ensure_dict, _to_np, _check_lengths,
    resolve_groups, stable_int_hash, make_bootstrap_indices
)
import copy
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np


def _coerce_y_inputs(y_true, y_scores):
    """
    Coerce inputs to internal formats while preserving back-compat:
      - y_true can be a single array OR a dict[label_name -> array]
      - y_scores can be a single array, list of arrays, or dict[model_name -> array]
    Returns:
      labels_dict : dict[label_name -> ndarray]
      scores_dict : dict[model_name -> ndarray]
      N           : int (length)
    """
    # Multi-label case
    if isinstance(y_true, dict):
        labels_dict = {str(k): _to_np(v) for k, v in y_true.items()}
        if len(labels_dict) == 0:
            raise ValueError("y_true dict is empty.")
        # Use one label to coerce/validate y_scores via existing helper
        first_label_name = next(iter(labels_dict))
        y_true_first = labels_dict[first_label_name]
        _, scores_dict = _validate_ys(y_true_first, y_scores)
        # Ensure all labels and scores share same length
        _check_lengths(*labels_dict.values(), *scores_dict.values())
        N = len(y_true_first)
        return labels_dict, scores_dict, N

    # Single-label case (back-compat path)
    y_true_arr, scores_dict = _validate_ys(y_true, y_scores)
    labels_dict = {"label_0": y_true_arr}
    N = len(y_true_arr)
    return labels_dict, scores_dict, N


def _get_model_pairs(model_names,
                     compatibility_compare="all_pairs",
                     compatibility_baseline_model=None,
                     compatibility_pair_filter=None):
    """
    Determine which model pairs to evaluate for compatibility metrics.
    """
    names = list(model_names)
    if compatibility_pair_filter is not None and len(compatibility_pair_filter) > 0:
        # Only keep pairs whose members exist
        return [(a, b) for (a, b) in compatibility_pair_filter if (a in names and b in names)]

    if compatibility_compare == "baseline_vs_each":
        base = compatibility_baseline_model or (names[0] if names else None)
        if base is None:
            return []
        return [(base, n) for n in names if n != base]

    if compatibility_compare == "ordered_pairs":
        # (m0,m1), (m1,m2), ...
        return [(names[i], names[i+1]) for i in range(len(names) - 1)]

    # Default: all_pairs (unordered; keep input order)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


def _eval_original(y_true, y_scores_dict,
                   metric_func_dict, metric_kwarg_dict,
                   curve_func_dict, curve_kwarg_dict,
                   compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                   compatibility_pairs):
    """
    Compute original (non-bootstrapped) metrics, curves, and compatibility metrics for a node.
    Returns dictionaries matching the package's existing key pattern.
    """
    original_metric_results = {k: {m: None for m in y_scores_dict.keys()} for k in metric_func_dict.keys()}
    original_curve_results = {k: {m: None for m in y_scores_dict.keys()} for k in curve_func_dict.keys()}

    for y_score_key, y_score in y_scores_dict.items():
        for metric_func_name, metric_func in metric_func_dict.items():
            original_metric_results[metric_func_name][y_score_key] = metric_func(
                y_true, y_score, **(metric_kwarg_dict.get(metric_func_name, {}))
            )
        for curve_func_name, curve_func in curve_func_dict.items():
            original_curve_results[curve_func_name][y_score_key] = curve_func(
                y_true, y_score, **(curve_kwarg_dict.get(curve_func_name, {}))
            )

    # compatibility metrics on original data
    original_compatibility_metric_results = {k: {} for k in compatibility_metric_func_dict.keys()}
    if len(compatibility_metric_func_dict) > 0 and len(compatibility_pairs) > 0:
        for cmf_name, compatibility_metric_func in compatibility_metric_func_dict.items():
            for (m_a, m_b) in compatibility_pairs:
                y_score_a = y_scores_dict[m_a]
                y_score_b = y_scores_dict[m_b]
                original_compatibility_metric_results[cmf_name][(m_a, m_b)] = compatibility_metric_func(
                    y_true, y_score_a, y_score_b,
                    **(compatibility_metric_kwarg_dict.get(cmf_name, {}))
                )

    return original_metric_results, original_curve_results, original_compatibility_metric_results


def _eval_bootstrap(y_true, y_scores_dict, indices,
                    metric_func_dict, metric_kwarg_dict,
                    curve_func_dict, curve_kwarg_dict,
                    compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                    compatibility_pairs):
    """
    Compute bootstrapped metrics/curves/compatibility using a precomputed indices array of shape (B, M).
    """
    B = indices.shape[0]
    # Initialize storage for bootstrapped results
    score_storage = {key: [] for key in y_scores_dict.keys()}
    metric_results = {key: copy.deepcopy(score_storage) for key in metric_func_dict.keys()}
    curve_results = {key: copy.deepcopy(score_storage) for key in curve_func_dict.keys()}

    score_pair_storage = {pair: [] for pair in compatibility_pairs}
    compatibility_metric_results = {key: copy.deepcopy(score_pair_storage) for key in compatibility_metric_func_dict.keys()}

    # Generate bootstrapped samples and calculate metrics
    for b in range(B):
        sel = indices[b]
        r_y_true = y_true[sel]
        r_y_scores = {k: v[sel] for k, v in y_scores_dict.items()}

        for y_score_key, r_y_score in r_y_scores.items():
            # Process metric functions
            for mf_name, metric_func in metric_func_dict.items():
                metric_result = metric_func(r_y_true, r_y_score, **(metric_kwarg_dict.get(mf_name, {})))
                metric_results[mf_name][y_score_key].append(metric_result)

            # Process curve functions
            for cf_name, curve_func in curve_func_dict.items():
                curve_result = curve_func(r_y_true, r_y_score, **(curve_kwarg_dict.get(cf_name, {})))
                curve_results[cf_name][y_score_key].append(curve_result)

        for cmf_name, compatibility_metric_func in compatibility_metric_func_dict.items():
            for (m_a, m_b) in compatibility_pairs:
                r_y_score_a = r_y_scores[m_a]
                r_y_score_b = r_y_scores[m_b]
                cm_val = compatibility_metric_func(
                    r_y_true, r_y_score_a, r_y_score_b,
                    **(compatibility_metric_kwarg_dict.get(cmf_name, {}))
                )
                compatibility_metric_results[cmf_name][(m_a, m_b)].append(cm_val)

    # Convert lists -> arrays for metric/compat results; leave curves as lists
    def _metrics_list_to_array(bootstrap_metrics_results):
        new_dict = {}
        for m_k, m_v in bootstrap_metrics_results.items():
            new_dict[m_k] = {}
            for s_k, s_v in m_v.items():
                new_dict[m_k][s_k] = np.array(s_v)
        return new_dict

    boot_metrics = _metrics_list_to_array(metric_results)
    boot_curves = curve_results
    boot_compat = _metrics_list_to_array(compatibility_metric_results)

    return boot_metrics, boot_curves, boot_compat


def _build_node(y_true, y_scores_dict,
                metric_func_dict, metric_kwarg_dict,
                curve_func_dict, curve_kwarg_dict,
                compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                compatibility_pairs,
                n_bootstraps, rng, store_indices=False, sample_size=None):
    """
    Build a single node (population slice) with original + bootstrap results.
    Returns the node dict and a small metadata dict.
    """
    # ORIGINAL
    original_metric_results, original_curve_results, original_compat_results = _eval_original(
        y_true, y_scores_dict,
        metric_func_dict, metric_kwarg_dict,
        curve_func_dict, curve_kwarg_dict,
        compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
        compatibility_pairs
    )

    # BOOTSTRAP indices (deterministic for this node via rng)
    N_here = len(y_true)
    idx = make_bootstrap_indices(N_here, n_bootstraps, rng, sample_size=sample_size)

    # BOOTSTRAP evaluations
    boot_metrics, boot_curves, boot_compat = _eval_bootstrap(
        y_true, y_scores_dict, idx,
        metric_func_dict, metric_kwarg_dict,
        curve_func_dict, curve_kwarg_dict,
        compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
        compatibility_pairs
    )

    # Package with your existing key names
    node = {
        'original_metrics': original_metric_results,
        'original_curves': original_curve_results,
        'original_compatibility_metrics': original_compat_results,
        'bootstrap_replication_metrics': boot_metrics,
        'bootstrap_replication_curves': boot_curves,
        'bootstrap_compatibility_metrics': boot_compat
    }

    meta = {'n': int(N_here), 'n_bootstraps': int(n_bootstraps)}
    if store_indices:
        meta['bootstrap_indices'] = idx
    return node, meta


def bootstrap_evaluation(
        y_true, y_scores,
        metric_funcs, metric_funcs_kwargs=None,
        curve_funcs=None, curve_funcs_kwargs=None,
        compatibility_metric_funcs=None, compatibility_metric_funcs_kwargs=None,
        # OPTIONAL: label↔label metrics (for multi-label y_true only)
        label_metrics=None, label_metrics_kwargs=None,
        n_bootstraps=1000, random_state=None,
        # NEW: grouping
        group_by=None, group_binning=None,
        missing_group_name="Missing",
        max_categorical_cardinality=100,
        # NEW: small-group policy
        min_group_size=0, small_group="compute_flag", small_group_flag_key="low_n",
        # NEW: compatibility controls (prefixed; default compare = all_pairs)
        compatibility_compare="all_pairs",
        compatibility_baseline_model=None,
        compatibility_pair_filter=None,
        # NEW: persistence of indices
        store_indices=False
    ):
    """
    Perform bootstrapping for model metrics, curves, and (optionally) compatibility
    and label↔label metrics, with optional subgrouping.

    This function generates bootstrap samples per node (overall and per subgroup),
    evaluates per-model metrics/curves, and (optionally) model-pair compatibility
    and label↔label metrics. For multi-label input (``y_true`` as a dict), each
    label gets its own node; bootstrap indices are aligned across labels within
    a node so cross-label comparisons are coherent.

    Parameters
    ----------
    y_true : array-like or dict[str, array-like]
        True labels (single label) or a mapping from label name → label array
        (multi-label). All arrays must share the same length N.
    y_scores : array-like, sequence of arrays, or dict[str, array-like]
        Model scores. If a dict, keys are model names; if a single array or a
        list/tuple, models are auto-named.
    metric_funcs : dict[str, callable] or callable
        Per-model metric functions of the form ``f(y_true, y_score, **kwargs) → float``.
        If a single callable is provided, it is applied under its own name.
    metric_funcs_kwargs : dict[str, dict], optional
        Per-metric kwargs: ``{metric_name: {k: v}}``.
    curve_funcs : dict[str, callable] or callable, optional
        Curve-generators of the form ``g(y_true, y_score, **kwargs) → (x, y, *meta)``.
        Must return at least ``(x, y)`` arrays of equal length.
    curve_funcs_kwargs : dict[str, dict], optional
        Per-curve kwargs.
    compatibility_metric_funcs : dict[str, callable] or callable, optional
        Functions comparing two model score arrays:
        ``h(y_true, y_score_a, y_score_b, **kwargs) → float``.
    compatibility_metric_funcs_kwargs : dict[str, dict], optional
        Per-compatibility-metric kwargs.
    label_metrics : dict[str, callable] or callable, optional
        **Multi-label only.** Functions of two label arrays:
        ``L(y_label_A, y_label_B, **kwargs) → float``. Computed for all ordered
        label pairs (A!=B) within each node.
    label_metrics_kwargs : dict[str, dict], optional
        Per-label-metric kwargs.
    n_bootstraps : int, default=1000
        Number of bootstrap replications per node.
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.
    group_by : dict[str, array-like], optional
        Subgroup columns (categorical or numeric). Numeric columns are binned
        per ``group_binning`` (default: quantile q=5).
    group_binning : dict[str, dict], optional
        Per-column binning config, e.g.:
        ``{'age': {'strategy': 'quantile', 'q': 5}}``,
        ``{'age': {'strategy': 'uniform', 'bins': 10}}``,
        or ``{'age': {'strategy': 'custom', 'edges': [...], 'labels': [...]}}``.
    missing_group_name : str, default "Missing"
        Label used for missing values within subgroup columns.
    max_categorical_cardinality : int, default=100
        If a categorical column has more levels than this, it is treated as numeric (binned).
    min_group_size : int, default=0
        Minimum N to treat a subgroup as “adequate.” If >0 and a subgroup has
        ``n < min_group_size``, behavior is controlled by ``small_group``.
    small_group : {'compute_flag', 'skip', 'upsample'}, default 'compute_flag'
        - ``'compute_flag'``: compute anyway and set metadata flag (see below).
        - ``'skip'``: omit the subgroup node.
        - ``'upsample'``: bootstrap sample size is forced to ``min_group_size``.
    small_group_flag_key : str, default 'low_n'
        Metadata key set to True when ``small_group='compute_flag'`` and ``n < min_group_size``.
    compatibility_compare : {'all_pairs', 'baseline_vs_each', 'ordered_pairs'}, default 'all_pairs'
        Policy for constructing model pairs for compatibility metrics.
    compatibility_baseline_model : str, optional
        Baseline model when ``compatibility_compare='baseline_vs_each'``.
    compatibility_pair_filter : list[tuple[str, str]], optional
        Explicit model pairs to compute. When provided, overrides ``compatibility_compare``.
    store_indices : bool, default False
        If True, include bootstrap index arrays in ``_metadata`` (per node).

    Returns
    -------
    dict
        Single-label: a node dict with keys:
            ``original_metrics``, ``original_curves``,
            ``original_compatibility_metrics``,
            ``bootstrap_replication_metrics``,
            ``bootstrap_replication_curves``,
            ``bootstrap_compatibility_metrics``,
            optional ``groups``, and ``_metadata``.
        Multi-label: a dict keyed by label name → node dict as above, plus a
        top-level ``_metadata``.

        Label↔label results (when provided) are attached under each **anchor label** node as:
            ``node['original_label_metrics'][other_label][metric_name] -> float``
            ``node['bootstrap_label_metrics'][other_label][metric_name] -> np.ndarray``
    """

    # Coerce inputs (supports multi-label and multi-model; preserves back-compat)
    labels_dict, scores_dict, N = _coerce_y_inputs(y_true, y_scores)
    label_names = list(labels_dict.keys())
    model_names = list(scores_dict.keys())

    # RNG
    rng_master = np.random.default_rng(random_state)

    # Normalize function dicts/kwargs
    metric_func_dict = _get_funcs_dict(metric_funcs, 'metric_funcs')
    curve_func_dict = _get_funcs_dict(curve_funcs, 'curve_funcs')
    compatibility_metric_func_dict = _get_funcs_dict(compatibility_metric_funcs, 'compatibility_metric_funcs')

    metric_kwarg_dict = _get_funcs_kwargs_dict(metric_func_dict, metric_funcs_kwargs, 'metric_funcs_kwargs')
    curve_kwarg_dict = _get_funcs_kwargs_dict(curve_func_dict, curve_funcs_kwargs, 'curve_funcs_kwargs')
    compatibility_metric_kwarg_dict = _get_funcs_kwargs_dict(compatibility_metric_func_dict, compatibility_metric_funcs_kwargs, 'compatibility_metric_funcs_kwargs')

    # OPTIONAL: label↔label metrics
    label_metric_func_dict = _get_funcs_dict(label_metrics, 'label_metrics')
    label_metric_kwarg_dict = _get_funcs_kwargs_dict(label_metric_func_dict, label_metrics_kwargs, 'label_metrics_kwargs')

    # Compatibility pairs
    compatibility_pairs = _get_model_pairs(
        model_names,
        compatibility_compare=compatibility_compare,
        compatibility_baseline_model=compatibility_baseline_model,
        compatibility_pair_filter=compatibility_pair_filter
    )

    # Resolve groups
    groups = resolve_groups(
        group_by=group_by,
        group_binning=group_binning,
        N=N,
        max_categorical_cardinality=max_categorical_cardinality,
        missing_group_name=missing_group_name
    )

    # Helper: build one node (or per-label nodes) for a boolean mask slice
    def build_node_for_mask(mask, group_name=None, level_name=None):
        sel = np.ones(N, dtype=bool) if mask is None else mask
        n_here = int(sel.sum())
        if n_here == 0:
            return {
                'original_metrics': {},
                'original_curves': {},
                'original_compatibility_metrics': {},
                'bootstrap_replication_metrics': {},
                'bootstrap_replication_curves': {},
                'bootstrap_compatibility_metrics': {},
                # present when multi-label path is expected to merge shapes
                'original_label_metrics': {},
                'bootstrap_label_metrics': {},
                '_metadata': {'n': 0, 'empty': True}
            }

        labels_here = {ln: arr[sel] for ln, arr in labels_dict.items()}
        scores_here = {mn: arr[sel] for mn, arr in scores_dict.items()}

        # small-group policy
        md_flags, sample_size = {}, None
        if min_group_size and n_here < min_group_size:
            if small_group == "compute_flag":
                md_flags[small_group_flag_key] = True
            elif small_group == "skip":
                return {
                    'original_metrics': {},
                    'original_curves': {},
                    'original_compatibility_metrics': {},
                    'bootstrap_replication_metrics': {},
                    'bootstrap_replication_curves': {},
                    'bootstrap_compatibility_metrics': {},
                    'original_label_metrics': {},
                    'bootstrap_label_metrics': {},
                    '_metadata': {'n': n_here, 'skipped': True}
                }
            elif small_group == "upsample":
                sample_size = int(min_group_size)
            else:
                raise ValueError("small_group must be one of {'compute_flag','skip','upsample'}")

        # child RNG for this (group, level)
        base = int(rng_master.integers(0, (1 << 32) - 1))
        child_seed = stable_int_hash(base, "group", group_name, "level", level_name)
        child_rng = np.random.default_rng(child_seed)

        # ---------- single-label path (legacy shape) ----------
        if len(labels_here) == 1:
            (lname, y_arr), = labels_here.items()
            node, _meta = _build_node(
                y_arr, scores_here,
                metric_func_dict, metric_kwarg_dict,
                curve_func_dict, curve_kwarg_dict,
                compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                compatibility_pairs,
                n_bootstraps, child_rng, store_indices=store_indices, sample_size=sample_size
            )
            node['_metadata'] = {**node.get('_metadata', {}), **md_flags, 'n': n_here}
            return node

        # ---------- multi-label path ----------
        node_ml = {}

        # Shared bootstrap indices across labels in this slice
        first_label = next(iter(labels_here))
        idx = make_bootstrap_indices(len(labels_here[first_label]), n_bootstraps, child_rng, sample_size=sample_size)

        # (A) OPTIONAL: compute label↔label metrics once for the slice (original + bootstrap)
        original_label_metrics_map = {}
        bootstrap_label_metrics_map = {}

        if label_metric_func_dict:
            # ORIGINAL
            for A, yA in labels_here.items():
                original_label_metrics_map[A] = {}
                for B, yB in labels_here.items():
                    if B == A:
                        continue
                    bundle = {}
                    for mname, fn in label_metric_func_dict.items():
                        val = fn(yA, yB, **(label_metric_kwarg_dict.get(mname, {}) or {}))
                        bundle[mname] = float(val) if np.ndim(val) == 0 else val
                    original_label_metrics_map[A][B] = bundle

            # BOOTSTRAP
            tmp = {
                A: {
                    B: {m: [] for m in label_metric_func_dict.keys()}
                    for B in labels_here.keys() if B != A
                } for A in labels_here.keys()
            }
            for b in range(idx.shape[0]):
                sel_b = idx[b]
                for A, yA_full in labels_here.items():
                    yA_b = yA_full[sel_b]
                    for B, yB_full in labels_here.items():
                        if B == A:
                            continue
                        yB_b = yB_full[sel_b]
                        for mname, fn in label_metric_func_dict.items():
                            val = fn(yA_b, yB_b, **(label_metric_kwarg_dict.get(mname, {}) or {}))
                            tmp[A][B][mname].append(val)
            for A in labels_here.keys():
                bootstrap_label_metrics_map[A] = {}
                for B in labels_here.keys():
                    if B == A:
                        continue
                    bundle = {}
                    for mname in label_metric_func_dict.keys():
                        bundle[mname] = np.asarray(tmp[A][B][mname])
                    bootstrap_label_metrics_map[A][B] = bundle

        # (B) per-label nodes (metrics/curves/compat) using shared idx
        for lname, y_arr in labels_here.items():
            om, oc, ocm = _eval_original(
                y_arr, scores_here,
                metric_func_dict, metric_kwarg_dict,
                curve_func_dict, curve_kwarg_dict,
                compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                compatibility_pairs
            )
            bm, bc, bcm = _eval_bootstrap(
                y_arr, scores_here, idx,
                metric_func_dict, metric_kwarg_dict,
                curve_func_dict, curve_kwarg_dict,
                compatibility_metric_func_dict, compatibility_metric_kwarg_dict,
                compatibility_pairs
            )
            node_ml[lname] = {
                'original_metrics': om,
                'original_curves': oc,
                'original_compatibility_metrics': ocm,
                'bootstrap_replication_metrics': bm,
                'bootstrap_replication_curves': bc,
                'bootstrap_compatibility_metrics': bcm,
                # NEW: attach label↔label results (present only if label_metrics provided)
                'original_label_metrics': original_label_metrics_map.get(lname, {}),
                'bootstrap_label_metrics': bootstrap_label_metrics_map.get(lname, {}),
                '_metadata': {'n': n_here, **md_flags}
            }
            if store_indices:
                node_ml[lname]['_metadata']['bootstrap_indices'] = idx

        return node_ml

    # -------- top node (no mask) --------
    top_node = build_node_for_mask(mask=None, group_name=None, level_name=None)

    # -------- attach groups (if any) --------
    if groups:
        # single-label
        if len(labels_dict) == 1:
            groups_out, skipped_groups = {}, []
            for gname, levels in groups.items():
                levels_out = {}
                for lvl_name, mask in levels.items():
                    node = build_node_for_mask(mask=mask, group_name=gname, level_name=lvl_name)
                    meta = node.get('_metadata', {})
                    if isinstance(meta, dict) and meta.get('skipped', False):
                        skipped_groups.append((gname, lvl_name))
                        continue
                    levels_out[str(lvl_name)] = node
                if levels_out:
                    groups_out[str(gname)] = levels_out

            top_node['groups'] = groups_out
            top_node['_metadata'] = {
                **top_node.get('_metadata', {}),
                'n': int(N),
                'n_bootstraps': int(n_bootstraps),
                'store_indices': bool(store_indices),
                'groups_present': list(groups_out.keys()),
                'skipped_groups': skipped_groups,
                'compatibility_compare': compatibility_compare,
                'compatibility_baseline_model': compatibility_baseline_model,
                'models': model_names,
                'labels': label_names,
            }
            return top_node

        # multi-label
        groups_by_label = {lname: {} for lname in label_names}
        skipped_groups = []
        for gname, levels in groups.items():
            for lname in label_names:
                groups_by_label[lname].setdefault(str(gname), {})
            for lvl_name, mask in levels.items():
                node_per_label = build_node_for_mask(mask=mask, group_name=gname, level_name=lvl_name)
                maybe_meta = node_per_label.get(label_names[0], {}).get('_metadata', {})
                if isinstance(maybe_meta, dict) and maybe_meta.get('skipped', False):
                    skipped_groups.append((gname, lvl_name))
                    continue
                for lname in label_names:
                    groups_by_label[lname][str(gname)][str(lvl_name)] = node_per_label[lname]

        for lname in label_names:
            top_node[lname]['groups'] = groups_by_label[lname]
            top_node[lname]['_metadata'] = {
                **top_node[lname].get('_metadata', {}),
                'n': int(N),
                'n_bootstraps': int(n_bootstraps),
                'store_indices': bool(store_indices),
                'groups_present': list(groups_by_label[lname].keys()),
                'compatibility_compare': compatibility_compare,
                'compatibility_baseline_model': compatibility_baseline_model,
                'models': model_names,
                'label': lname,
            }

        top_node['_metadata'] = {
            'n': int(N),
            'n_bootstraps': int(n_bootstraps),
            'store_indices': bool(store_indices),
            'groups_present': list(groups.keys()),
            'skipped_groups': skipped_groups,
            'compatibility_compare': compatibility_compare,
            'compatibility_baseline_model': compatibility_baseline_model,
            'models': model_names,
            'labels': label_names,
        }
        return top_node

    # -------- no groups: finalize metadata --------
    if len(labels_dict) == 1:
        top_node['_metadata'] = {
            **top_node.get('_metadata', {}),
            'n': int(N),
            'n_bootstraps': int(n_bootstraps),
            'store_indices': bool(store_indices),
            'groups_present': [],
            'compatibility_compare': compatibility_compare,
            'compatibility_baseline_model': compatibility_baseline_model,
            'models': model_names,
            'labels': label_names,
        }
        return top_node

    top_node['_metadata'] = {
        'n': int(N),
        'n_bootstraps': int(n_bootstraps),
        'store_indices': bool(store_indices),
        'groups_present': [],
        'compatibility_compare': compatibility_compare,
        'compatibility_baseline_model': compatibility_baseline_model,
        'models': model_names,
        'labels': label_names,
    }
    return top_node




def analyze_bootstrap_results(bootstrapped_results, metric_func_name,
                              y_score_names=None,
                              confidence_level=0.95, alternative='two-sided',
                              method='basic'):
    """
    Analyze bootstrapped results for a specific metric, returning confidence
    intervals and the replication indices that fall within the CI.

    Also returns a filtered view of bootstrap replications (metrics and curves)
    restricted to those indices. Curves are assumed to be tuples of at least
    ``(x, y)`` (additional metadata is ignored).

    Parameters
    ----------
    bootstrapped_results : dict
        A single node from :func:`bootstrap_evaluation` (i.e., the legacy
        single-label shape or a per-label/per-group node).
    metric_func_name : str
        The metric key to analyze.
    y_score_names : list[str], optional
        Subset of models to include. Defaults to all models present.
    confidence_level : float, default 0.95
        CI level.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis shaping the CI bounds.
    method : {'percentile', 'basic', 'bca'}, default 'basic'
        CI method. ``'bca'`` is **not implemented** and will raise.

    Returns
    -------
    dict[str, tuple]
        Mapping from model name → ``( (ci_lo, ci_hi), ci_indices, filtered_boot )``, where
        - ``(ci_lo, ci_hi)`` are floats,
        - ``ci_indices`` is a 1D ndarray of replication indices,
        - ``filtered_boot`` has the structure:
          ``{'metrics': {metric_name: 1D array}, 'curves': {curve_name: list[(x, y, ...)]}}``.

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
    
    return ci_results


def summarize_bootstrap_results(bootstrapped_results, confidence_level=0.95, alternative='two-sided', method='basic', decimal_places=3):
    """
    Summarize bootstrapped results into human-readable CI strings.

    Returns a triple ``(mf_summary, cmf_summary, lmf_summary)``; structures
    differ for single-label vs multi-label inputs (see below).

    Parameters
    ----------
    confidence_level : float, default 0.95
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
    method : {'percentile', 'basic', 'bca'}, default 'basic'
        CI method. ``'bca'`` is **not implemented** and will raise.
    decimal_places : int, default 3
        Rounding used for the formatted strings.

    Returns
    -------
    tuple
        Single-label:
          - ``mf_summary``  : ``dict[metric][model] -> "center (lo, hi)"``
          - ``cmf_summary`` : ``dict[compat_metric][(model_a, model_b)] -> "center (lo, hi)"``
          - ``lmf_summary`` : ``dict[label_metric][other_label] -> "center (lo, hi)"``

        Multi-label:
          - ``mf_summary``  : ``dict[label][metric][model] -> {'overall': str, 'groups': {group: {level: str}}}``
          - ``cmf_summary`` : ``dict[label][compat_metric][(model_a, model_b)] -> {... as above ...}``
          - ``lmf_summary`` : ``dict[label][label_metric][other_label] -> {... as above ...}``

    """
    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))

    if method == 'bca':
        raise ValueError("method='bca' not implemented")
    else:
        interval = alpha, 1 - alpha

        def percentile_func(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # ---- helpers -------------------------------------------------------------

    def _fmt_ci(center, lo, hi):
        return f"{float(np.round(center, decimal_places))} ({float(np.round(lo, decimal_places))}, {float(np.round(hi, decimal_places))})"

    def _compute_ci_strings(theta_hat, theta_hat_b):
        ci_l = percentile_func(theta_hat_b, interval[0]*100)
        ci_u = percentile_func(theta_hat_b, interval[1]*100)
        if method == 'basic':
            ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l
        if alternative == 'less':
            ci_l = np.full_like(ci_l, -np.inf)
        elif alternative == 'greater':
            ci_u = np.full_like(ci_u, np.inf)
        return float(ci_l), float(ci_u), _fmt_ci(theta_hat, ci_l, ci_u)

    def _summarize_single_node_core(node):
        """Return (mf, cmf) for a single node (no recursion, no groups)."""
        mf_summary_results = {}
        for metric_func_name, bsr_om_mfn in (node.get('original_metrics', {}) or {}).items():
            mf_summary_results[metric_func_name] = {}
            for y_score_key, theta_hat in bsr_om_mfn.items():
                theta_hat_b = node['bootstrap_replication_metrics'][metric_func_name][y_score_key]
                _, _, s = _compute_ci_strings(theta_hat, theta_hat_b)
                mf_summary_results[metric_func_name][y_score_key] = s

        cmf_summary_results = {}
        for metric_func_name, bsr_ocm_mfn in (node.get('original_compatibility_metrics', {}) or {}).items():
            cmf_summary_results[metric_func_name] = {}
            for y_score_pair_key, theta_hat in bsr_ocm_mfn.items():
                theta_hat_b = node['bootstrap_compatibility_metrics'][metric_func_name][y_score_pair_key]
                _, _, s = _compute_ci_strings(theta_hat, theta_hat_b)
                cmf_summary_results[metric_func_name][y_score_pair_key] = s

        return mf_summary_results, cmf_summary_results

    def _summarize_label_metrics_single_node(node):
        """
        Build label↔label metric summary for a single node.
        Returns dict[label_metric][other_label] -> "center (lo, hi)".
        """
        out = {}
        om = (node.get('original_label_metrics', {}) or {})
        bm = (node.get('bootstrap_label_metrics', {}) or {})
        if not om or not bm:
            return out
        for other_label, metric_map in om.items():
            for metric_name, theta_hat in (metric_map or {}).items():
                theta_hat_b = (bm.get(other_label, {}) or {}).get(metric_name, None)
                if theta_hat_b is None:
                    continue
                _, _, s = _compute_ci_strings(theta_hat, theta_hat_b)
                out.setdefault(metric_name, {})[other_label] = s
        return out

    def _summarize_node_for_multilabel(node):
        """
        Return triple (mf_label, cmf_label, lmf_label) where groups live under metric -> column.
        """
        mf_overall, cmf_overall = _summarize_single_node_core(node)
        lmf_overall = _summarize_label_metrics_single_node(node)

        # prepare per-metric dicts with {"overall": ..., "groups": {...}}
        mf_packed = {m: {} for m in mf_overall.keys()}
        cmf_packed = {m: {} for m in cmf_overall.keys()}
        lmf_packed = {m: {} for m in lmf_overall.keys()}

        # fill overall
        for metric_name, model_map in mf_overall.items():
            for model_name, s in model_map.items():
                mf_packed[metric_name][model_name] = {"overall": s, "groups": {}}

        for compat_name, pair_map in cmf_overall.items():
            for pair_key, s in pair_map.items():
                cmf_packed[compat_name][pair_key] = {"overall": s, "groups": {}}

        for lm_name, other_map in lmf_overall.items():
            for other_label, s in other_map.items():
                lmf_packed[lm_name][other_label] = {"overall": s, "groups": {}}

        # add groups
        groups = node.get('groups', {}) or {}
        for gname, levels in groups.items():
            for lvl_name, gnode in levels.items():
                g_mf, g_cmf = _summarize_single_node_core(gnode)
                g_lmf = _summarize_label_metrics_single_node(gnode)

                for metric_name, model_map in g_mf.items():
                    if metric_name not in mf_packed:
                        mf_packed[metric_name] = {}
                    for model_name, s in model_map.items():
                        mf_packed[metric_name].setdefault(model_name, {"overall": None, "groups": {}})
                        mf_packed[metric_name][model_name]["groups"].setdefault(gname, {})
                        mf_packed[metric_name][model_name]["groups"][gname][str(lvl_name)] = s

                for compat_name, pair_map in g_cmf.items():
                    if compat_name not in cmf_packed:
                        cmf_packed[compat_name] = {}
                    for pair_key, s in pair_map.items():
                        cmf_packed[compat_name].setdefault(pair_key, {"overall": None, "groups": {}})
                        cmf_packed[compat_name][pair_key]["groups"].setdefault(gname, {})
                        cmf_packed[compat_name][pair_key]["groups"][gname][str(lvl_name)] = s

                for lm_name, other_map in g_lmf.items():
                    if lm_name not in lmf_packed:
                        lmf_packed[lm_name] = {}
                    for other_label, s in other_map.items():
                        lmf_packed[lm_name].setdefault(other_label, {"overall": None, "groups": {}})
                        lmf_packed[lm_name][other_label]["groups"].setdefault(gname, {})
                        lmf_packed[lm_name][other_label]["groups"][gname][str(lvl_name)] = s

        return mf_packed, cmf_packed, lmf_packed

    # ---- dispatch ------------------------------------------------------------

    # Case 1: single node (legacy shape)
    if 'original_metrics' in bootstrapped_results and 'bootstrap_replication_metrics' in bootstrapped_results:
        mf, cmf = _summarize_single_node_core(bootstrapped_results)
        lmf = _summarize_label_metrics_single_node(bootstrapped_results)
        return mf, cmf, lmf

    # Case 2: multi-label at top
    label_names = [k for k in bootstrapped_results.keys()
                   if k != '_metadata' and isinstance(bootstrapped_results[k], dict)]
    if len(label_names) > 0 and all(('original_metrics' in bootstrapped_results[k] or 'groups' in bootstrapped_results[k])
                                    for k in label_names):
        mf_all, cmf_all, lmf_all = {}, {}, {}
        for lname in label_names:
            node = bootstrapped_results[lname]
            mf_l, cmf_l, lmf_l = _summarize_node_for_multilabel(node)
            mf_all[lname] = mf_l
            cmf_all[lname] = cmf_l
            lmf_all[lname] = lmf_l
        return mf_all, cmf_all, lmf_all

    raise ValueError("summarize_bootstrap_results expected a single result node or a multi-label dict keyed by labels.")




def make_summary_tables(bootstrapped_results,
                        which='all',
                        show=True,
                        include_overall=True,
                        include_groups=True,
                        label_order=None,
                        group_order=None,
                        sort=True):
    """
    Build and optionally display pivoted, presentation-ready summary tables from
    :func:`summarize_bootstrap_results`.

    Produces up to three families of tables:
      - ``out['mf']``  : per-model metrics
      - ``out['cmf']`` : compatibility metrics between model pairs
      - ``out['lmf']`` : label↔label metrics (multi-label only)

    Curves are not included here. For metrics with a single model, the column
    is renamed to ``'value'`` for readability. For compatibility and label↔label,
    columns are tuple-typed (e.g., ``('model_a','model_b')``).

    Behavior
    --------
    * Single label (legacy)
        - No label header is printed.
        - If no groups are present, only an “Overall” table is shown.
    * Multi label
        - For each label, an Overall table and one table per subgroup are produced.
        - Each subgroup table includes an "overall" reference row (if ``include_overall=True``).
    * Metrics (mf)
        - Overall table: rows = ``metric``, columns = model names, values = CI strings.
        - Subgroup tables: rows = (``metric``, ``level``), columns = model names.
    * Compatibility (cmf)
        - Overall table: rows = ``compat_metric``, columns = pairs as tuples, for example ``('model_0','model_1')``.
        - Subgroup tables: rows = (``compat_metric``, ``level``), columns = pairs as tuples.
        - Column order for pairs is preserved from the summary traversal.
    * Label to label (lmf)
        - Overall table: rows = ``label_metric``, columns = pairs as tuples, for example ``('mortality','icu_admit')``.
        - Subgroup tables: rows = (``label_metric``, ``level``), columns = pairs as tuples.

    Parameters
    ----------
    bootstrapped_results : dict
        Output of :func:`bootstrap_evaluation`. Can be a single node (legacy shape) or a
        multi label dict keyed by label.
    which : {'mf','cmf','lmf','all'}, default 'all'
        Choose which result type or types to build and return or display.
        Note: 'both' is not supported.
    show : bool, default True
        If True, display tables using ``IPython.display.display``. The DataFrames are always
        returned regardless of this flag.
    include_overall : bool, default True
        Include overall (non grouped) rows. For grouped runs, adds an "overall" level row to each
        subgroup table as a baseline reference.
    include_groups : bool, default True
        Include subgroup tables when groups are present. Has no effect if no groups exist.
    label_order : list of str, optional
        Explicit ordering of label names when multiple labels are present. Unlisted labels follow.
    group_order : list of str, optional
        Explicit ordering of subgroup names across subgroup tables. Unlisted groups follow.
    sort : bool, default True
        Reserved for future use. Kept for API compatibility.

    Returns
    -------
    dict
        Nested mapping of DataFrames. Keys present depend on ``which``::

            {
              'mf':  { <label_or_single>: {'overall': df, 'groups': {g: df}} },
              'cmf': { <label_or_single>: {'overall': df, 'groups': {g: df}} },
              'lmf': { <label_or_single>: {'overall': df, 'groups': {g: df}} }
            }

        Notes on columns:
        - For metrics, if there is only one model the single model column is renamed to ``'value'``.
        - For compatibility and label to label, even with a single pair, the column header remains
          the pair tuple.

    See Also
    --------
    summarize_bootstrap_results : Produces the nested summary dict that this function formats.
    """
    if which == 'both':
        raise ValueError("which='both' is no longer supported. Use 'all' or one of {'mf','cmf','lmf'}.")

    import pandas as pd
    from IPython.display import display

    # tolerate older summarize function that returned only two values
    res = summarize_bootstrap_results(bootstrapped_results)
    if isinstance(res, tuple) and len(res) == 3:
        mf_summary, cmf_summary, lmf_summary = res
    else:
        mf_summary, cmf_summary = res
        lmf_summary = {}

    # detect single label legacy shape
    is_single_label = isinstance(bootstrapped_results, dict) and (
        ('original_metrics' in bootstrapped_results and 'bootstrap_replication_metrics' in bootstrapped_results)
        or (('_metadata' in bootstrapped_results) and not any(
            isinstance(v, dict) and ('original_metrics' in v or 'groups' in v)
            for k, v in bootstrapped_results.items() if k != '_metadata'
        ))
    )
    if is_single_label:
        labels = ['__single__']
        mf_ml = {'__single__': mf_summary}
        cmf_ml = {'__single__': cmf_summary}
        lmf_ml = {'__single__': lmf_summary}
    else:
        labels = list(mf_summary.keys()) if isinstance(mf_summary, dict) else []
        if label_order:
            labels = [l for l in label_order if l in labels] + [l for l in labels if l not in (label_order or [])]
        mf_ml, cmf_ml, lmf_ml = mf_summary, cmf_summary, lmf_summary

    out = {}
    include_mf  = which in ('mf', 'all')
    include_cmf = which in ('cmf', 'all')
    include_lmf = which in ('lmf', 'all')
    if include_mf:
        out['mf'] = {}
    if include_cmf:
        out['cmf'] = {}
    if include_lmf:
        out['lmf'] = {}

    def _ordered_levels(level_series):
        if level_series.empty:
            return ['overall']
        vals = [str(v) for v in level_series.tolist() if pd.notna(v)]
        seen = set()
        vals_unique = [v for v in vals if not (v in seen or seen.add(v))]
        rest = [v for v in vals_unique if v != 'overall']
        cats = (['overall'] if 'overall' in vals_unique or include_overall else []) + sorted(set(rest))
        return list(dict.fromkeys(cats))

    def _pivot(df, index_cols, column_col):
        """Safe pivot with stable ordering."""
        import pandas as pd

        if df.empty:
            return pd.DataFrame(columns=index_cols)

        p = df.pivot_table(index=index_cols, columns=column_col, values='value',
                           aggfunc='first', observed=False)

        # Keep column order stable when categorical, else sort by string
        if hasattr(df[column_col].dtype, "categories") and df[column_col].dtype.ordered:
            p = p.reindex(df[column_col].dtype.categories, axis=1)
        else:
            p = p.reindex(sorted(p.columns, key=lambda x: str(x)), axis=1)

        # Remove the top header like "model" or "pair" for consistency
        p.columns.name = None

        return p.reset_index()


    def _mf_long_rows_for_label(label):
        block = mf_ml.get(label, {}) or {}
        for metric, model_map in block.items():
            if metric == 'groups':
                continue
            for model, val in model_map.items():
                if isinstance(val, dict) and ('overall' in val or 'groups' in val):
                    if include_overall and (val.get('overall') is not None):
                        yield (metric, 'overall', 'overall', model, val['overall'])
                    if include_groups:
                        for gname, levels in (val.get('groups', {}) or {}).items():
                            for lvl, s in levels.items():
                                yield (metric, str(gname), str(lvl), model, s)
                else:
                    if include_overall:
                        yield (metric, 'overall', 'overall', model, val)

    def _cmf_long_rows_for_label(label):
        block = cmf_ml.get(label, {}) or {}
        for compat_metric, pair_map in block.items():
            if compat_metric == 'groups':
                continue
            for pair_key, val in pair_map.items():
                pair_tpl = tuple(pair_key) if isinstance(pair_key, tuple) else (str(pair_key),)
                if isinstance(val, dict) and ('overall' in val or 'groups' in val):
                    if include_overall and (val.get('overall') is not None):
                        yield (compat_metric, 'overall', 'overall', pair_tpl, val['overall'])
                    if include_groups:
                        for gname, levels in (val.get('groups', {}) or {}).items():
                            for lvl, s in levels.items():
                                yield (compat_metric, str(gname), str(lvl), pair_tpl, s)
                else:
                    if include_overall:
                        yield (compat_metric, 'overall', 'overall', pair_tpl, val)

    def _lmf_long_rows_for_label(label):
        """
        Yield rows as (label_metric, group, level, pair_tuple, value),
        where pair_tuple == (anchor_label, other_label).
        """
        block = lmf_ml.get(label, {}) or {}
        anchor = label
        for lm_name, other_map in block.items():
            if lm_name == 'groups':
                continue
            for other_label_name, val in other_map.items():
                pair_tpl = (anchor, other_label_name)
                if isinstance(val, dict) and ('overall' in val or 'groups' in val):
                    if include_overall and (val.get('overall') is not None):
                        yield (lm_name, 'overall', 'overall', pair_tpl, val['overall'])
                    if include_groups:
                        for gname, levels in (val.get('groups', {}) or {}).items():
                            for lvl, s in levels.items():
                                yield (lm_name, str(gname), str(lvl), pair_tpl, s)
                else:
                    if include_overall:
                        yield (lm_name, 'overall', 'overall', pair_tpl, val)

    for label in labels:
        label_title = "" if label == '__single__' else f" (label = {label})"

        # ---------- METRICS ----------
        if include_mf:
            out.setdefault('mf', {})[label] = {'overall': None, 'groups': {}}
            mf_long = pd.DataFrame(list(_mf_long_rows_for_label(label)),
                                   columns=['metric', 'group', 'level', 'model', 'value'])
            if not mf_long.empty:
                # overall
                mf_overall_long = mf_long[(mf_long['group'] == 'overall') & (mf_long['level'] == 'overall')]
                if not mf_overall_long.empty:
                    mf_overall_long = mf_overall_long.sort_values(['metric', 'model']).reset_index(drop=True)
                mf_overall = _pivot(mf_overall_long, index_cols=['metric'], column_col='model')
                if not mf_overall.empty and (mf_overall.shape[1] == 2):
                    only_model = [c for c in mf_overall.columns if c != 'metric'][0]
                    mf_overall = mf_overall.rename(columns={only_model: 'value'})
                out['mf'][label]['overall'] = mf_overall

                # groups
                if include_groups:
                    gnames = sorted([g for g in mf_long['group'].unique() if g != 'overall'])
                    if group_order:
                        gnames = [g for g in group_order if g in gnames] + [g for g in gnames if g not in (group_order or [])]
                    for gname in gnames:
                        sub = mf_long[mf_long['group'].isin([gname])]
                        if include_overall and not mf_overall_long.empty:
                            ref = mf_overall_long.assign(group=gname, level='overall')
                            sub = pd.concat([ref, sub], ignore_index=True)
                        if not sub.empty:
                            cats = _ordered_levels(sub['level'])
                            sub['level'] = pd.Categorical(sub['level'].astype(str), categories=cats, ordered=True)
                            sub = sub.sort_values(['metric', 'level', 'model']).reset_index(drop=True)
                        mf_group = _pivot(sub, index_cols=['metric', 'level'], column_col='model')
                        if not mf_group.empty and (mf_group.shape[1] == 3):
                            only_model = [c for c in mf_group.columns if c not in ('metric', 'level')][0]
                            mf_group = mf_group.rename(columns={only_model: 'value'})
                        out['mf'][label]['groups'][gname] = mf_group

                if show:
                    if not mf_overall.empty:
                        heading = "=== Metrics, Overall ===" if label == '__single__' else f"=== Metrics{label_title}, Overall ==="
                        print(f"\n{heading}")
                        display(mf_overall)
                    if include_groups and len(out['mf'][label]['groups']) > 0:
                        for gname, df_g in out['mf'][label]['groups'].items():
                            if df_g.empty:
                                continue
                            heading_g = f"=== Metrics{label_title}, Group: {gname} ==="
                            print(f"\n{heading_g}")
                            display(df_g)

        # ---------- COMPATIBILITY ----------
        if include_cmf:
            out.setdefault('cmf', {})[label] = {'overall': None, 'groups': {}}
            cmf_long = pd.DataFrame(list(_cmf_long_rows_for_label(label)),
                                    columns=['compat_metric', 'group', 'level', 'pair', 'value'])
            if not cmf_long.empty:
                # preserve pair order
                if 'pair' in cmf_long.columns and not cmf_long.empty:
                    pair_order = []
                    seen = set()
                    for p in cmf_long['pair'].tolist():
                        if p not in seen:
                            seen.add(p); pair_order.append(p)
                    cmf_long['pair'] = pd.Categorical(cmf_long['pair'], categories=pair_order, ordered=True)

                # overall
                cmf_overall_long = cmf_long[(cmf_long['group'] == 'overall') & (cmf_long['level'] == 'overall')]
                if not cmf_overall_long.empty:
                    cmf_overall_long = cmf_overall_long.sort_values(['compat_metric', 'pair']).reset_index(drop=True)
                cmf_overall = _pivot(cmf_overall_long, index_cols=['compat_metric'], column_col='pair')
                out['cmf'][label]['overall'] = cmf_overall

                # groups
                if include_groups:
                    gnames = sorted([g for g in cmf_long['group'].unique() if g != 'overall'])
                    if group_order:
                        gnames = [g for g in group_order if g in gnames] + [g for g in gnames if g not in (group_order or [])]
                    for gname in gnames:
                        sub = cmf_long[cmf_long['group'].isin([gname])]
                        if include_overall and not cmf_overall_long.empty:
                            ref = cmf_overall_long.assign(group=gname, level='overall')
                            sub = pd.concat([ref, sub], ignore_index=True)
                        if not sub.empty:
                            cats = _ordered_levels(sub['level'])
                            sub['level'] = pd.Categorical(sub['level'].astype(str), categories=cats, ordered=True)
                            sub = sub.sort_values(['compat_metric', 'level', 'pair']).reset_index(drop=True)
                        cmf_group = _pivot(sub, index_cols=['compat_metric', 'level'], column_col='pair')
                        out['cmf'][label]['groups'][gname] = cmf_group

                if show:
                    if not cmf_overall.empty:
                        heading = "=== Compatibility, Overall ===" if label == '__single__' else f"=== Compatibility{label_title}, Overall ==="
                        print(f"\n{heading}")
                        display(cmf_overall)
                    if include_groups and len(out['cmf'][label]['groups']) > 0:
                        for gname, df_g in out['cmf'][label]['groups'].items():
                            if df_g.empty:
                                continue
                            heading_g = f"=== Compatibility{label_title}, Group: {gname} ==="
                            print(f"\n{heading_g}")
                            display(df_g)

        # ---------- LABEL TO LABEL ----------
        if include_lmf:
            out.setdefault('lmf', {})[label] = {'overall': None, 'groups': {}}
            lmf_long = pd.DataFrame(list(_lmf_long_rows_for_label(label)),
                                    columns=['label_metric', 'group', 'level', 'pair', 'value'])
            if not lmf_long.empty:
                # preserve pair order by encounter
                if 'pair' in lmf_long.columns and not lmf_long.empty:
                    pair_order = []
                    seen = set()
                    for p in lmf_long['pair'].tolist():
                        if p not in seen:
                            seen.add(p); pair_order.append(p)
                    lmf_long['pair'] = pd.Categorical(lmf_long['pair'], categories=pair_order, ordered=True)

                # overall
                lmf_overall_long = lmf_long[(lmf_long['group'] == 'overall') & (lmf_long['level'] == 'overall')]
                if not lmf_overall_long.empty:
                    lmf_overall_long = lmf_overall_long.sort_values(['label_metric', 'pair']).reset_index(drop=True)
                lmf_overall = _pivot(lmf_overall_long, index_cols=['label_metric'], column_col='pair')
                out['lmf'][label]['overall'] = lmf_overall

                # groups
                if include_groups:
                    gnames = sorted([g for g in lmf_long['group'].unique() if g != 'overall'])
                    if group_order:
                        gnames = [g for g in group_order if g in gnames] + [g for g in gnames if g not in (group_order or [])]
                    for gname in gnames:
                        sub = lmf_long[lmf_long['group'].isin([gname])]
                        if include_overall and not lmf_overall_long.empty:
                            ref = lmf_overall_long.assign(group=gname, level='overall')
                            sub = pd.concat([ref, sub], ignore_index=True)
                        if not sub.empty:
                            cats = _ordered_levels(sub['level'])
                            sub['level'] = pd.Categorical(sub['level'].astype(str), categories=cats, ordered=True)
                            sub = sub.sort_values(['label_metric', 'level', 'pair']).reset_index(drop=True)
                        lmf_group = _pivot(sub, index_cols=['label_metric', 'level'], column_col='pair')
                        out['lmf'][label]['groups'][gname] = lmf_group

                if show:
                    if not lmf_overall.empty:
                        heading = "=== Label to Label, Overall ===" if label == '__single__' else f"=== Label to Label (label = {label}), Overall ==="
                        print(f"\n{heading}")
                        display(lmf_overall)
                    if include_groups and len(out['lmf'][label]['groups']) > 0:
                        for gname, df_g in out['lmf'][label]['groups'].items():
                            if df_g.empty:
                                continue
                            heading_g = f"=== Label to Label{' ' if label == '__single__' else f' (label = {label}) '}, Group: {gname} ==="
                            print(f"\n{heading_g}")
                            display(df_g)

    return out








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
    - method : {'percentile', 'basic', 'bca'}, default='percentile'
        Method for CI calculation.
        CI method. ``'bca'`` is **not yet implemented**.
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
    >>> _ = plot_bootstrap_curve(
    ...     bootstrapped_results,
    ...     metric_func_name='roc_auc',
    ...     curve_func_name='roc_curve',
    ...     xlabel='False Positive Rate',
    ...     ylabel='True Positive Rate',
    ...     title='ROC Curve',
    ...     legend_title='AUROC',
    ... )
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
    
    
def make_curve_figures(
    bootstrapped_results,
    curves,
    *,
    y_score_names=None,
    include_overall=True,
    include_groups=True,
    label_order=None,
    group_order=None,
    show=True,
    figsize=(6, 6),
    rep_line_alpha=0.03,
    line_alpha=1.0,
    max_rep_lines_per_model=500,
    legend_title_ci_flag=True,
    save_dir=None,
    save_format="png",
    dpi=150,
    # NEW: panel layout control
    panel_max_cols=3,
):
    """
    Build (and optionally display/save) matplotlib Figures for curves, mirroring
    :func:`make_summary_tables` traversal.

    - Overall figures: single axes per curve with a legend on that axes.
    - Subgroup figures: **panel of subplots (one per level)**; each subplot has
      its **own legend** so the metric center+CI are visible per level.
    - Panel width auto-adapts to number of levels; capped by ``panel_max_cols``.

    Curves must be available in the node as tuples of at least ``(x, y)``.
    CI computation uses the metric named in each curve config entry.
    ```
    curves = {
      "roc_curve": {
        "metric": "roc_auc",
        "xlabel": "FPR", "ylabel": "TPR",
        "title": "ROC", "legend_title": "AUROC",
        "confidence_level": 0.95, "method": "basic", "alternative": "two-sided",
      },
      ...
    }

    Parameters
    ----------
    bootstrapped_results : dict
        Output of :func:`bootstrap_evaluation`. Supports single-label legacy shape or multi-label dict.
    curves : dict[str, dict]
        Mapping of curve function name -> plotting config. For each entry:
            {
              "metric": "<metric_func_name_for_CI>",            # required
              "xlabel": "x label",                               # optional
              "ylabel": "y label",                               # optional
              "title":  "Figure title (base, no context)",       # optional; default=f"{curve_name}"
              "legend_title": "Legend title (base, no CI suffix)"# optional; default=metric
            }
    y_score_names : list[str], optional
        Subset of model names to plot. Defaults to all models available within each node.
    include_overall : bool, default=True
        Render the overall (non-grouped) figures.
    include_groups : bool, default=True
        Render subgroup panel figures if present.
    label_order : list[str], optional
        Explicit ordering of labels when multi-label; unlisted follow in original order.
    group_order : list[str], optional
        Explicit ordering of subgroup names; unlisted follow in sorted order.
    show : bool, default=True
        If True, call `plt.show()` for each figure as it is created.
    figsize : tuple, default=(6, 6)
        Size for **single-axes** figures (overall). Subgroup panel figures scale by rows.
    rep_line_alpha : float, default=0.03
        Alpha for bootstrap replication lines (“spaghetti”).
    line_alpha : float, default=1.0
        Alpha for the original (bold) curve line.
    max_rep_lines_per_model : int, default=500
        Cap on bootstrap replication lines per model per axes (sampled without replacement).
    legend_title_ci_flag : bool, default=True
        If True, append "(95% CI)" (or appropriate %) to the legend title.
    save_dir : str or Path, optional
        If provided, save each figure under this directory. Filenames are auto-slugged.
    save_format : {"png","pdf","svg",...}, default="png"
        File format when saving.
    dpi : int, default=150
        Save resolution.

    Returns
    -------
    dict
        Nested mapping of figures. Keys mirror `make_summary_tables`:
            {
              <label_or_single>: {
                  "overall": { <curve_name>: Figure },
                  "groups":  { <group_name>: { <curve_name>: Figure } }   # panel per group
              },
              ...
            }

    Notes
    -----
    * Compatibility metrics are not plotted here; this function focuses on per-model curves.
    * If a requested curve function is missing from a node, it is skipped with a single-line warning.
    """
    from pathlib import Path
    import math

 # ---------------- helpers ----------------

    def _is_single_label_node(res):
        return isinstance(res, dict) and (
            ('original_metrics' in res and 'bootstrap_replication_metrics' in res)
            or (('_metadata' in res) and not any(
                isinstance(v, dict) and ('original_metrics' in v or 'groups' in v)
                for k, v in res.items() if k != '_metadata'
            ))
        )

    def _label_names(res):
        return [k for k, v in res.items()
                if k != '_metadata' and isinstance(v, dict) and ('original_metrics' in v or 'groups' in v)]

    def _slug(s):
        return str(s).strip().replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace(",", "")

    def _get_models_from_node(node, metric_name):
        om = node.get('original_metrics', {}) or {}
        if metric_name not in om:
            return []
        return list((om[metric_name] or {}).keys())

    def _pick_models(node, metric_name, requested):
        models_here = _get_models_from_node(node, metric_name)
        if requested is None:
            return models_here
        return [m for m in requested if m in models_here]

    def _legend_title(curve_cfg):
        lt = curve_cfg.get("legend_title", curve_cfg.get("metric", ""))
        if legend_title_ci_flag:
            cl = curve_cfg.get("confidence_level", 0.95)
            lt = f"{lt} ({int(cl*100)}% CI)"
        return lt

    def _auto_cols(n_levels, max_cols):
        # Heuristic, then cap by max_cols
        if n_levels <= 4:  cols = 2
        elif n_levels <= 6: cols = 2   # 2×3
        elif n_levels <= 9: cols = 3   # 3×3
        elif n_levels <= 16: cols = 4  # 4×4
        else: cols = 5
        return min(cols, max_cols if max_cols and max_cols > 0 else cols)

    def _plot_overall(node, *, curve_name, curve_cfg, label_for_title, save_stub):
        metric_name = curve_cfg.get("metric", None)
        if not metric_name:
            print(f"[make_curve_figures] curve '{curve_name}': missing required 'metric' key; skipping.")
            return None
        if curve_name not in (node.get('original_curves', {}) or {}):
            print(f"[make_curve_figures] curve '{curve_name}' not present in node; skipping.")
            return None

        model_list = _pick_models(node, metric_name, y_score_names)
        if len(model_list) == 0:
            print(f"[make_curve_figures] curve '{curve_name}': no models for metric '{metric_name}'; skipping.")
            return None

        ci_results = analyze_bootstrap_results(
            node, metric_name,
            y_score_names=model_list,
            confidence_level=curve_cfg.get("confidence_level", 0.95),
            alternative=curve_cfg.get("alternative", "two-sided"),
            method=curve_cfg.get("method", "basic")
        )

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.cm.get_cmap('tab10', max(10, len(model_list)))
        legend_title = _legend_title(curve_cfg)

        for i, m in enumerate(model_list):
            (ci_lo, ci_hi), _, ci_boot = ci_results[m]
            color = plt.cm.get_cmap('tab10')(i % 10)
            light = _lighten_color(color, amount=0.9)

            reps = ci_boot['curves'].get(curve_name, [])
            if len(reps) > max_rep_lines_per_model and max_rep_lines_per_model > 0:
                rng = np.random.default_rng(0)
                keep = np.sort(rng.choice(len(reps), size=max_rep_lines_per_model, replace=False))
                reps = [reps[j] for j in keep]
            for (x, y, *_) in reps:
                ax.plot(x, y, color=light, alpha=rep_line_alpha, zorder=1)

            center = node['original_metrics'][metric_name][m]
            (x0, y0, *_) = node['original_curves'][curve_name][m]
            lab = f"{m}: {float(center):.2f} ({float(ci_lo):.2f}, {float(ci_hi):.2f})"
            ax.plot(x0, y0, color=color, alpha=line_alpha, label=lab, zorder=2)

        base = curve_cfg.get("title", curve_name)
        title = f"{label_for_title}, {base}" if label_for_title else f"{base}"
        ax.set_title(title)
        ax.set_xlabel(curve_cfg.get("xlabel", ""))
        ax.set_ylabel(curve_cfg.get("ylabel", ""))
        ax.legend(title=legend_title, loc="lower right", frameon=False)

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fname = f"{_slug(save_stub)}.{_slug(curve_name)}.{save_format}"
            fig.savefig(Path(save_dir) / fname, format=save_format, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        return fig

    def _plot_group_panel(node, *, curve_name, curve_cfg, label_for_title, group_name, save_stub):
        """
        Panel figure for a single group across all its levels.
        Each subplot shows that level with all models, and **has its own legend**.
        """
        groups = (node.get('groups', {}) or {})
        levels_map = (groups.get(group_name, {}) or {})
        if len(levels_map) == 0:
            return None

        metric_name = curve_cfg.get("metric", None)
        if not metric_name:
            print(f"[make_curve_figures] curve '{curve_name}': missing required 'metric' key; skipping group '{group_name}'.")
            return None

        lvl_keys = [str(k) for k in levels_map.keys()]
        # deterministic sort (string); feel free to adjust to your bin label style
        lvl_keys.sort()
        n_levels = len(lvl_keys)
        ncols = _auto_cols(n_levels, panel_max_cols)
        nrows = int(math.ceil(n_levels / ncols))

        # Scale panel -> keep each axes around `figsize`
        single_w, single_h = figsize
        panel_w = single_w * ncols
        panel_h = max(single_h * nrows, single_h * 1.2)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(panel_w, panel_h), squeeze=False)
        axes_flat = axes.ravel()

        # Fix color mapping across all levels for consistency
        model_list_parent = _pick_models(node, metric_name, y_score_names)
        if len(model_list_parent) == 0:
            print(f"[make_curve_figures] curve '{curve_name}': no models for metric '{metric_name}'; skipping group '{group_name}'.")
            plt.close(fig)
            return None
        cmap = plt.cm.get_cmap('tab10', max(10, len(model_list_parent)))
        legend_title = _legend_title(curve_cfg)

        for ax_idx, lvl in enumerate(lvl_keys):
            ax = axes_flat[ax_idx]
            level_node = levels_map[lvl]
            if not isinstance(level_node, dict) or 'original_curves' not in level_node:
                ax.axis('off')
                continue

            # CI per level (so CI reflects level-specific sampling)
            ci_results = analyze_bootstrap_results(
                level_node, metric_name,
                y_score_names=model_list_parent,
                confidence_level=curve_cfg.get("confidence_level", 0.95),
                alternative=curve_cfg.get("alternative", "two-sided"),
                method=curve_cfg.get("method", "basic")
            )

            # draw
            handles, labels = [], []
            for i, m in enumerate(model_list_parent):
                (ci_lo, ci_hi), _, ci_boot = ci_results[m]
                color = cmap(i % cmap.N)
                light = _lighten_color(color, amount=0.9)

                reps = ci_boot['curves'].get(curve_name, [])
                if len(reps) > max_rep_lines_per_model and max_rep_lines_per_model > 0:
                    rng = np.random.default_rng(0)
                    keep = np.sort(rng.choice(len(reps), size=max_rep_lines_per_model, replace=False))
                    reps = [reps[j] for j in keep]
                for (x, y, *_) in reps:
                    ax.plot(x, y, color=light, alpha=rep_line_alpha, zorder=1)

                center = level_node['original_metrics'][metric_name][m]
                (x0, y0, *_) = level_node['original_curves'][curve_name][m]
                lab = f"{m}: {float(center):.2f} ({float(ci_lo):.2f}, {float(ci_hi):.2f})"
                line, = ax.plot(x0, y0, color=color, alpha=line_alpha, label=lab, zorder=2)
                handles.append(line); labels.append(lab)

            # axis title + labels
            base = curve_cfg.get("title", curve_name)
            axis_title = f"{label_for_title}, {base}, {group_name}: {lvl}" if label_for_title else f"{base}, {group_name}: {lvl}"
            ax.set_title(axis_title)
            ax.set_xlabel(curve_cfg.get("xlabel", ""))
            ax.set_ylabel(curve_cfg.get("ylabel", ""))
            ax.legend(handles, labels, title=legend_title, loc="lower right", frameon=False)

        # turn off extra axes
        for j in range(n_levels, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.tight_layout()

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fname = f"{_slug(save_stub)}.{_slug(curve_name)}.{save_format}"
            fig.savefig(Path(save_dir) / fname, format=save_format, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        return fig

    # ---------------- dispatch & traversal ----------------

    out = {}

    # Single-label
    if _is_single_label_node(bootstrapped_results):
        label_key = "__single__"
        out[label_key] = {"overall": {}, "groups": {}}
        label_for_title = ""

        if include_overall:
            node = bootstrapped_results
            for c_name, c_cfg in curves.items():
                fig = _plot_overall(
                    node,
                    curve_name=c_name,
                    curve_cfg=c_cfg,
                    label_for_title=label_for_title,
                    save_stub="overall"
                )
                if fig is not None:
                    out[label_key]["overall"][c_name] = fig

        if include_groups:
            node = bootstrapped_results
            groups = (node.get('groups', {}) or {})
            gnames = list(groups.keys())
            if group_order:
                gnames = [g for g in group_order if g in gnames] + [g for g in gnames if g not in (group_order or [])]
            else:
                gnames = sorted(gnames)
            for g in gnames:
                out[label_key]["groups"].setdefault(g, {})
                for c_name, c_cfg in curves.items():
                    fig = _plot_group_panel(
                        node,
                        curve_name=c_name,
                        curve_cfg=c_cfg,
                        label_for_title=label_for_title,
                        group_name=g,
                        save_stub=f"group={_slug(g)}"
                    )
                    if fig is not None:
                        out[label_key]["groups"][g][c_name] = fig
        return out

    # Multi-label
    labels = _label_names(bootstrapped_results)
    if label_order:
        labels = [l for l in label_order if l in labels] + [l for l in labels if l not in (label_order or [])]

    for lbl in labels:
        out[lbl] = {"overall": {}, "groups": {}}
        node_lbl = bootstrapped_results[lbl]
        label_for_title = lbl

        if include_overall:
            for c_name, c_cfg in curves.items():
                fig = _plot_overall(
                    node_lbl,
                    curve_name=c_name,
                    curve_cfg=c_cfg,
                    label_for_title=label_for_title,
                    save_stub=f"label={_slug(lbl)}.overall"
                )
                if fig is not None:
                    out[lbl]["overall"][c_name] = fig

        if include_groups:
            groups = (node_lbl.get('groups', {}) or {})
            gnames = list(groups.keys())
            if group_order:
                gnames = [g for g in group_order if g in gnames] + [g for g in gnames if g not in (group_order or [])]
            else:
                gnames = sorted(gnames)
            for g in gnames:
                out[lbl]["groups"].setdefault(g, {})
                for c_name, c_cfg in curves.items():
                    fig = _plot_group_panel(
                        node_lbl,
                        curve_name=c_name,
                        curve_cfg=c_cfg,
                        label_for_title=label_for_title,
                        group_name=g,
                        save_stub=f"label={_slug(lbl)}.group={_slug(g)}"
                    )
                    if fig is not None:
                        out[lbl]["groups"][g][c_name] = fig

    return out


    
    
__all__ = [
  'bootstrap_evaluation',
  'analyze_bootstrap_results',
  'summarize_bootstrap_results',
  'make_summary_tables',
  'plot_bootstrap_curve',
  'make_curve_figures',
]
