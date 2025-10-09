import numpy as np
import pytest

from med_metrics.bootstrap import (
    bootstrap_evaluation,
    analyze_bootstrap_results,
    summarize_bootstrap_results,
)


def test_bootstrap_supports_multiple_outcomes_and_subgroups():
    y_true = {
        'mortality': np.array([0, 1, 0, 1, 1, 0]),
        'readmission': np.array([1, 0, 1, 0, 0, 1]),
    }
    y_scores = {
        'model_a': np.array([0.2, 0.9, 0.1, 0.85, 0.7, 0.3]),
        'model_b': np.array([0.4, 0.6, 0.3, 0.5, 0.55, 0.2]),
    }
    categories = {'sex': np.array(['F', 'M', 'F', 'M', 'F', 'F'])}

    metric_funcs = {
        'mean_label': lambda y_true_values, y_score_values: float(np.mean(y_true_values))
    }

    results = bootstrap_evaluation(
        y_true,
        y_scores,
        metric_funcs=metric_funcs,
        n_bootstraps=32,
        random_state=7,
        category_vectors=categories,
    )

    mortality_mean = results['outcomes']['mortality']['overall']['original']['metrics']['mean_label']['model_a']
    assert mortality_mean == pytest.approx(np.mean(y_true['mortality']))

    readmission_mean = results['outcomes']['readmission']['overall']['original']['metrics']['mean_label']['model_b']
    assert readmission_mean == pytest.approx(np.mean(y_true['readmission']))

    female_mask = categories['sex'] == 'F'
    female_expected = np.mean(y_true['mortality'][female_mask])
    female_metric = (
        results['outcomes']['mortality']['subgroups']['sex']['F']['original']['metrics']['mean_label']['model_a']
    )
    assert female_metric == pytest.approx(female_expected)

    overall_bootstrap = (
        results['outcomes']['mortality']['overall']['bootstrap']['metrics']['mean_label']['model_a']
    )
    assert overall_bootstrap.shape == (32,)

    subgroup_bootstrap = (
        results['outcomes']['mortality']['subgroups']['sex']['M']['bootstrap']['metrics']['mean_label']['model_b']
    )
    assert subgroup_bootstrap.shape == (32,)

    assert set(results['category_levels']['sex']) == {'F', 'M'}
    assert results['bootstrap_indices'].shape == (32, len(y_true['mortality']))

    ci_results = analyze_bootstrap_results(
        results,
        'mean_label',
        outcome_name='mortality',
        category_name='sex',
        category_value='F',
    )
    assert 'model_a' in ci_results
    ci, indices, ci_payload = ci_results['model_a']
    assert len(indices) <= 32
    assert 'metrics' in ci_payload and 'curves' in ci_payload
    assert ci[0] <= ci[1]

    summary_metrics, summary_compat = summarize_bootstrap_results(
        results,
        outcome_name='mortality',
    )
    assert 'mean_label' in summary_metrics
    assert isinstance(summary_metrics['mean_label']['model_a'], str)
    assert summary_compat == {}

    subgroup_summary_metrics, _ = summarize_bootstrap_results(
        results,
        outcome_name='mortality',
        category_name='sex',
        category_value='M',
    )
    assert 'mean_label' in subgroup_summary_metrics
