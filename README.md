# med_metrics

`med_metrics` is a small Python package for **bootstrapped evaluation of predictive models**, including:
- point estimates plus bootstrap distributions and confidence intervals for metrics
- support for **multiple labels** (multi-task style evaluation)
- support for **subgroup evaluation** via a `group_by` dictionary
- optional **curve-style outputs** (for example ROC curves, decision curves)
- optional **model compatibility** metrics (pairwise comparisons between models)

The public API in this README is based on the package’s example notebook in `example_usage_labels_subgroups.pdf`.

## Installation

```bash
pip install -U "scikit-learn>=1.4"
pip install -U med-metrics
```

Import name is `med_metrics`:

```python
import med_metrics
```

## Quickstart

### Bootstrapped evaluation (single label)

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from med_metrics.bootstrap import (
    bootstrap_evaluation,
    summarize_bootstrap_results,
    make_summary_tables,
)

# Example data
rng = np.random.default_rng(0)
n = 500
y_true = rng.integers(0, 2, size=n)

# Two example models
y_scores = {
    "model_a": rng.random(n),
    "model_b": rng.random(n),
}

metric_funcs = {
    "roc_auc_score": roc_auc_score,
}

curve_funcs = {
    "roc_curve": roc_curve,
}

boot = bootstrap_evaluation(
    y_true=y_true,
    y_scores=y_scores,
    metric_funcs=metric_funcs,
    curve_funcs=curve_funcs,
    n_bootstraps=1000,
    random_state=42,
)

# Summaries as dicts / dataframes
summary = summarize_bootstrap_results(boot)
tables = make_summary_tables(boot)
```

### Multiple labels and subgroup evaluation

For multi-label evaluation, pass `y_true` as a dict keyed by label, and pass `group_by` as a dict keyed by subgroup name.

```python
from sklearn.metrics import roc_auc_score, roc_curve

from med_metrics.bootstrap import (
    bootstrap_evaluation,
    make_summary_tables,
)

# Labels (each value is an array-like of length n)
y_true = {
    "mortality": y_mortality,
    "icu_admit": y_icu,
    "sepsis": y_sepsis,
}

# Models (each value is an array-like of length n)
y_scores = {
    "model_0": y_score_0,
    "model_1": y_score_1,
    "model_2": y_score_2,
}

# Subgroups (each value is an array-like of length n)
group_by = {
    "sex": sex,   # example categorical
    "age": age,   # example numeric (will appear as binned ranges in outputs)
}

metric_funcs = {
    "roc_auc_score": roc_auc_score,
}

curve_funcs = {
    "roc_curve": roc_curve,
}

boot = bootstrap_evaluation(
    y_true=y_true,
    y_scores=y_scores,
    group_by=group_by,
    metric_funcs=metric_funcs,
    curve_funcs=curve_funcs,
    n_bootstraps=1000,
    random_state=42,
)

tables = make_summary_tables(boot)
```

## Decision-focused curves (NNT vs Treated, Net Benefit)

`med_metrics` includes helpers to compute decision curves and summary statistics:

```python
from med_metrics.curves import (
    NNTvsTreated_curve,
    average_NNTvsTreated,
    net_benefit_curve,
    average_net_benefit,
)

curve_funcs = {
    "roc_curve": roc_curve,
    "NNTvsT": NNTvsTreated_curve,
    "net_benefit": net_benefit_curve,
}

metric_funcs = {
    "roc_auc_score": roc_auc_score,
    "average_NNTvsTreated": average_NNTvsTreated,
    "average_net_benefit": average_net_benefit,
}
```

You can pass these into `bootstrap_evaluation(...)` via `curve_funcs` and `metric_funcs`.

## Model compatibility metrics

If you want to compare models pairwise (for example, stability of relative ordering), you can provide compatibility metric functions:

```python
from med_metrics.compatibility_metrics import (
    rank_based_compatibility,
    backwards_trust_compatibility,
    backwards_error_compatibility,
)

compatibility_metric_funcs = {
    "rank_based_compatibility": rank_based_compatibility,
    # You can add these as needed:
    # "backwards_trust_compatibility": backwards_trust_compatibility,
    # "backwards_error_compatibility": backwards_error_compatibility,
}
```

Then pass `compatibility_metric_funcs=...` to `bootstrap_evaluation(...)`.

## Label-to-label metrics (within a label)

For multi-label tasks, you can also compute metrics that compare labels to each other (for example, Jaccard, MCC) within each bootstrap sample.

```python
from med_metrics.label_metrics import mcc, jaccard

label_metrics = {
    "mcc": mcc,
    "jaccard": jaccard,
}

boot = bootstrap_evaluation(
    y_true=y_true,
    y_scores=y_scores,
    group_by=group_by,
    metric_funcs=metric_funcs,
    curve_funcs=curve_funcs,
    label_metrics=label_metrics,
    n_bootstraps=1000,
    random_state=42,
)
```

## Plotting: bootstrap curves and figures

There are two plotting helpers demonstrated in the example notebook:

- `plot_bootstrap_curve` for a single curve
- `make_curve_figures` to generate a collection of figures (overall and subgroup panels, per label)

```python
from med_metrics.plotting import plot_bootstrap_curve, make_curve_figures

curves = {
    "roc_curve": {
        "metric": "roc_auc_score",
        "xlabel": "False Positive Rate",
        "ylabel": "True Positive Rate",
        "title": "ROC",
        "confidence_level": 0.95,
        "method": "basic",  # or "percentile"
        "legend_title": "AUROC",
    },
    "NNTvsT": {
        "metric": "average_NNTvsTreated",
        "xlabel": "Treatment Threshold",
        "ylabel": "Avg NNT vs Treated",
        "title": "NNT vs Treated",
        "confidence_level": 0.95,
        "method": "percentile",
        "legend_title": "Avg NNTvsT",
    },
}

figs = make_curve_figures(
    boot,
    curves,
    y_score_names=["model_0", "model_1"],  # or None for all models present
    include_overall=True,
    include_groups=True,
    label_order=["mortality", "icu_admit"],  # optional display order
    group_order=["sex", "age"],
    show=True,
    figsize=(6, 6),
    rep_line_alpha=0.02,
    line_alpha=1.0,
    max_rep_lines_per_model=200,
    legend_title_ci_flag=True,
    save_dir="figs",       # optional: write images to ./figs
    save_format="png",
    dpi=150,
)

# Examples of accessing figures from the nested return dict:
fig_overall_roc_mort = figs["mortality"]["overall"]["roc_curve"]
fig_panel_age_roc_mort = figs["mortality"]["groups"]["age"]["roc_curve"]
```

## Output structure

`bootstrap_evaluation(...)` returns a nested dictionary. In the example notebook, the top-level keys include:
- one key per label (for multi-label input)
- `_metadata` with run configuration and bookkeeping

Each label entry includes (when requested) original (non-bootstrapped) values and bootstrap replication arrays for:
- metrics
- curves
- compatibility metrics
- label-to-label metrics

When `group_by` is provided, results are also organized under group names and group levels.

## Contributing

Issues and PRs are welcome. If you are adding metrics or curve functions, it helps to also add a small example and a test.

## License

See the repository for licensing details.

## Version notes (0.0.6)

- New: multi-outcome and subgroup evaluation workflows.
- Improved: NNT metrics now explicitly treat ARR=0 as NNT=∞ and warn when no finite NNT exists.
- Added: `policy` and `warn` parameters for better numerical handling.
- Added: `example_usage_labels_subgroups.ipynb` notebook.
- Added: Docker and ReadTheDocs scaffolding.

## Citation

If you use `med_metrics` in academic work, please cite the repository (and add a DOI or Zenodo badge if you mint one for releases).

## License

med_metrics is released under a MIT License.

## Contact

For questions or feedback, please contact Erkin Ötleş at hi@eotles.com .
