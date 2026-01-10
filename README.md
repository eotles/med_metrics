# med_metrics

`med_metrics` is a Python package for evaluating machine learning models in medicine, with an emphasis on (1) robust uncertainty via bootstrapping, (2) subgroup and fairness-style evaluation, and (3) clinically oriented decision metrics (for example Number Needed to Treat, and net benefit style curves).

PyPI name: `med-metrics`  
Import name: `med_metrics`

## Installation

```bash
pip install med-metrics==0.0.6
# or
pip install med-metrics
```

## Quick start (bootstrapped AUROC)

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from med_metrics.bootstrap import bootstrap_evaluation

rng = np.random.default_rng(42)

n = 1000
y_true = rng.integers(0, 2, size=n)
y_score = rng.random(size=n)

results = bootstrap_evaluation(
    y_true=y_true,
    y_score=y_score,
    metric_func=roc_auc_score,
    n_iterations=1000,
    alpha=0.05,
)

print(f"AUROC mean: {results['mean']:.3f}")
print(f"95% CI: ({results['ci_lower']:.3f}, {results['ci_upper']:.3f})")
```

## More metrics (AUPRC, accuracy)

```python
from sklearn.metrics import average_precision_score, accuracy_score

ap_results = bootstrap_evaluation(y_true, y_score, metric_func=average_precision_score)
acc_results = bootstrap_evaluation(
    y_true,
    y_score,
    metric_func=accuracy_score,
    metric_func_kwargs={"threshold": 0.5},
)

print(ap_results)
print(acc_results)
```

## Clinical decision metrics (NNT vs treated, average NNT)

`med_metrics` supports Number Needed to Treat (NNT) style analysis using a relative risk reduction parameter `rho` (0 to 1).

Important behavior in v0.0.6:
- Regions with no absolute risk reduction are represented as NNT = ∞ (ARR = 0 → NNT = ∞).
- `average_NNTvsTreated` supports a `policy` argument controlling how ∞ regions affect the average.

```python
from med_metrics.curves import NNTvsTreated_curve
from med_metrics.metrics import average_NNTvsTreated

rho = 0.4

treated, nnt, thresholds = NNTvsTreated_curve(
    y_true=y_true,
    y_score=y_score,
    rho=rho,
    min_treated=0,
    max_treated=len(y_true),
    warn="auto",  # "auto" (default), "always", or "never"
)

avg_nnt = average_NNTvsTreated(
    y_true=y_true,
    y_score=y_score,
    rho=rho,
    min_treated=0,
    max_treated=len(y_true),
    policy="finite",   # "finite" (default), "propagate", or "clip"
    # epsilon=1e-12,    # used only if policy == "clip"
)

print("Average NNT:", avg_nnt)
```

## Subgroup and fairness-style evaluation

### Binary fairness evaluation across subgroups

```python
import pandas as pd
from med_metrics.group_evaluation import binary_fairness_evaluation

subgroups = pd.DataFrame({
    "sex": rng.choice(["F", "M"], size=n),
    "age_group": rng.choice(["<50", "50+"], size=n),
})

results = binary_fairness_evaluation(
    y_true=y_true,
    y_score=y_score,
    subgroups=subgroups,
    threshold=0.5,
)

print(results)
```

### Subgroup evaluation for multiple models

```python
from med_metrics.group_evaluation import subgroup_evaluation

y_scores_dict = {
    "model_a": y_score,
    "model_b": np.clip(y_score + rng.normal(0, 0.05, size=n), 0, 1),
}

subgroup_results = subgroup_evaluation(
    y_true=y_true,
    y_scores=y_scores_dict,
    subgroups=subgroups,
    metric_func=roc_auc_score,
)

print(subgroup_results)
```

### Binary grouped evaluation (metrics per subgroup)

```python
from med_metrics.group_evaluation import binary_grouped_evaluation

grouped_results = binary_grouped_evaluation(
    y_true=y_true,
    y_score=y_score,
    group=subgroups["sex"],
    threshold=0.5,
)

print(grouped_results)
```

## Confusion matrices

```python
from med_metrics.utils import confusion_matrix_df

cm = confusion_matrix_df(y_true, y_score, threshold=0.5)
print(cm)
```

## Notebooks (recommended for end-to-end examples)

See the `notebooks/` directory for fuller workflows, including:
- `example_usage.ipynb`
- `example_usage_labels_subgroups.ipynb`
- `extended_example.ipynb`

## Development

### Docker workflow (recommended)

```bash
docker-compose up --build
```

Then open JupyterLab at:

- http://localhost:8888

### Local (conda)

```bash
conda env create -f requirements.txt
conda activate med_metrics
```

Run tests:

```bash
pytest
```

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

