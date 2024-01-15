# Med Metrics: Advanced Medical Machine Learning Evaluation Toolkit

## Overview

`med_metrics` is a Python package tailored for the evaluation of machine learning models in medical contexts. This package offers a unique suite of metrics, compatibility assessments, and bootstrapping techniques specifically designed to assess the performance and impact of models in healthcare.

## Key Features
- Specialized Medical Metrics: Functions for advanced metrics such as Number Needed to Treat (NNT) across decision thresholds, average height of NNT vs. treated curves, and net benefit analysis.
- Curves for Evaluation: Generate various curves like NNT vs. treated, and net benefit curves, essential for visual and quantitative model analysis.
- Compatibility Metrics: Evaluate prediction compatibility across model updates or between different models, crucial for maintaining trust in evolving medical ML applications.
- Bootstrap Evaluation: Robust tools for performing bootstrap evaluations, enabling detailed performance comparisons across different machine learning models.


## Features

- Specialized Medical Metrics: Calculate metrics like Number Needed to Treat (NNT) across decision thresholds, average height of NNT vs. treated curves, and net benefit analysis.
- Compatibility Assessment: Evaluate how predictions change with model updates or across different models using functions like backwards_trust_compatibility.
- Utility Functions: A range of utility functions for generating classification and confusion matrix curves, and handling various inputs for metrics calculation.

## Installation

To install med_metrics, run the following command:
``` bash
pip install med-metrics
```

## Dependencies

med_metrics requires the following libraries:

- numpy
- scikit-learn
- scipy

These dependencies are automatically installed with med_metrics.

## Usage

The med_metrics package can be used to perform bootstrap evaluations for model comparison. Below is an example showcasing how to compare two machine learning models using the package's bootstrapping functionality.

### Model Comparison 
This example demonstrates a bootstrap analysis to compare two models using `roc_auc_score` and `average_NNTvsTreated` metrics, as well as generating `roc_curve` and `NNT vs. Number Treated curves`.

```python
import numpy as np
from med_metrics.bootstrap import bootstrap_evaluation, summarize_bootstrap_results
from med_metrics.plotting import plot_bootstrap_curve
from sklearn.metrics import roc_auc_score, roc_curve
from med_metrics.metrics import average_NNTvsTreated
from med_metrics.curves import NNTvsTreated_curve
import pandas as pd

# Simulation of ground truth and model predictions
n = 1000
rng = np.random.default_rng(42)
p = rng.uniform(0, 1, n)
q = rng.uniform(0, 1, n)
y_true = rng.binomial(1, p)

# Bootstrap parameters
y_scores = {'model_0': p * q, 'model_1': p}
metric_funcs = {'roc_auc_score': roc_auc_score, 'average_NNTvsTreated': average_NNTvsTreated}
metric_funcs_kwargs = {'average_NNTvsTreated': {'rho': 0.4}}
curve_funcs = {'roc_curve': roc_curve, 'NNTvsT': NNTvsTreated_curve}
curve_funcs_kwargs = {'NNTvsT': {'rho': 0.4}}

# Perform the bootstrap analysis
bootstrapped_results = bootstrap_evaluation(
    y_true=y_true,
    y_scores=y_scores,
    metric_funcs=metric_funcs,
    curve_funcs=curve_funcs,
    n_bootstraps=1000,
    random_state=42,
    metric_funcs_kwargs=metric_funcs_kwargs,
    curve_funcs_kwargs=curve_funcs_kwargs
)

# Summarize the bootstrap results
mf_summary_results, _ = summarize_bootstrap_results(bootstrapped_results)
display(pd.DataFrame(mf_summary_results))

# Plot the bootstrap analysis results
_ = plot_bootstrap_curve(bootstrapped_results, 'average_NNTvsTreated', 'NNTvsT',
                         xlabel='Number Treated', ylabel='NNT',
                         title='NNT vs. Number Treated', legend_title='Mean NNT (95% CI)')
```

The above code performs the bootstrap analysis and generates a summary table, as well as a plot for NNT vs. Number Treated. The results are shown below:

Bootstrap Summary Table
```scss
Copy code
roc_auc_score    average_NNTvsTreated
model_0  0.709 (0.676, 0.741)  3.844 (3.624, 4.04)
model_1  0.818 (0.794, 0.844)  3.539 (3.366, 3.698)
```

Plot: NNT vs. Number Treated
![NNT vs. Number Treated Plot](docs/images/example_nnt_vs_treated_plot.png)



## Modules Overview
- bootstrap.py: Perform bootstrap evaluations and analyses.
- compatibility_metrics.py: Functions for assessing prediction compatibility.
- curves.py: Generate various evaluative curves.
- metrics.py: Core module for specialized medical metrics.

## Contributing

Contributions to med_metrics are welcome! Please read our contributing guidelines for more information on how to submit pull requests, report issues, or suggest enhancements.

## License

med_metrics is released under a MIT License.

## Contact

For questions or feedback, please contact Erkin Ötleş at hi@eotles.com .
