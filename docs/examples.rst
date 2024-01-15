Examples
========

Med Metrics offers robust functionalities for evaluating machine learning models in medical contexts. Below are several examples demonstrating how to use the package.




Generate Data Setup
------------------------------
This generates the data that will be used in all the examples.

.. code-block:: python
    
    import numpy as np
    n = 1000
    rng = np.random.default_rng(41)
    p = rng.uniform(0,1,n)
    q = np.random.uniform(0,1,n)
    y_true = rng.binomial(1, p)

    y_model_0 = p*q
    y_model_1 = p


NNT vs. Number Treated Example
------------------------------

This example shows how to calculate and visualize the Number Needed to Treat (NNT) versus the number of patients treated using `NNTvsTreated_curve` and `average_NNTvsTreated` from the `curves` and `metrics` modules.

.. code-block:: python
    # Setup and imports
    from med_metrics.curves import NNTvsTreated_curve
    from med_metrics.metrics import average_NNTvsTreated
    import matplotlib.pyplot as plt
    
    # Relative risk reduction
    rho = 0.4
    n_pos = sum(y_true)
    perfect_NNT = 1/rho
    
    # Calculate NNT vs. Number Treated Curve
    mean_NNTvsTreated = average_NNTvsTreated(y_true, y_model_0, rho=rho)
    treated, NNT, thresholds = NNTvsTreated_curve(y_true, y_model_0, rho=rho)
    
    # Plot
    plt.plot(treated, NNT, color='tab:blue',
             label='NNT vs. n Treated: {0:.3f}'.format(mean_NNTvsTreated))
    plt.scatter(n_pos, perfect_NNT, color='tab:green', marker='*',
                label='Omniscient Model: {0:.3f}'.format(perfect_NNT))
    plt.legend()
    plt.xlabel('Number Treated')
    plt.ylabel('NNT')

Backwards Trust Compatibility Example
-------------------------------------

Here, we illustrate using the `backwards_trust_compatibility` function from the `compatibility_metrics` module to calculate the BTC score.

.. code-block:: python

    from med_metrics.compatibility_metrics import backwards_trust_compatibility
    
    # Example code for calculating BTC
    btc = backwards_trust_compatibility(y_true, y_model_0>0.5, y_model_1>0.5)
    print('Backwards Trust Compatibility: {0:.3f}'.format(btc))


Rank-Based Compatibility Example
--------------------------------

This example demonstrates using the `rank_based_compatibility` function to assess the consistency in instance ranking between model versions.

.. code-block:: python

   from med_metrics.compatibility_metrics import rank_based_compatibility
    # Example code for calculating RBC
    rbc = rank_based_compatibility(y_true, y_model_0, y_model_1)
    print('Rank-based Compatibility: {0:.3f}'.format(rbc))
    

Bootstrap Evaluation Example
----------------------------

Illustration of performing a comprehensive bootstrap analysis using `bootstrap_evaluation` from the `bootstrap` module.

.. code-block:: python

    from med_metrics.bootstrap import bootstrap_evaluation, summarize_bootstrap_results, plot_bootstrap_curve
    # Example code for bootstrap analysis...
    bootstrapped_results = bootstrap_evaluation(
        y_true=y_true,
        y_scores={'model_0': y_model_0},
        metric_funcs={'roc_auc_score': roc_auc_score,
            'mean_NNTvsTreated': average_NNTvsTreated},
        curve_funcs={'roc_curve': roc_curve,
            'NNTvsT': NNTvsTreated_curve},
        n_bootstraps=1000,
        random_state=42,
        metric_funcs_kwargs={'mean_NNTvsTreated': {'rho':0.4}},
        curve_funcs_kwargs={'NNTvsT': {'rho':0.4}}
    )
    

Model Comparison via Bootstrapping
----------------------------------

This example demonstrates how to compare multiple models using bootstrapping techniques provided by Med Metrics.

.. code-block:: python
    # Set up
    metric_funcs = {'roc_auc_score': roc_auc_score,
                'mean_NNTvsTreated': average_NNTvsTreated,
                'mean_NNTvsTreated_10to30PercentTreated': average_NNTvsTreated,
                'mean_net_benefit': average_net_benefit,
                'mean_net_benefit_0to25PercentThreshold': average_net_benefit
               }

    metric_funcs_kwargs = {'mean_NNTvsTreated': {'rho':0.4},
                       'mean_NNTvsTreated_10to30PercentTreated': {'rho':0.4,
                                                                     'min_treated': n*0.1,
                                                                     'max_treated': n*0.3},
                       'mean_net_benefit_0to25PercentThreshold': {'max_threshold': 0.25},
                      }

    curve_funcs = {'roc_curve': roc_curve,
               'NNTvsT': NNTvsTreated_curve,
               'NNTvsT_10to30PercentTreated': NNTvsTreated_curve,
               'net_benefit_curve': net_benefit_curve,
               'net_benefit_curve_0to25PercentThreshold': net_benefit_curve,
              }

    curve_funcs_kwargs = {'NNTvsT': {'rho':0.4},
                      'NNTvsT_10to30PercentTreated': {'rho':0.4,
                                               'min_treated': n*0.1,
                                               'max_treated': n*0.3},
                      'net_benefit_curve_0to25PercentThreshold': {'max_threshold': 0.25},
                     }
    
    # Perform Boostrap Analysis
    bootstrapped_results = bootstrap_evaluation(
        y_true=y_true,
        y_scores={'model_0': y_model_0},
        metric_funcs=metric_funcs,
        curve_funcs=curve_funcs,
        n_bootstraps=1000,
        random_state=42,
        metric_funcs_kwargs=metric_funcs_kwargs,
        curve_funcs_kwargs=curve_funcs_kwargs
    )
    
    # Summarize Bootstrap Analysis
    mf_summary_results, _ = summarize_bootstrap_results(bootstrapped_results)
    display(pd.DataFrame(mf_summary_results))
    
    # Display Boostrap Curve Results
    _ = plot_bootstrap_curve(bootstrapped_results, 'roc_auc_score', 'roc_curve',
                     xlabel='Number Treated', ylabel='NNT',
                     title='ROC Curve', legend_title='AUROC')

