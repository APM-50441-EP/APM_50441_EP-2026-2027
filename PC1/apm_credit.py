"""Shared helpers of the APM_50441_EP labs (Machine Learning 2) on the German credit data.

One verified cost model shared by the labs. The cost model is built step by step in Lab 1, and
the fairness audit is used in Lab 3 (to go further). The labs import them from this module instead of
copying them:

    import apm_credit
    from apm_credit import cost_metric, cost_scorer, cost_scorer_from_proba
    train_prop, custom_weight = apm_credit.set_proportions(y_train, real_prop)

The cost functions use the class weights `custom_weight` computed by `set_proportions`, which
must thus be called once the training set is known.

Importing this module enables the metadata routing of scikit-learn
(`sklearn.set_config(enable_metadata_routing=True)`), which is needed to send `LoanAmount`
to the cost scorers.
"""

__version__ = "2026.1"

import importlib
import sys
from functools import partial

import numpy as np
import polars as pl
import sklearn
import sklearn.metrics
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, count, selection_rate, true_positive_rate, false_positive_rate

sklearn.set_config(enable_metadata_routing=True)


# ---------------------------------------------------------------------------
# Cost model (Lab 1)
# ---------------------------------------------------------------------------

REAL_PROP = {'Risk': .02, 'No Risk': .98}
"""Assumed proportions of the classes in the real world."""

train_prop = None
custom_weight = None


def set_proportions(y_train, real_prop=REAL_PROP):
    """Compute the class proportions of the training set and the importance weights
    `real_prop / train_prop` used by all the cost functions. Returns `(train_prop, custom_weight)`."""
    global train_prop, custom_weight
    train_prop = dict(y_train.value_counts(normalize=True).sort(y_train.name).iter_rows())
    custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
    return train_prop, custom_weight


def _check_proportions():
    if custom_weight is None:
        raise RuntimeError("Call apm_credit.set_proportions(y_train) before using the cost functions.")


loss_rate = .6
gain_rate = .05


def set_costs(loss=.6, gain=.05):
    """Change the coefficients of the cost model: a default costs `5.0 + loss * LoanAmount`, and a
    good loan brings `gain * LoanAmount - 1.0`. Called without arguments, restores the default values."""
    global loss_rate, gain_rate
    loss_rate, gain_rate = loss, gain


def compute_costs(LoanAmount):
    """Cost of each (true class, predicted class) pair, keyed by 'true_predicted'."""
    return({'Risk_No Risk': 5.0 + loss_rate * LoanAmount, 'No Risk_No Risk': 1.0 - gain_rate * LoanAmount,
            'Risk_Risk': 1.0, 'No Risk_Risk': 1.0})


def cost_metric(y_true, y_pred, LoanAmount):
    """Average real-world cost of the predictions (a negative value is a gain)."""
    _check_proportions()
    costs = compute_costs(LoanAmount)
    y_pred = pl.Series(y_pred)
    loss = (y_true=='Risk') * custom_weight['Risk'] *\
                  ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) +\
           (y_true=='No Risk') * custom_weight['No Risk'] *\
                  ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
    return(loss.mean())


cost_scorer = sklearn.metrics.make_scorer(cost_metric, greater_is_better=False).set_score_request(LoanAmount=True)
"""Scorer of `cost_metric` (the opposite of the cost, i.e. the gain); needs `LoanAmount` as metadata."""


def cost_predict_from_proba(proba, LoanAmount):
    """Bayes decision minimizing the expected real-world cost, from the 2-column probabilities
    of ['No Risk', 'Risk']."""
    _check_proportions()
    classes_ = np.array(['No Risk', 'Risk'])
    costs = compute_costs(LoanAmount)
    decision = np.vstack([( proba[:, 0] * custom_weight['No Risk'] * costs['No Risk_No Risk'] + proba[:, 1] * custom_weight['Risk'] * costs['Risk_No Risk']),
                         ( proba[:, 0] * custom_weight['No Risk'] * costs['No Risk_Risk'] + proba[:, 1] * custom_weight['Risk'] * costs['Risk_Risk'])])
    return(pl.Series(classes_[decision.argmin(axis=0)]))


def cost_predict_from_single_proba(proba, LoanAmount):
    """Same decision from the probability of 'Risk' only (a 1-D array)."""
    proba = np.column_stack((1.0-proba, proba))
    return(cost_predict_from_proba(proba, LoanAmount))


def cost_predict(pipeline, X_test, LoanAmount):
    """Bayes decision of a fitted classifier with `predict_proba`."""
    proba = pipeline.predict_proba(X_test)
    return(cost_predict_from_proba(proba, LoanAmount))


def cost_metric_from_proba(y_true, proba, LoanAmount):
    """Cost of the Bayes decision computed from the probability of 'Risk'."""
    y_pred = cost_predict_from_single_proba(proba, LoanAmount)
    return(cost_metric(y_true, y_pred, LoanAmount))


cost_scorer_from_proba = sklearn.metrics.make_scorer(cost_metric_from_proba, response_method="predict_proba", greater_is_better=False).set_score_request(LoanAmount=True)
"""Scorer of the Bayes decision (the gain); needs `LoanAmount` as metadata."""


def compute_cost_weights(y_true, LoanAmount):
    """Sample weights proportional to the extra cost of a wrong decision, normalized to mean 1."""
    _check_proportions()
    costs = compute_costs(LoanAmount)
    weights = ((y_true=='Risk') * custom_weight['Risk'] * (costs['Risk_No Risk']-costs['Risk_Risk'])
               + (y_true=='No Risk') * custom_weight['No Risk'] * (costs['No Risk_Risk']-costs['No Risk_No Risk']))
    return(weights / weights.mean())


# ---------------------------------------------------------------------------
# Fairness audit (Lab 3, to go further)
# ---------------------------------------------------------------------------

def sensitive_features(X):
    """Sensitive attributes of the loans: Sex, ForeignWorker and AgeGroup (25 or less / over 25)."""
    return X.select('Sex', 'ForeignWorker',
                    AgeGroup=pl.when(pl.col('Age') <= 25).then(pl.lit('25 or less')).otherwise(pl.lit('over 25')))


group_metrics = {'count': count,
                 'accuracy': accuracy_score,
                 'approval rate': partial(selection_rate, pos_label='No Risk'),
                 'TPR': partial(true_positive_rate, pos_label='No Risk'),
                 'FPR': partial(false_positive_rate, pos_label='No Risk')}
"""Metrics computed for each group; the favorable outcome is the granting of the loan ('No Risk')."""


def fairness_audit(y_true, y_pred, sensitive, metrics=group_metrics, sample_params=None):
    """Metrics by group for each sensitive attribute, and the demographic parity and equalized odds
    differences. Returns two polars DataFrames `(by_group, gaps)`."""
    by_group, gaps = [], []
    for attribute in sensitive.columns:
        mf = MetricFrame(metrics=metrics, y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
                         sensitive_features=sensitive[attribute].to_numpy(), sample_params=sample_params)
        table = pl.from_pandas(mf.by_group.reset_index(names='group'))
        by_group.append(table.select(pl.lit(attribute).alias('attribute'), 'group', *metrics)
                             .with_columns(pl.col('count').cast(pl.Int64)))
        difference = mf.difference()
        gaps.append({'attribute': attribute,
                     'demographic parity difference': difference['approval rate'],
                     'equalized odds difference': max(difference['TPR'], difference['FPR'])})
    return pl.concat(by_group), pl.DataFrame(gaps)


def gain_metric(y_true, y_pred, LoanAmount):
    """Average gain (opposite of `cost_metric`), usable in a `MetricFrame` with `sample_params`."""
    return -cost_metric(pl.Series(y_true), pl.Series(y_pred), pl.Series(LoanAmount))


metrics_with_gain = group_metrics | {'gain': gain_metric}
"""`group_metrics` plus the gain, which needs `sample_params={'gain': {'LoanAmount': ...}}`."""


def fairness_summary(y_true, decisions, audits, LoanAmount):
    """One row per model: its gain and its demographic parity and equalized odds differences.
    `decisions` maps model names to predictions and `audits` to the results of `fairness_audit`."""
    return pl.DataFrame([{'model': name,
                          'gain': -cost_metric(y_true, pl.Series(y_pred), LoanAmount),
                          **{'parity ' + row['attribute']: row['demographic parity difference'] for row in audits[name][1].iter_rows(named=True)},
                          **{'odds ' + row['attribute']: row['equalized odds difference'] for row in audits[name][1].iter_rows(named=True)}}
                         for name, y_pred in decisions.items()])


# ---------------------------------------------------------------------------
# Parallel computations
# ---------------------------------------------------------------------------

# With n_jobs > 1, scikit-learn sends the scorers to worker processes, which would import this
# module again and lose the proportions set by set_proportions. Pickling the module by value sends
# its current state along with its functions.
for _pickler in ['joblib.externals.cloudpickle', 'cloudpickle']:
    try:
        importlib.import_module(_pickler).register_pickle_by_value(sys.modules[__name__])
    except (ImportError, AttributeError):
        pass
