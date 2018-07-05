from lightgbm.sklearn import LGBMClassifier
from scipy.stats import uniform


def uniform_rand_float(count=1, lower=1, upper=1):
    a = uniform(loc=lower, scale=upper)
    return a.rvs(size=count)


CORE_SETTINGS = {
    'estimator': LGBMClassifier(random_state=123),
    'name': 'GBM',
    'eval_metric': 'roc_auc'
}


SEARCH_SETTINGS = {
    'score_cutoff': 0.75,
    'cv_folds': 5,
    'n_models': 1,
    'rand_seed': 12345,
    'param_space': {
        'bagging_fraction': uniform_rand_float(count=10, lower=6, upper=10) / 10,
        # 'bagging_fraction': [(6 + (x/10)) / 10 for x in range(41, 3)],
        'feature_fraction': [(5 + (x/10)) / 10 for x in range(51, 3)],
        'min_data_in_leaf': list(range(30, 71, 2)),
        'max_depth': list(range(5, 16)),
        'learning_rate': [0.005],
        'n_estimators': [1],  # [1000 * x for x in range(1, 11)],
        'num_leaves': [10 * x for x in range(1, 16)],
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['binary'],
        'random_state': [501],  # Updated from 'seed'
        'reg_alpha': [1 + (x/10) for x in range(10)],
        'reg_lambda': [1 + (x/10) for x in range(10)]
    }
}
