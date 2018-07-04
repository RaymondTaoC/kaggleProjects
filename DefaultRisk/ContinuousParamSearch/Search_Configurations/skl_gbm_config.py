from lightgbm.sklearn import LGBMClassifier


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
        'bagging_fraction': [6 + (x/10) for x in range(41)],
        'feature_fraction': [5 + (x/10) for x in range(51)],
        'min_data_in_leaf': list(range(30, 71)),
        'max_depth': list(range(5, 16)),
        'learning_rate': [0.005],
        'n_estimators': [1000 * x for x in range(1, 11)],
        'num_leaves': [10 * x for x in range(1, 16)],
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['binary'],
        'random_state': [501],  # Updated from 'seed'
        'reg_alpha': [1 + (x/10) for x in range(10)],
        'reg_lambda': [1 + (x/10) for x in range(10)]
    }
}
