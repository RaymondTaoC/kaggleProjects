from lightgbm.sklearn import LGBMClassifier

CORE_SETTINGS = {
    'estimator': LGBMClassifier(random_state=123,
                                max_bin=15,
                                device='gpu',
                                gpu_use_dp=False,
                                save_binary=True,
                                verbose=-1),
    'name': 'GBM',
    'eval_metric': 'roc_auc'
}

SEARCH_SETTINGS = {
    'score_cutoff': 0.77,
    'cv_folds': 5,
    'n_models': 100,
    'rand_seed': 1341,
    'param_space': {
        'num_iterations': [200, 225, 250],
        'subsample': [(6 + (x / 10)) / 10 for x in range(0, 41, 5)],
        'colsample_bytree': [(5 + (x / 10)) / 10 for x in range(0, 51, 5)],
        'min_child_samples': list(range(30, 61, 3)),
        'max_depth': list(range(5, 16)),
        'learning_rate': [0.001, 0.1, 0.01],
        'n_estimators': [1000 * x for x in range(1, 11)],
        'num_leaves': [10 * x for x in range(5, 11)],
        'boosting_type': ['gbdt', 'dart'],
        'objective': ['binary'],
        'random_state': [53121],  # Updated from 'seed'
        'reg_alpha': [1 + (x / 10) for x in range(10)],
        'reg_lambda': [1 + (x / 10) for x in range(10)]
    }
}
