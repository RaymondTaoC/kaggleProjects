from h2o.estimators import H2OGradientBoostingEstimator


H2O_SETTINGS = {
    "min_mem_size_GB": 5,
    "nthreads": 3,
    "enable_assertions": False
}
SEARCH_SETTINGS = {
    'estimator': H2OGradientBoostingEstimator,
    'runtime': 5,  # minutes
    'name': 'GBM_TEST',
    'eval_metric': 'auc'
}
train_rows = 307511
GRID_SEARCH_SETTINGS = {
    'cv_folds': 2,
    'score_cutoff': 0.5,
    'n_models': 2,
    'rand_seed': 123,
    'const_params': {
        'score_tree_interval': 10,  # makes early stopping reproducible (it depends on the scoring interval)
        'ntrees': 1,  # more trees  ais better if the learning rate is small enough; use "more than enough" trees
        # since we have early stopping
        'learn_rate': 0.05,  # smaller learning rate is better; since we have learning_rate_annealing, we can afford
        # to start with a bigger learning rate
        'learn_rate_annealing': 0.99,  # learning rate annealing: learning_rate shrinks by 1% after every tree
        # (use 1.00 to disable, but then lower the learning_rate)
        'max_runtime_secs': 3600,  # early stopping based on timeout (no single model should take more than 1 hour;
        # modify as needed)
        # early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
        'stopping_rounds': 5, 'stopping_tolerance': 1e-4, 'stopping_metric': "AUC"
    },

    'param_space': {
        'max_depth': list(range(5, 29)),
        'sample_rate': [0.2 + (0.01 * i) for i in range(1, 81)],
        'col_sample_rate': [0.2 + (0.01 * i) for i in range(1, 81)],
        'col_sample_rate_per_tree': [0.2 + (0.01 * i) for i in range(1, 81)],
        'col_sample_rate_change_per_level': [0.9 + (0.01 * i) for i in range(1, 20)],
        'min_rows': [pow(2, i) for i in range(17)],
        'nbins': [pow(2, i) for i in range(4, 11)],
        'nbins_cats': [pow(2, i) for i in range(4, 13)],
        'min_split_improvement': [0, 1e-8, 1e-6, 1e-4],
        'histogram_type': ["UniformAdaptive", "QuantilesGlobal", "RoundRobin"],
    }
}