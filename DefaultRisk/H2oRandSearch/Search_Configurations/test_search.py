import h2o.estimators as algos

CV_FOLDS = 2
MAX_RUNTIME_MINUTES = 2  # Max search time for each estimator
SAVE_DIR = 'C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/temp'

H2O_INIT_SETTINGS = {
    "min_mem_size_GB": 5,
    "nthreads": 1,
    "enable_assertions": False
}

# Models to include
INCLUDE_GBM = True  # 2 models/hour
INCLUDE_XGB = True
INCLUDE_DEEP = True  # 1 model/hour
INCLUDE_RF = True   # 2 models/hour
INCLUDE_NAIVE_BAYES = False
INCLUDE_GLM = True  # 6 models/hour

# Reference: https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.Rmd
_train_rows = 307511
GBM_SETTINGS = {
    'name': "GBM_TEST",
    'estimator': algos.gbm.H2OGradientBoostingEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,
    'const_params': {
        'score_tree_interval': 10,  # makes early stopping reproducible (it depends on the scoring interval)
        'learn_rate': 0.05,  # smaller learning rate is better; since we have learning_rate_annealing, we can afford
        # to start with a bigger learning rate
        'learn_rate_annealing': 0.99,  # learning rate annealing: learning_rate shrinks by 1% after every tree
        # (use 1.00 to disable, but then lower the learning_rate)
        'max_runtime_secs': 2,
        # early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
        'stopping_rounds': 5, 'stopping_tolerance': 1e-4, 'stopping_metric': "AUC"
    },

    'param_space': {
        'ntrees': list(range(1, 10))
    }
}

XGB_SETTINGS = {
    'name': "XGB_TEST",
    'estimator': algos.xgboost.H2OXGBoostEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,

    'const_params': {
        'max_runtime_secs': 2,
        'seed': 123,
        'distribution': 'bernoulli'
    },

    'param_space': {
        'ntrees': list(range(1, 10))
    }
}

DEEP_SETTINGS = {
    'name': "DEEP_TEST",
    'estimator': algos.deeplearning.H2ODeepLearningEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,

    'const_params': {
        'adaptive_rate': False,
        'rate': 0.01,
        'nesterov_accelerated_gradient': True,
        'distribution': 'bernoulli',
        'seed': 123,
        'stopping_metric': 'auc',
        'max_runtime_secs': 3,
        'mini_batch_size': 300,
        'epochs': 40
    },

    'param_space': {
        'hidden': [[200, 200], [200, 200, 200], [250, 240, 230]],
        'input_dropout_ratio': [0, 0.1, 0.2],
    }
}

RF_SETTINGS = {
    'name': "RAND_FOREST",
    'estimator': algos.random_forest.H2ORandomForestEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,

    'const_params': {
        'stopping_metric': 'auc',
        'stopping_rounds': 3,
        'stopping_tolerance': 1e-2,
        'seed': 123,
        'max_runtime_secs': 3
    },

    'param_space': {
        'nbins_cats': [8, 16, 32]
    }
}

NAI_BAYES_SETTINGS = {
    'name': "Naïve_Bayes",
    'estimator': algos.naive_bayes.H2ONaiveBayesEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,

    'const_params': {
        'max_runtime_secs': 3,
        'seed': 123
    },
    'param_space': {
        'laplace': [0, 1, 2, 3]}
}

GLM_SETTINGS = {
    'name': "Linear-logistic",
    'estimator': algos.glm.H2OGeneralizedLinearEstimator,
    'n_models': 2,
    'save_num': 1,
    'rand_seed': 123,

    'const_params': {
        'family': 'binomial',
        'seed': 123
    },
    'param_space': {
        'alpha': [i / 1000 for i in range(0, 1001, 2)],
    }
}
