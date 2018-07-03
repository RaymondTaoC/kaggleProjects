import h2o.estimators as algos
from math import log
from numpy import linspace

WORK_STATION = 'Windows'
TRAIN_ROWS = 307511
CV_FOLDS = 5
MAX_RUNTIME_MINUTES = 2  # Max search time for each estimator

H2O_INIT_SETTINGS = {
    "min_mem_size_GB": 5,
    "nthreads": 3,
    "enable_assertions": False
}

# Models to include
INCLUDE_GBM = False  # 2 models/hour
INCLUDE_XGB = False
INCLUDE_DEEP = False  # 1 model/hour
INCLUDE_RF = False   # 2 models/hour
INCLUDE_NAIVE_BAYES = False
INCLUDE_GLM = True  # 6 models/hour

# Reference: https://github.com/h2oai/h2o-3/blob/master/h2o-docs/src/product/tutorials/gbm/gbmTuning.Rmd
GBM_SETTINGS = {
    'name': "GBM",
    'estimator': algos.gbm.H2OGradientBoostingEstimator,
    'n_models': 100,
    'save_num': 10,
    'rand_seed': 123,
    'const_params': {
        'score_tree_interval': 10,  # makes early stopping reproducible (it depends on the scoring interval)
        'ntrees': 10000,  # more trees is better if the learning rate is small enough; use "more than enough" trees
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
        'min_rows': [pow(2, i) for i in range(int(log(TRAIN_ROWS, 2)))],
        'nbins': [pow(2, i) for i in range(4, 11)],
        'nbins_cats': [pow(2, i) for i in range(4, 13)],
        'min_split_improvement': [0, 1e-8, 1e-6, 1e-4],
        'histogram_type': ["UniformAdaptive", "QuantilesGlobal", "RoundRobin"],
    }
}

# References: [1] https://i.stack.imgur.com/9GgQK.jpg
#             [2] https://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters
XGB_SETTINGS = {
    'name': "XGB",
    'estimator': algos.xgboost.H2OXGBoostEstimator,
    'n_models': 100,
    'save_num': 10,
    'rand_seed': 123,

    'const_params': {
        'ntrees': 1000,
        'max_runtime_secs': 3600,
        'seed': 123,
        'distribution': 'bernoulli'
    },

    'param_space': {
        'learn_rate': [i / 1000 for i in range(2, 11)],
        'sample_rate': [0.5 + (i / 100) for i in range(0, 51)],
        'col_sample_rate': [0.4 + (i / 100) for i in range(0, 61, 2)],
        'max_abs_leafnode_pred': [0, 1, 2, 3],
        'max_depth': list(range(4, 11))
    }
}

DEEP_SETTINGS = {
    'name': "DEEP",
    'estimator': algos.deeplearning.H2ODeepLearningEstimator,
    'n_models': 50,
    'save_num': 5,
    'rand_seed': 123,

    'const_params': {
        'adaptive_rate': False,
        'rate': 0.01,
        'nesterov_accelerated_gradient': True,
        'distribution': 'bernoulli',
        'seed': 123,
        'stopping_metric': 'auc',
        'max_runtime_secs': 3600,
        'mini_batch_size': 300,
        'epochs': 40
    },

    'param_space': {
        'hidden': [[200, 200], [200, 200, 200], [250, 240, 230]],
        'input_dropout_ratio': [0, 0.1, 0.2],
        'loss': ['Quadratic', 'ModifiedHuber', 'CrossEntropy'],
        'activation': ['tanh_with_dropout', 'rectifier_with_dropout', 'rectifier']}
}

# Reference:
# [1] https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# [2] http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/nbins.html
RF_SETTINGS = {
    'name': "RAND_FOREST",
    'estimator': algos.random_forest.H2ORandomForestEstimator,
    'n_models': 100,
    'save_num': 10,
    'rand_seed': 123,

    'const_params': {
        'stopping_metric': 'auc',
        'stopping_rounds': 3,
        'stopping_tolerance': 1e-2,
        'seed': 123,
        'max_runtime_secs': 3600
    },

    'param_space': {
        'histogram_type': ['UniformAdaptive', 'Random', 'QuantilesGlobal', 'RoundRobin'],
        # Number of trees in random forest
        'ntrees': [int(x) for x in linspace(start=200, stop=2000, num=10)],
        # Maximum number of levels in tree
        'max_depth': [int(x) for x in linspace(10, 110, num=11)],
        # Minimum number of samples required at each leaf node
        'min_rows': [1, 2, 4],
        # the number of bins for the histogram to build, then split at the best point.
        'nbins': [16, 32, 64, 128, 256, 512],
        # the number of bins to be included in the histogram and then split at the best point
        'nbins_cats': [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    }
}

NAI_BAYES_SETTINGS = {
    'name': "Naïve_Bayes",
    'estimator': algos.naive_bayes.H2ONaiveBayesEstimator,
    'n_models': 4,
    'save_num': 2,
    'rand_seed': 123,

    'const_params': {
        'max_runtime_secs': 10 * 60,
        'seed': 123
    },
    'param_space': {
        'laplace': [0, 1, 2, 3]}
}

GLM_SETTINGS = {
    'name': "Linear-logistic",
    'estimator': algos.glm.H2OGeneralizedLinearEstimator,
    'n_models': 100,
    'save_num': 10,
    'rand_seed': 123,

    'const_params': {
        'family': 'binomial',
        'lambda_search': True,
        'remove_collinear_columns': True,
        'seed': 123
    },
    'param_space': {
        'alpha': [i / 1000 for i in range(0, 1001, 2)],
    }
}
