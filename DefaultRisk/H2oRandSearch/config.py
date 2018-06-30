from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator


# Models to include
INCLUDE_PCA = True
INCLUDE_GBM = True
INCLUDE_XGB = True

PCA_SETTINGS = {
    'name':         "PCA",
    'estimator':    H2OPrincipalComponentAnalysisEstimator,
    'n_models':     3,
    'save_num':     2,
    'rand_seed':    123,
    'cv_folds':     5,
    'param_space':  {'learn_rate': [0.01, 0.001],
                     'ntrees': [1, 2, 3]}
}

GBM_SETTINGS = {
    'name':         "GBM",
    'estimator':    H2OGradientBoostingEstimator,
    'n_models':     3,
    'save_num':     2,
    'rand_seed':    123,
    'cv_folds':     5,
    'param_space':  {'learn_rate': [0.01, 0.001],
                     'ntrees': [1, 2, 3]}
}

XGB_SETTINGS = {
    'name':         "XGB",
    'estimator':    H2OXGBoostEstimator,
    'n_models':     3,
    'save_num':     2,
    'rand_seed':    123,
    'cv_folds':     5,
    'param_space':  {'learn_rate': [0.01, 0.001],
                     'ntrees': [1, 2, 3]}
}
