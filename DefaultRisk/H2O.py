from __future__ import print_function
import h2o
from kaggleProjects.directory_table import get_paths
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
from kaggleProjects.DefaultRisk.DataWrapper import HomeCreditDataWrapper
import logging
import pandas as pd

# Import directories
paths = get_paths(station='Subgraph')
data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
h2o_rand_dir, log_dir = paths['h2o_rand_search'], paths['logs']

# Implemented logging since the H2O console outputs and logging gives too much information.
logger = logging.getLogger('H2O_random_search')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(log_dir + '/H2oRandSearch.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def best_found_params(param_grid, model_params):
    found_params = {}
    for key in param_grid:
        found_params[key] = model_params[key]
    return found_params


def log_training_results(_logger, results, worst_model_index, search_grid, name):
    best_mod = results[0]
    worst_mod = results[worst_model_index]
    best_mod_found_params = best_found_params(search_grid, best_mod.params)
    logger_entry = \
        """
    {} Grid Search Results:
    \tBest Collected Model Score:\t{}
    \tWorst Collected Model Score:\t{}
    \tBest Model Params (non listed params are left as their default)
    \t{}""".format(name,
                   best_mod.auc(),
                   worst_mod.auc(),
                   best_mod_found_params)
    _logger.info(logger_entry)


def save_model_list(model_lst, name):
    for model in model_lst:
        score_path = h2o_rand_dir + "/{}_({})".format(name, round(model.auc(), 5))
        h2o.save_model(model=model, path=score_path, force=True)


data = HomeCreditDataWrapper(data_dir, pkl_dir, n_rows=100)
data.manage_na()
data.data_set.to_csv(index=False, path_or_buf=pkl_dir + "/imputed_train.csv")

meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')

h2o.init(min_mem_size_GB=5, nthreads=1)
logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
credit_data = h2o.upload_file(pkl_dir + "/imputed_train.csv")
logger.info("Loaded data into cluster")

# Grid searching parameters
seed = 123
folds = 5
X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
Y = 'TARGET'
credit_data[Y] = credit_data[Y].asfactor()

# Model collection params
PCA_MODELS = 3
PCA_MODELS_TO_COLLECT = 2

GBM_MODELS = 3
GBM_MODELS_TO_COLLECT = 2

XGB_MODELS = 3
XGB_MODELS_TO_COLLECT = 2

best_models = []
"""
# Random set of PCA
search_space = {'learn_rate': [i * 0.001 for i in range(1, 101)],
                'max_depth': [x for x in range(2, 11)],
                'sample_rate': [i * 0.1 for i in range(5, 11)],
                'col_sample_rate': [i * 0.1 for i in range(1, 11)],
                'ntrees': [1]}
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': PCA_MODELS, 'seed': seed}
pca_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                         grid_id='pca_grid',
                         hyper_params=search_space,
                         search_criteria=search_criteria)
logger.info("Training PCA models ...")
pca_grid.train(x=X, y=Y, nfolds=folds, seed=seed, training_frame=credit_data)
logger.info("Finished training PCA models.")
# Get the grid results, sorted
pca_grid_res = pca_grid.get_grid(sort_by='auc',
                                 decreasing=True
                                 )
log_training_results(logger, results=pca_grid_res,
                     search_grid=search_space, name="PCA",
                     worst_model_index=PCA_MODELS_TO_COLLECT)
best_models += pca_grid_res[:PCA_MODELS_TO_COLLECT]
save_model_list(name="PCA", model_lst=pca_grid_res[:PCA_MODELS_TO_COLLECT])
del search_space, search_criteria, pca_grid, pca_grid_res

# Random set of GBM
search_space = {'learn_rate': [i * 0.01 for i in range(1, 11)],
                'max_depth': [x for x in range(2, 11)],
                'sample_rate': [i * 0.1 for i in range(5, 11)],
                'col_sample_rate': [i * 0.1 for i in range(1, 11)],
                'ntrees': [1]}
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': GBM_MODELS, 'seed': seed}
gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                         grid_id='gbm_grid',
                         hyper_params=search_space,
                         search_criteria=search_criteria)
logger.info("Training GBM models ...")
gbm_grid.train(x=X, y=Y, nfolds=folds, seed=seed, training_frame=credit_data)
logger.info("Finished training GBM models.")
# Get the grid results, sorted
gbm_grid_res = gbm_grid.get_grid(sort_by='auc', decreasing=True)
log_training_results(logger, results=gbm_grid_res,
                     search_grid=search_space, name="GBM",
                     worst_model_index=GBM_MODELS_TO_COLLECT)
best_models += gbm_grid_res[:GBM_MODELS_TO_COLLECT]
save_model_list(name="GBM", model_lst=gbm_grid_res[:GBM_MODELS_TO_COLLECT])
del search_space, search_criteria, gbm_grid, gbm_grid_res
"""

# Random set of XGB
search_space = {'learn_rate': [0.01, 0.001],
                'ntrees': [1]}
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': XGB_MODELS, 'seed': seed}
xgb_grid = H2OGridSearch(model=H2OXGBoostEstimator(dmatrix_type="sparse"),
                         grid_id='xgb_grid',
                         hyper_params=search_space,
                         search_criteria=search_criteria)
logger.info("Training XGB models ...")
xgb_grid.train(x=X, y=Y, nfolds=folds, seed=seed, training_frame=credit_data)
logger.info("Finished training xgb models.")
# Get the grid results, sorted
xgb_grid_res = xgb_grid.get_grid(sort_by='auc', decreasing=True)
log_training_results(logger, results=xgb_grid_res,
                     search_grid=search_space, name="xgb",
                     worst_model_index=XGB_MODELS_TO_COLLECT)
best_models += xgb_grid_res[:XGB_MODELS_TO_COLLECT]
save_model_list(name="xgb", model_lst=xgb_grid_res[:XGB_MODELS_TO_COLLECT])
del search_space, search_criteria, xgb_grid, xgb_grid_res
h2o.cluster().shutdown()
