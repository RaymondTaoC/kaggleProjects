from __future__ import print_function
import h2o
from kaggleProjects.directory_table import get_paths
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
import logging
import platform
import kaggleProjects.DefaultRisk.H2oRandSearch.config as config
from h2o.exceptions import H2OResponseError
import pandas as pd

# Import directories
paths = get_paths(station='Windows')
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


def random_h2o_model_search(name, param_space, estimator, rand_seed, cv_folds,
                            n_models, save_num):
    if ("Windows" in platform.platform()) and (estimator == H2OXGBoostEstimator):
        incompatible_message = "Windows currently doesn't support H2OXGBoostEstimator. " \
                               "No xgboost models will be trained."
        logger.info(incompatible_message)
        print(incompatible_message)
        return
    assert save_num < n_models, "Cannot save more models than the number of models to be trained."
    criteria = {'strategy': 'RandomDiscrete', 'max_models': n_models, 'seed': rand_seed}
    grid = H2OGridSearch(model=estimator,
                         grid_id=name + '_grid',
                         hyper_params=param_space,
                         search_criteria=criteria)
    logger.info("Training {} models ...".format(name))
    try:
        grid.train(x=X, y=Y, nfolds=cv_folds, seed=rand_seed, training_frame=credit_data)
    except H2OResponseError:
        logger.error('Encountered server error. Skipping ' + name)
        return
    logger.info("Finished training {} models.".format(name))
    # Get the grid results, sorted
    results = grid.get_grid(sort_by='auc',
                            decreasing=True
                            )
    log_training_results(logger, results=results,
                         search_grid=param_space, name=name,
                         worst_model_index=save_num)
    save_model_list(name=name, model_lst=results[:save_num])


if __name__ == "__main__":
    meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')

    h2o.init(min_mem_size_GB=5, nthreads=1)
    logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
    credit_data = h2o.upload_file(pkl_dir + "/train_imp_na_df.csv")
    logger.info("Loaded data into cluster")

    # Grid searching parameters
    X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
    Y = 'TARGET'
    credit_data[Y] = credit_data[Y].asfactor()

    if config.INCLUDE_PCA:
        random_h2o_model_search(**config.PCA_SETTINGS)

    if config.INCLUDE_GBM:
        random_h2o_model_search(**config.GBM_SETTINGS)

    if config.INCLUDE_XGB:
        random_h2o_model_search(**config.XGB_SETTINGS)
    logger.info("Completed search. Shutting down cluster " + str(h2o.cluster().cloud_name))
h2o.cluster().shutdown()
