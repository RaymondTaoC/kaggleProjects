from __future__ import print_function
import h2o
from kaggleProjects.directory_table import get_paths
from kaggleProjects.logger_factory import get_logger
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.grid.grid_search import H2OGridSearch
import platform
# import kaggleProjects.DefaultRisk.H2oRandSearch.config as config
from h2o.exceptions import H2OResponseError
import pandas as pd
from importlib import import_module


def best_found_params(param_grid, model_params):
    found_params = {}
    for key in param_grid:
        found_params[key] = model_params[key]
    return found_params


def get_model_cv_metric(model, metric):
    cv_summary = model.cross_validation_metrics_summary().as_data_frame()
    scores = cv_summary.loc[cv_summary[''] == metric]
    return scores['mean'][1]


def log_training_results(_logger, results, worst_model_index, search_grid, name, scring_metric):
    best_mod = results[0]
    worst_model_index = min(worst_model_index, len(results) - 1)
    worst_mod = results[worst_model_index]
    best_mod_found_params = best_found_params(search_grid, best_mod.params)
    logger_entry = \
        """
    {} Grid Search Results of the {} collected:
    \tBest Collected Model {}:\t{}
    \tWorst Collected Model {}:\t{}
    \tBest Model Params (non listed params are left as their default)
    \t{}""".format(name, len(results),
                   scring_metric,
                   get_model_cv_metric(best_mod, scring_metric),
                   scring_metric,
                   get_model_cv_metric(worst_mod, scring_metric),
                   best_mod_found_params)
    _logger.info(logger_entry)


def save_model_list(model_lst, name, seed, directory):
    for model in model_lst:
        score_path = directory + "/{}_({}){}".format(name, round(model.auc(), 4), seed)
        h2o.save_model(model=model, path=score_path, force=True)


def random_h2o_model_search(name, param_space, estimator, rand_seed,
                            n_models, save_num, const_params, configuration,
                            logger, X, Y, training_frame, model_directory):
    if ("Windows" in platform.platform()) and (estimator == H2OXGBoostEstimator):
        incompatible_message = "Windows currently doesn't support H2OXGBoostEstimator. " \
                               "No xgboost models will be trained."
        logger.info(incompatible_message)
        print(incompatible_message)
        return
    assert save_num < n_models, "Cannot save more models than the number of models to be trained."
    criteria = {
        'strategy': 'RandomDiscrete',
        'max_models': n_models,
        'seed': rand_seed,
        # limit the runtime to 60 minutesS
        'max_runtime_secs': configuration.MAX_RUNTIME_MINUTES * 60,
        # early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
        'stopping_rounds': 5,
        'stopping_metric': "AUC",
        'stopping_tolerance': 1e-3
    }
    # Required for H2OStackedEnsembleEstimator
    const_params.update({
        'nfolds': configuration.CV_FOLDS,
        'keep_cross_validation_predictions': True,
        'fold_assignment': "Modulo",
        'seed': rand_seed
    })
    grid = H2OGridSearch(model=estimator(**const_params),
                         grid_id=name + '_grid',
                         hyper_params=param_space,
                         search_criteria=criteria)
    logger.info("Training {} models ...".format(name))
    grid.train(x=X, y=Y, nfolds=configuration.CV_FOLDS, seed=rand_seed, training_frame=training_frame)
    # try:
    #     grid.train(x=X, y=Y, training_frame=training_frame)
    # except H2OResponseError:
    #     logger.error('Encountered server error. Skipping ' + name)
    #     return
    logger.info("Finished training {} models.".format(name))
    # Get the grid results, sorted
    results = grid.get_grid(sort_by='auc', decreasing=True)
    log_training_results(logger, results=results,
                         search_grid=param_space, name=name,
                         worst_model_index=save_num,
                         scring_metric='auc')
    save_model_list(name=name, model_lst=results[:save_num], seed=rand_seed, directory=model_directory)


def run(config_path, work_station):
    config = import_module(config_path)

    # Import directories
    paths = get_paths(station=work_station)
    data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
    h2o_rand_dir, log_dir = config.SAVE_DIR, paths['logs']
    # Get new logger
    logger = get_logger('H2oRandSearch', log_dir)

    meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')
    h2o.init(**config.H2O_INIT_SETTINGS)
    logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
    credit_data = h2o.upload_file(pkl_dir + "/train_imp_na_df.csv")
    logger.info("Loaded data into cluster")
    # Grid searching parameters
    X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
    Y = 'TARGET'
    credit_data[Y] = credit_data[Y].asfactor()

    data_info = {
        'X': X,
        'Y': Y,
        'training_frame': credit_data,
        'model_directory': h2o_rand_dir,
        'logger': logger,
        'configuration': config
    }
    del meta

    if config.INCLUDE_GBM:
        config.GBM_SETTINGS.update(data_info)
        random_h2o_model_search(**config.GBM_SETTINGS)
    if config.INCLUDE_XGB:
        config.XGB_SETTINGS.update(data_info)
        random_h2o_model_search(**config.XGB_SETTINGS)
    if config.INCLUDE_DEEP:
        config.DEEP_SETTINGS.update(data_info)
        random_h2o_model_search(**config.DEEP_SETTINGS)
    if config.INCLUDE_RF:
        config.RF_SETTINGS.update(data_info)
        random_h2o_model_search(**config.RF_SETTINGS)
    if config.INCLUDE_NAIVE_BAYES:
        config.NAI_BAYES_SETTINGS.update(data_info)
        random_h2o_model_search(**config.NAI_BAYES_SETTINGS)
    if config.INCLUDE_GLM:
        config.GLM_SETTINGS.update(data_info)
        random_h2o_model_search(**config.GLM_SETTINGS)
    logger.info("Completed search. Shutting down cluster " + str(h2o.cluster().cloud_name))
    h2o.cluster().shutdown()
