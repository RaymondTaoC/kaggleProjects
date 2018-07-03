from __init__ import sys
from h2o.estimators import H2OXGBoostEstimator
from h2o.exceptions import H2OResponseError
from h2o.grid import H2OGridSearch
from searcher import BaseRandomSearch
from platform import platform
import h2o
import argparse
from pandas import read_pickle
from directory_table import get_paths
from logger_factory import get_logger
from importlib import import_module


class H2oRandSearch(BaseRandomSearch):

    def __init__(self, estimator, runtime, station, name, eval_metric, search_logger):
        super().__init__(estimator, runtime, station, name, eval_metric, search_logger)

    def search(self, score_cutoff, param_space,
               rand_seed, n_models,
               const_params, cv_folds, training_frame,
               model_directory, predictors, response):
        if ("Windows" in platform()) and (self.estimator == H2OXGBoostEstimator):
            incompatible_message = "Windows currently doesn't support H2OXGBoostEstimator. " \
                                   "No xgboost models will be trained."
            self.logger.info(incompatible_message)
            print(incompatible_message)
            return
        criteria = {
            'strategy': 'RandomDiscrete',
            'max_models': n_models,
            'seed': rand_seed,
            # limit the runtime to 60 minutesS
            'max_runtime_secs': self.max_minutes * 60,
            # early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
            'stopping_rounds': 5,
            'stopping_metric': self.eval_metric,
            'stopping_tolerance': 1e-3
        }
        # Required for H2OStackedEnsembleEstimator
        const_params.update({
            'nfolds': cv_folds,
            'keep_cross_validation_predictions': True,
            'fold_assignment': "Modulo",
            'seed': rand_seed
        })
        grid = H2OGridSearch(model=self.estimator(**const_params),
                             grid_id=self.name + '_grid',
                             hyper_params=param_space,
                             search_criteria=criteria)
        self.logger.info("Training {} models ...".format(self.name))
        # grid.train(x=X, y=Y, nfolds=configuration.CV_FOLDS, seed=rand_seed, training_frame=credit_data)
        try:
            grid.train(x=predictors, y=response, training_frame=training_frame)
        except H2OResponseError:
            self.logger.error('Encountered server error. Skipping ' + self.name)
            return
        self.logger.info("Finished training {} models.".format(self.name))
        # Get the grid results, sorted
        results = grid.get_grid(sort_by=self.eval_metric, decreasing=True)

        for x in results:
            print(get_model_cv_metric(x, self.eval_metric))

        high_scoring = [model for model in results if get_model_cv_metric(model, self.eval_metric) > score_cutoff]
        if not high_scoring:
            self.logger.info('Failed to find models that meet the cut off.')
            return
        self.log_training_results(results=results, search_grid=param_space)
        self.save_model_list(model_lst=high_scoring, seed=rand_seed, directory=model_directory)

    def log_training_results(self, results, search_grid):
        best_mod = results[0]
        worst_model_index = -1
        worst_mod = results[worst_model_index]
        best_mod_found_params = best_found_params(search_grid, best_mod.params)
        logger_entry = \
            """
        {} Grid Search Results of the {} collected:
        \tBest Collected Model {}:\t{}
        \tWorst Collected Model {}:\t{}
        \tBest Model Params (non listed params are left as their default)
        \t{}""".format(self.name, len(results),
                       self.eval_metric,
                       get_model_cv_metric(best_mod, self.eval_metric),
                       self.eval_metric,
                       get_model_cv_metric(worst_mod, self.eval_metric),
                       best_mod_found_params)
        self.logger.info(logger_entry)

    def save_model_list(self, model_lst, directory, seed):
        for model in model_lst:
            score_path = directory + "/{}_({}){}".format(self.name,
                                                         round(get_model_cv_metric(model, self.eval_metric), 4),
                                                         seed)
            h2o.save_model(model=model, path=score_path, force=True)


def best_found_params(param_grid, model_params):
    found_params = {}
    for key in param_grid:
        found_params[key] = model_params[key]
    return found_params


def get_model_cv_metric(model, metric):
    cv_summary = model.cross_validation_metrics_summary().as_data_frame()
    scores = cv_summary.loc[cv_summary[''] == metric]
    return float(scores['mean'][1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='configuration file (python script in Search_Configurations)')
    parser.add_argument('-s', help='work station registered in kaggleProjects.directory_table.py')
    args = parser.parse_args()
    config = import_module('Search_Configurations.' + args.c)

    # Import directories
    paths = get_paths(station=args.s)
    data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
    h2o_cont_dir, log_dir = paths['h2o_cont_search'], paths['logs']
    # Get new logger
    logger = get_logger('H2oContRandSearch', log_dir)

    meta = read_pickle(pkl_dir + '/meta_df.pkl')
    h2o.init(**config.H2O_SETTINGS)
    logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
    credit_data = h2o.upload_file(pkl_dir + "/train_imp_na_df.csv")
    logger.info("Loaded data into cluster")
    # Grid searching parameters
    X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
    Y = 'TARGET'
    credit_data[Y] = credit_data[Y].asfactor()

    data_info = {
        'predictors': X,
        'response': Y,
        'training_frame': credit_data,
        'model_directory': h2o_cont_dir,
    }

    config.GRID_SEARCH_SETTINGS.update(data_info)
    search_engine = H2oRandSearch(station=args.s, search_logger=logger, **config.SEARCH_SETTINGS)
    search_engine.search(**config.GRID_SEARCH_SETTINGS)
    h2o.cluster().shutdown()
