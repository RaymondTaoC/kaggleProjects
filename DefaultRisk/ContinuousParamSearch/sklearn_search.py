from __future__ import print_function
from __init__ import sys
import numpy as np

from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from importlib import import_module
from logger_factory import get_logger
from directory_table import get_paths
import argparse
import pandas
from searcher import BaseRandomSearch
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-c', help='configuration file (python script in Search_Configurations)')
parser.add_argument('-s', help='work station registered in kaggleProjects.directory_table.py')
args = parser.parse_args()
config = import_module('Search_Configurations.' + args.c)


# Utility function to report best scores
def report(results, n_top=3):
    report_str = ""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            top_model = \
                """
                Model with rank: {0}
                Mean validation score: {1:.3f} (std: {2:.3f})
                Parameters: {3}
                """.format(i,
                           results['mean_test_score'][candidate],
                           results['std_test_score'][candidate],
                           results['params'][candidate])
            report_str += top_model
    return report_str


def best_scoring_models(current_best_models, new_results, score_cutoff):
    from copy import deepcopy
    best_models = deepcopy(current_best_models)
    score_map = {}
    batch_count = len(new_results['rank_test_score'])
    for i in range(1, batch_count):
        candidates = np.flatnonzero(new_results['rank_test_score'] == i)
        for candidate in candidates:
            score_map[new_results['mean_test_score'][candidate]] = new_results['params'][candidate]
    best_models.update(score_map)
    keys = list(best_models.keys())
    keys.sort(reverse=True)
    new_best = {}
    for key in keys:
        if key > score_cutoff:
            new_best[key] = best_models[key]
    return new_best


class SklRandSearch(BaseRandomSearch):
    def __init__(self, estimator, station, name, eval_metric, _logger, search_metric='binary_logloss'):
        super().__init__(estimator, runtime=None, station=station, name=name, eval_metric=eval_metric, logger=_logger)
        self.metric = search_metric

    def search(self, score_cutoff, param_space, rand_seed, n_models, cv_folds, training_frame,
               model_directory, response, predictors=None, const_params=None):
        # Load previous best
        best_models = self._import_best_models(model_directory)
        best_models_count = len(best_models)
        self.logger.info("loaded the previously found {} models at the {} auc level".format(best_models_count,
                                                                                            score_cutoff))

        # run randomized search
        rand_searcher = RandomizedSearchCV(
            param_distributions=param_space,
            scoring=self.eval_metric,
            cv=cv_folds,
            estimator=self.estimator,
            random_state=rand_seed,
            n_iter=n_models
        )
        _start = time()
        #  rand_searcher.fit(X, y, eval_metric='binary_logloss')
        self.logger.info("started training ...")
        rand_searcher.fit(training_frame, response, eval_metric=self.metric)
        self.logger.info("randomizedSearchCV took %.2f seconds for %d candidates"
                         " parameter settings." % ((time() - _start), config.SEARCH_SETTINGS['n_models']))
        self._log_training_results(rand_searcher.cv_results_)
        updated_best_models = best_scoring_models(best_models, rand_searcher.cv_results_, score_cutoff=score_cutoff)
        self._save_models(updated_best_models, model_directory)
        self.logger.info("saved {} new models".format(len(updated_best_models) - best_models_count))

    def _import_best_models(self, model_dir):
        path = '{}/best_{}.pkl'.format(model_dir, self.name)
        my_file = Path(path)
        if my_file.is_file():
            with open(path, 'rb') as handle:
                return pickle.load(handle)
        else:
            self._save_models({}, model_dir)
            return {}

    def _log_training_results(self, results, search_grid=None):
        self.logger.info(report(results))

    def _save_models(self, models, directory):
        with open('{}/best_{}.pkl'.format(directory, self.name), 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # Import directories
    paths = get_paths(station=args.s)
    # get some data
    data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
    lgbm_cont_dir, log_dir = paths['lgbm_cont_search'], paths['logs']
    # Get new logger
    logger = get_logger('LGBMContRandSearch', log_dir)

    logger.info("########## Started new {} session ##########".format(config.CORE_SETTINGS['name']))
    logger.info("loading data...")
    X = pandas.read_csv(pkl_dir + "/train_imp_na_df.csv")
    y = X["TARGET"].values
    X = X.drop("TARGET", axis=1)

    meta = pandas.read_pickle(pkl_dir + '/meta_df.pkl')
    for x in set(meta.columns):
        if x in set(X.columns):
            X = X.drop(x, axis=1)
    del meta

    X = X.values

    random_trainer = SklRandSearch(station=args.s, _logger=logger, **config.CORE_SETTINGS)
    random_trainer.search(training_frame=X, response=y, model_directory=lgbm_cont_dir, **config.SEARCH_SETTINGS)
    logger.info("########## Ended new {} session ##########".format(config.CORE_SETTINGS['name']))
