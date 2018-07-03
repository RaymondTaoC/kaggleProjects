from abc import abstractmethod


class BaseRandomSearch:
    def __init__(self, estimator, runtime, station, name, eval_metric, logger):
        self.estimator = estimator
        self.max_minutes = runtime
        self.station = station
        self.name = name
        self.eval_metric = eval_metric
        self.logger = logger

    @abstractmethod
    def search(self, score_cutoff, param_space, estimator,
               rand_seed, n_models,
               const_params, cv_folds, training_frame,
               model_directory, predictors, response):
        pass

    @abstractmethod
    def log_training_results(self, results, search_grid):
        pass

    @abstractmethod
    def save_model_list(self, model_lst, directory, seed):
        pass
