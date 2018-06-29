import pandas as pd
from sklearn.model_selection import GridSearchCV  # Performing grid search


def _fix_constant_params(p_dict):
    param_map = {}
    for key in p_dict:
        val = p_dict[key]
        if not isinstance(val, list):
            param_map[key] = [val]
        else:
            param_map[key] = val
    return param_map


def refined_lower_decimal(optimal):
    dp = _decimal_place(optimal)
    refined = []
    for i in range(1, 20):
        _eval = (optimal - pow(10, dp)) + (i * pow(10, dp - 1))
        if _eval > 0:
            refined += [_eval]
    return refined


def _decimal_place(value):
    if value > 1:
        value = 1 / value
    i = 0
    while not (value * pow(10, i)) in range(10):
        i += 1
    if value > 1:
        return i
    return -i


def sequential_search(classifier_algo, classifier_algo_param_dict, param_group,
                      train, predictors, target, scoring,
                      n_jobs=1, folds=5, verbose=False):
    assert isinstance(train, pd.DataFrame), "Parameter 'train' must be a pandas data-frame."

    # Constant grid search params
    iid = False

    optimals = {}
    i = 0
    for param_set in param_group:
        test_params = _fix_constant_params({**optimals, **param_set})
        classifier = classifier_algo(**classifier_algo_param_dict)
        grid_search = GridSearchCV(estimator=classifier, param_grid=test_params, scoring=scoring,
                                   n_jobs=n_jobs, iid=iid, cv=folds, verbose=verbose)
        grid_search.fit(train[predictors], train[target])
        optimals = {**optimals, **grid_search.best_params_}

        if verbose:
            print("First grid search yields:\nBest Params:\n\t{}\nBest Score\n\t{}" \
                  .format(grid_search.best_params_, grid_search.best_score_))

            print("Iteration: {}\nTest Prams:\t{}\nOptimal:\t{}\nBest:\t{}\n\n"\
                  .format(i, test_params, optimals, grid_search.best_params_))
        del classifier, grid_search
        i += 1
    return optimals
