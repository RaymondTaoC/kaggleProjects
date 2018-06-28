from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from kaggleProjects.directory_table import get_paths
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4


def fix_constant_params(p_dict):
    param_map = {}
    for key in p_dict:
        val = p_dict[key]
        if not isinstance(val, list):
            param_map[key] = [val]
        else:
            param_map[key] = val
    return param_map


def refined_lower_decimal(optimal):
    dp = decimal_place(optimal)
    refined = []
    for i in range(1, 20):
        _eval = (optimal - pow(10, dp)) + (i * pow(10, dp - 1))
        if _eval > 0:
            refined += [_eval]
    return refined


def decimal_place(value):
    if value > 1:
        value = 1 / value
    i = 0
    while not (value * pow(10, i)) in range(10):
        i += 1
    if value > 1:
        return i
    return -i


def sequential_search(param_group, verbose=False):
    # Constant LightGBM params
    objective = 'binary'
    eval_metric = 'auc'
    seed = 123

    # Constant grid search params
    scoring = 'roc_auc'
    n_jobs = 3
    iid = False
    folds = 5

    optimals = {}
    i = 0
    for param_set in param_group:
        test_params = fix_constant_params({**optimals, **param_set})
        classifier = LGBMClassifier(objective=objective, eval_metric=eval_metric, random_state=seed)
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


if __name__ == "__main__":
    data_dir, pkl_dir = get_paths(station='Windows')
    app_train_df = pd.read_csv(data_dir + '/application_train.csv')
    len_train = len(app_train_df)
    del app_train_df

    meta = pd.read_pickle(pkl_dir + r'\meta_df.pkl')
    train = pd.read_csv(pkl_dir + '/train.csv', nrows=len_train)
    target = 'TARGET'
    print(train.columns)
    predictors = list(set(train.columns) - set(meta.columns) - {target, 'Unnamed: 0'})
    del meta

    param_group1 = [
        {
            'n_estimators': list(range(20, 101, 10))
        },
        {
            'max_depth': list(range(5, 16)),
            'min_data_per_group': list([20, 100, 250, 500, 750, 1000])
        },
        {
            'min_data_in_leaf': list(range(30, 101, 10))
        },
        {
            'feature_fraction': [i / 10.0 for i in range(5, 11)]
        },
        {
            'bagging_fraction': [i / 10.0 for i in range(5, 11)]
        },
        {
            'learning_rate': [pow(1/10, i) for i in range(1, 4)]
        }
    ]
    optimal_params = sequential_search(param_group1, True)
    print(optimal_params)
