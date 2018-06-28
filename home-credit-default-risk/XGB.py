import pandas as pd
from xgboost.sklearn import XGBClassifier
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
    # Constant XGBClassifier params
    threads = 1
    objective = 'binary:logistic'
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
        xgbp = {
            'n_estimators': 100, 'objective': objective, 'scale_pos_weight': 1, 'tree_method': 'gpu_hist'
        }
        classifier = XGBClassifier(**xgbp)
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
            'max_depth': list(range(3, 10)),
            'min_child_weight': list(range(1, 6))
        },
        {
            'gamma': [i / 10.0 for i in range(0, 5)]
        },
        {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        },
        {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
        },
        {
            'learning_rate': [pow(1/10, i) for i in range(1, 4)]
        }
    ]
    optimal_params = sequential_search(param_group1, True)

    gamma, subsample = optimal_params['gamma'], optimal_params['subsample']
    colsample_bytree, reg_alpha = optimal_params['colsample_bytree'], optimal_params['reg_alpha']
    learning_rate = optimal_params["learning_rate"]

    param_group2 = [
        {
            'gamma': refined_lower_decimal(gamma)
        },
        {
            'subsample': refined_lower_decimal(subsample),
            'colsample_bytree': refined_lower_decimal(colsample_bytree)
        },
        {
            'reg_alpha': refined_lower_decimal(reg_alpha)
        },
        {
            'learning_rate': refined_lower_decimal(learning_rate)
        }
    ]

    optimal_params = sequential_search(param_group2, True)
    print(optimal_params)
