import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
from kaggleProjects.directory_table import get_paths
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print
    "\nModel Report"
    print
    "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print
    "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


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
    n_jobs = 4
    iid = False
    folds = 5

    optimals = {}
    i = 0
    for param_set in param_group:
        test_params = fix_constant_params({**optimals, **param_set})
        classifier = XGBClassifier(learning_rate=0.1, n_estimators=1000, mins_child_weight=1,
                                   gamma=0, subsample=0.8, colsample_bytree=0.8,
                                   objective=objective, nthread=threads, scale_pos_weight=1,
                                   seed=seed)
        grid_search = GridSearchCV(estimator=classifier, param_grid=test_params, scoring=scoring,
                                   n_jobs=n_jobs, iid=iid, cv=folds, verbose=verbose)
        grid_search.fit(train[predictors], train[target])
        optimals = {**optimals, **grid_search.best_params_}

        if verbose:
            print("First grid search yields:\nGrid Scores:\n\t{}\nBest Params:\n\t{}\nBest Score\n\t{}" \
                  .format(grid_search.grid_scores_, grid_search.best_params_, grid_search.best_score_))

            print("Iteration: {}\nTest Prams:\t{}\nOptimal:\t{}\nBest:\t{}\n\n"\
                  .format(i, test_params, optimals, grid_search.best_params_))
        del classifier, grid_search
        i += 1
    return optimals


if __name__ == "__main__":
    data_dir, pkl_dir = get_paths(station='Subgraph')
    app_train_df = pd.read_csv(data_dir + '/application_train.csv')
    len_train = len(app_train_df)
    del app_train_df

    meta = pd.read_pickle(pkl_dir + r'\meta_df.pkl')
    target = 'TARGET'
    predictors = list(set(train.columns) - set(meta.columns) - {target})
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
