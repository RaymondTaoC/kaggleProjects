import pandas as pd
from xgboost.sklearn import XGBClassifier
from kaggleProjects.directory_table import get_paths
from kaggleProjects.SequentialParamSearch import sequential_search, refined_lower_decimal


if __name__ == "__main__":
    data_dir, pkl_dir = get_paths(station='Subgraph')
    app_train_df = pd.read_csv(data_dir + '/application_train.csv')
    len_train = len(app_train_df)
    del app_train_df

    meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')
    train = pd.read_csv(pkl_dir + '/train.csv', nrows=len_train)
    target = 'TARGET'
    print(train.columns)
    predictors = list(set(train.columns) - set(meta.columns) - {target, 'Unnamed: 0'})
    del meta

    xgb_params = {
        'n_estimators': 100, 'objective': 'binary:logistic', 'scale_pos_weight': 1,
        'tree_method': 'gpu_hist'  # Comment out this line if xgb was not complied with gpu support.
    }

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
    optimal_params = sequential_search(param_group=param_group1,
                                       classifier_algo=XGBClassifier,
                                       classifier_algo_param_dict=xgb_params,
                                       train=train,
                                       predictors=predictors,
                                       target=target,
                                       scoring='roc_auc',
                                       n_jobs=3,
                                       verbose=True)

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

    optimal_params = sequential_search(param_group=param_group2,
                                       classifier_algo=XGBClassifier,
                                       classifier_algo_param_dict=xgb_params,
                                       train=train,
                                       predictors=predictors,
                                       target=target,
                                       scoring='roc_auc',
                                       n_jobs=3,
                                       verbose=True)
    print(optimal_params)
