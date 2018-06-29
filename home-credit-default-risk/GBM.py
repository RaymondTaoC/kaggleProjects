import pandas as pd
from lightgbm import LGBMClassifier
from kaggleProjects.directory_table import get_paths
from kaggleProjects.SequentialParamSearch import sequential_search


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
lgbm_const_params = {
    'objective': 'binary',
    'random_state': 123,
}
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
optimal_params = sequential_search(classifier_algo=LGBMClassifier,
                                   classifier_algo_param_dict=lgbm_const_params,
                                   param_group=param_group1,
                                   train=train,
                                   predictors=predictors,
                                   target=target,
                                   scoring='roc_auc',
                                   n_jobs=3,
                                   verbose=True)
print(optimal_params)
