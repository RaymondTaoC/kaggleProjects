from sklearn.model_selection import cross_val_score
from lightgbm.sklearn import LGBMClassifier
from kaggleProjects.directory_table import get_paths
import numpy as np
import pandas

hyper_parameters = [
    'num_leaves',
    'reg_lambda',
    'min_data_in_leaf',
    'n_estimators',
    'random_state',
    'max_depth',
    'reg_alpha',
    'bagging_fraction',
    'feature_fraction',
    'boosting_type',
    'learning_rate'
]

parameter_lower_bound = [
    10,  # num_leaves
    # objective
    # reg_lambda
    # min_data_in_leaf
    # n_estimators
    # random_state
    # max_depth
    # reg_alpha
    # bagging_fraction
    # feature_fraction
    # boosting_type
    # learning_rate
]


parameter_upper_bound = [
    # num_leaves
    # objective
    # reg_lambda
    # min_data_in_leaf
    # n_estimators
    # random_state
    # max_depth
    # reg_alpha
    # bagging_fraction
    # feature_fraction
    # boosting_type
    # learning_rate
]

# Import directories
paths = get_paths(station='Subgraph')
# get some data
data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
lgbm_cont_dir, log_dir = paths['lgbm_cont_search'], paths['logs']

X = pandas.read_csv(pkl_dir + "/train_imp_na_df.csv")
y = X["TARGET"].values
X = X.drop("TARGET", axis=1)

meta = pandas.read_pickle(pkl_dir + '/meta_df.pkl')
for x in set(meta.columns):
    if x in set(X.columns):
        X = X.drop(x, axis=1)
del meta

X = X.values

const_params = {
    'objective'
}

integer_param = {'num_leaves', 'min_data_in_leaf'}


def lightgbm_cost(particle, hyper_params=hyper_parameters,
                  scoring='roc_auc', integer_params=integer_param):
    """

    :param scoring:
    :param particle:
    :param hyper_params:
    :return:

    >>> p = [25, 1, 0.1]
    >>> print(lightgbm_cost(p, ['num_boost_round', 'n_estimators', 'learning_rate']))
    0.41650322815485197
    """
    fit_params = {}
    i = 0
    for param in hyper_params:
        if param in integer_params:
            fit_params[param] = int(particle[i])
        else:
            fit_params[param] = particle[i]
        i += 1
    clf = LGBMClassifier(random_state=123, **fit_params)
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring=scoring)
    return np.average(scores)


def lgbm_swarm_cost(particles):
    """Higher-level method to calculate the score of each particle.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = particles.shape[0]
    j = [lightgbm_cost(particles[i]) for i in range(n_particles)]
    return np.array(j)
