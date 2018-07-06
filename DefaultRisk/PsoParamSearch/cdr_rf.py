from sklearn.ensemble import RandomForestClassifier
from __init__ import sys
from directory_table import get_paths
from cross_val_cost import CVCostSwarm
import pandas
from argparse import ArgumentParser
from logger_factory import get_logger


# For terminal execution
parser = ArgumentParser()
parser.add_argument('-s', help='work station registered in kaggleProjects.directory_table.py')
parser.add_argument('-n', help='name of session')
parser.add_argument('-i', help='iterations of swarm optimisation')
parser.add_argument('-p', help='number of particles to generate')
args = parser.parse_args()


# Import paths
paths = get_paths(station=args.s)
pkl_dir = paths['pkl_dir']
pso_dir, log_dir = paths['temp'], paths['logs']

# Get new logger
logger = get_logger('XgbPso', log_dir)
logger.info('# started session {} #'.format(args.n))

# Import data
logger.info('importing data ...')
X = pandas.read_csv(pkl_dir + "/train_imp_na_df.csv")
y = X["TARGET"].values
X = X.drop("TARGET", axis=1)
meta = pandas.read_pickle(pkl_dir + '/meta_df.pkl')
for x in set(meta.columns):
    if x in set(X.columns):
        X = X.drop(x, axis=1)
del meta
X = X.values

# Init swarm optimisation
constant_params_cpu = {
    'class_weight': 'balanced',
    'n_jobs': 1  # = 4,
}
constant_params_gpu = {
    'class_weight': 'balanced',
    'n_jobs': 1  # = 4,
}
swarm_optimizer = CVCostSwarm(
    session_name=args.n,
    estimator=RandomForestClassifier(**constant_params_cpu),
    cutoff=0.5,
    eval_metric='roc_auc',
    save_dir=pso_dir,
    folds=5,
    x_data=X,
    y_data=y
)

#  Add the hyperparameters to be tuned
swarm_optimizer.add_hyperparameter('n_estimators', 100, 2000, True)
swarm_optimizer.add_hyperparameter('max_depth', 1, 100, True)
swarm_optimizer.add_hyperparameter('min_samples_leaf', 1, 5, True)
swarm_optimizer.add_hyperparameter('min_samples_split', 2, 12, True)
swarm_optimizer.add_hyperparameter('max_features', 0.5, 1)


# Optimise swarm options
# logger.info('optimising swarm options ...')
# option_space = {
#     'c1': (1, 5),
#     'c2': (6, 10),
#     'w': (2, 5),
#     'k': (11, 15),
#     'p': 1
# }
# swarm_optimizer.optimise_options(
#     n_particles=1,
#     iterations=3,
#     n_samples=1,
#     option_space=option_space
# )

logger.info('started running swarm ...')
swarm_optimizer.run(
    particles=int(args.p),
    print_step=1,
    iters=int(args.i)
)
logger.info('# successfully completed swarm optimisation for {} shutting down #'.format(args.n))


