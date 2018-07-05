from __init__ import sys
from directory_table import get_paths
from cross_val_cost import CVCostSwarm
import pandas
from argparse import ArgumentParser
from logger_factory import get_logger
from lightgbm.sklearn import LGBMClassifier

# For terminal execution
parser = ArgumentParser()
parser.add_argument('-s', help='work station registered in kaggleProjects.directory_table.py')
parser.add_argument('-i', help='iterations of swarm optimisation')
parser.add_argument('-p', help='number of particles to generate')
parser.add_argument('-n', help='name of session')
args = parser.parse_args()


# Import paths
paths = get_paths(station=args.s)
pkl_dir = paths['pkl_dir']
pso_dir, log_dir = paths['pso'], paths['logs']

# Get new logger
logger = get_logger('LgbmPso', log_dir)
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
    'random_state': 123,
    'max_bin': 15,
    'device': 'cpu',
    'save_binary': True,
    'verbose': -1,
    'boosting_type': 'gbdt',
    'objective': 'binary'
}
constant_params_gpu = {
    'random_state': 123,
    'max_bin': 15,
    'device': 'gpu',
    'gpu_use_dp': False,
    'save_binary': True,
    'verbose': -1,
    'boosting_type': 'gbdt',
    'objective': 'binary'
}
swarm_optimizer = CVCostSwarm(
    session_name=args.n,
    estimator=LGBMClassifier(**constant_params_cpu),
    cutoff=0.5,
    eval_metric='roc_auc',
    save_dir=pso_dir,
    folds=5,
    x_data=X,
    y_data=y
)
swarm_optimizer.add_hyperparameter('num_iterations', 175, 275, True)
swarm_optimizer.add_hyperparameter('subsample', 0.6, 1.0)
swarm_optimizer.add_hyperparameter('colsample_bytree', 0.5, 1.0)
swarm_optimizer.add_hyperparameter('min_child_samples', 30, 70, True)
swarm_optimizer.add_hyperparameter('max_depth', 5, 17, True)
swarm_optimizer.add_hyperparameter('learning_rate', 0.001, 0.1)
swarm_optimizer.add_hyperparameter('n_estimators', 1000, 10000, True)
swarm_optimizer.add_hyperparameter('num_leaves', 50, 100, True)
swarm_optimizer.add_hyperparameter('reg_alpha', 1, 2)
swarm_optimizer.add_hyperparameter('reg_lambda', 1, 2)

# Optimise swarm options
logger.info('optimising swarm options ...')
option_space = {
    'c1': (1, 5),
    'c2': (6, 10),
    'w': (2, 5),
    'k': (11, 15),
    'p': 1
}
swarm_optimizer.optimise_options(
    n_particles=1,
    iterations=3,
    n_samples=1,
    option_space=option_space
)

logger.info('started running swarm ...')
swarm_optimizer.run(
    particles=int(args.p),
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    print_step=1,
    iters=int(args.i)
)
logger.info('# successfully completed swarm optimisation for {} shutting down #'.format(args.n))

# Todo: A continous system that combine the previous best several seen pos and new random pos into an init pos; also add
# automatically the default parameters for entered hypers then add that to position and perhaps others that are randomly
# generated from a normal dist around the default parameters
# Todo: Implement other estimators
# Todo: Random grid search for pso options
# Todo: improve memory efficiency i,e, immediately delete vars after usage

