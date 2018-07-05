from sklearn.model_selection import cross_val_score
from lightgbm.sklearn import LGBMClassifier
from kaggleProjects.directory_table import get_paths
import numpy as np
import pandas
from pyswarms.single import GlobalBestPSO

# Import directories
paths = get_paths(station='Subgraph')
# get some data
data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
pso_dir, log_dir = paths['temp'], paths['logs']

X = pandas.read_csv(pkl_dir + "/train_imp_na_df.csv")
y = X["TARGET"].values
X = X.drop("TARGET", axis=1)

meta = pandas.read_pickle(pkl_dir + '/meta_df.pkl')
for x in set(meta.columns):
    if x in set(X.columns):
        X = X.drop(x, axis=1)
del meta

X = X.values


class Lightgbm_Pso:
    def __init__(self, session_name, cutoff, eval_metric, save_dir, init_position=None):
        self.name = session_name
        self.save_dir = save_dir
        self.cutoff = cutoff
        self.position = init_position
        self.eval_metric = eval_metric

        self.param_minimums = []
        self.param_maximums = []

        self.hyperparameters = []  # needs to be ordered
        self.integer_params = set()
        self.const_params = []

    def add_hyperparameter(self, name, lower_bound, upper_bound, is_int=False):
        self.hyperparameters += [name]
        self.param_minimums += [lower_bound]
        self.param_maximums += [upper_bound]
        if is_int:
            self.integer_params.add(name)

    def particle_cost_func(self, particle):
        fit_params = {}
        i = 0
        for param in self.hyperparameters:
            if param in self.integer_params:
                fit_params[param] = int(particle[i])
            else:
                fit_params[param] = particle[i]
            i += 1
        clf = LGBMClassifier(random_state=123,
                             max_bin=15,
                             device='gpu',
                             gpu_use_dp=False,
                             save_binary=True,
                             verbose=-1,
                             boosting_type='gbdt',
                             objective='binary',
                             **fit_params)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring=self.eval_metric)
        return 1 - np.average(scores)

    def cost_func(self, particles):
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
        j = [self.particle_cost_func(particles[i]) for i in range(n_particles)]
        return np.array(j)

    def run(self, particles, options, print_step=100, iters=1000, verbose=3):
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Call instance of PSO
        optimizer = GlobalBestPSO(n_particles=particles,
                                  dimensions=len(self.hyperparameters),
                                  options=options,
                                  bounds=(np.array(self.param_minimums), np.array(self.param_maximums)),
                                  init_pos=self.position)

        # Perform optimization
        best_cost, best_position = optimizer.optimize(self.cost_func, print_step=print_step,
                                                      iters=iters, verbose=verbose)

        if 1 - best_cost > self.cutoff:
            self.save_position(best_cost, best_position)
        else:  # doesnt score well enough
            print('not high')

    def save_position(self, score, position):
        import os
        pos = pandas.DataFrame(data=[position], columns=self.hyperparameters)
        # Convert all int params to int
        for int_param in self.integer_params:
            pos[int_param] = pos[int_param].astype(int)

        pos['Score'] = 1 - score
        os.mkdir("{}/{}".format(self.save_dir, self.name))
        pos.to_csv("{}/{}/{}.csv".format(self.save_dir, self.name, score), index=True)


if __name__ == "__main__":
    test = Lightgbm_Pso(save_dir=pso_dir, session_name='test_run', cutoff=0.5, eval_metric='roc_auc')

    test.add_hyperparameter('num_iterations', 175, 275, True)
    test.add_hyperparameter('subsample', 0.6, 1.0)
    test.add_hyperparameter('colsample_bytree', 0.5, 1.0)
    test.add_hyperparameter('min_child_samples', 30, 70, True)
    test.add_hyperparameter('max_depth', 5, 17, True)
    test.add_hyperparameter('learning_rate', 0.001, 0.1)
    test.add_hyperparameter('n_estimators', 1000, 10000, True)
    test.add_hyperparameter('num_leaves', 50, 100, True)
    test.add_hyperparameter('reg_alpha', 1, 2)
    test.add_hyperparameter('reg_lambda', 1, 2)
    test.run(particles=2,
             options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
             print_step=1,
             iters=1)
