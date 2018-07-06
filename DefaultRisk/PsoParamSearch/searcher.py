import numpy as np
from pyswarms.single import GlobalBestPSO
from abc import abstractmethod
from pyswarms.utils.search import RandomSearch, GridSearch

default_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}


class PsoParamSearch:
    def __init__(self, session_name, cutoff,
                 eval_metric, save_dir, init_position=None, options=default_options):
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

        self.options = options

    def add_hyperparameter(self, name, lower_bound, upper_bound, is_int=False):
        self.hyperparameters += [name]
        self.param_minimums += [lower_bound]
        self.param_maximums += [upper_bound]
        if is_int:
            self.integer_params.add(name)

    @abstractmethod
    def particle_cost_func(self, particle):
        pass

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

    def run(self, particles, print_step=100, iters=1000, verbose=3):
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Call instance of PSO
        optimizer = GlobalBestPSO(n_particles=particles,
                                  dimensions=len(self.hyperparameters),
                                  options=self.options,
                                  bounds=(np.array(self.param_minimums), np.array(self.param_maximums)),
                                  init_pos=self.position)

        # Perform optimization
        best_cost, best_position = optimizer.optimize(self.cost_func, print_step=print_step,
                                                      iters=iters, verbose=verbose)

        if best_cost < self.cutoff:
            self.save_position(best_cost, best_position)
        else:  # doesnt score well enough
            print('not high')

    @abstractmethod
    def save_position(self, score, position):
        pass

    def optimise_options(self, n_particles, iterations, n_samples, option_space, search_type='random'):
        assert search_type in {'random', 'grid'}
        if search_type == 'random':
            g = RandomSearch(GlobalBestPSO, n_particles=n_particles,
                             dimensions=len(self.hyperparameters), options=option_space,
                             objective_func=self.cost_func, iters=iterations,
                             n_selection_iters=n_samples)
            best_score, best_options = g.search()
        else:  # search_type: grid
            g = GridSearch(GlobalBestPSO, n_particles=n_particles,
                           dimensions=len(self.hyperparameters), options=option_space,
                           objective_func=self.cost_func, iters=iterations)
            best_score, best_options = g.search()
        self.options = best_options
        print(best_score)
        print(best_options)
        return best_score
