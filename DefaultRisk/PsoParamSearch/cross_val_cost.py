from sklearn.model_selection import cross_val_score
import numpy as np
import pandas
from searcher import PsoParamSearch


class CVCostSwarm(PsoParamSearch):
    def __init__(self, session_name, estimator, cutoff,
                 eval_metric, save_dir, folds,
                 x_data, y_data,  init_position=None):
        super().__init__(session_name, cutoff, eval_metric, save_dir, init_position)
        self.cross_val_settings = {
            'estimator': estimator,
            'X': x_data,
            'y': y_data,
            'cv': folds,
            'scoring': eval_metric
        }

    def particle_cost_func(self, particle):
        # the lower the score the better
        fit_params = {}
        i = 0
        for param in self.hyperparameters:
            if param in self.integer_params:
                fit_params[param] = int(particle[i])
            else:
                fit_params[param] = particle[i]
            i += 1

        scores = cross_val_score(**self.cross_val_settings)
        return 1 - np.average(scores)

    def save_position(self, score, position):
        import os
        pos = pandas.DataFrame(data=[position], columns=self.hyperparameters)
        # Convert all int params to int
        for int_param in self.integer_params:
            pos[int_param] = pos[int_param].astype(int)

        pos['Score'] = 1 - score
        os.mkdir("{}/{}".format(self.save_dir, self.name))
        pos.to_csv("{}/{}/{}.csv".format(self.save_dir, self.name, score), index=True)
