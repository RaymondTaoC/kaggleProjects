from __future__ import print_function
import h2o
import directory_table
import numpy as np
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator

_, data_path = directory_table.get_paths("Windows")

h2o.init(min_mem_size_GB=5, nthreads=-1)
data = h2o.upload_file(data_path)

seed = 123
folds = 10
X = list(set(data.columns) - {'SK_ID_CURR'})
Y = 'TARGET'

data[Y] = data[Y].asfactor()


def tune_xgboost(params):

    # Tune XBG model
    #xgb_params = {'max_depth': [x for x in range(3, 11, 2)],
    #              'min_child_weight': [x for x in  range(1, 6, 2)]}
    xgb_params = {'learn_rate': [0.01, 0.001],
                  'ntrees': [5000, 10000]}
    xgb = H2OXGBoostEstimator()
    grid = H2OGridSearch(model=xgb, hyper_params=params, search_criteria={'strategy': 'Cartesian'})
    grid.train(x=X, y=Y, training_frame=data, nfolds=folds)
    # Get the grid results, sorted
    print('XGB ranking')
    grid_res = grid.get_grid(sort_by='rmsle', decreasing=True)
    print(grid_res)
    # Get best XGB model # best so far learn_rate:0.01, ntrees:5000, colsample_bytree:0.81, subsample:0.88999,
    # gamma:0.4, max_depth:4, min_child_weight:4
    # with RMSLE: 0.12430670916111265
    best_mod = grid_res.models[-1]
    print('Best XGB:')
    print(best_mod.params)
