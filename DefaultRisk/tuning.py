from __future__ import print_function
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
h2o.init(min_mem_size_GB=5, nthreads=-1)

data = h2o.upload_file(r"C:\Users\dean_\.kaggle\competitions\home-credit-default-risk\RawData\application_train.csv")

seed = 123
folds = 10
X = list(set(data.columns) - {'SK_ID_CURR'})
Y = 'TARGET'

data[Y] = data[Y].asfactor()

# Tune Deep model
params = {'hidden': [[100, 100, 100, 100], [200, 200, 200, 200]],
          'categorical_encoding': ["one_hot_internal", "one_hot_explicit"],
          'activation': ["rectifier", "maxout"]}
base_deep = H2ODeepLearningEstimator(stopping_metric='auc', epochs=500)
grid = H2OGridSearch(base_deep, hyper_params=params)
grid.train(x=X, y=Y, training_frame=data, nfolds=folds)
# Get the grid results, sorted
grid_res = grid.get_grid(sort_by='auc', decreasing=True)
print(grid_res)
# Get best top model based on rmsle # best so far hidd=[100, 100, 100, 100], epochs=500, activation='tanh'
# with cv rmsle 0.15401701720888492
best_mod = grid_res.models[0]
print('Best Deep:')
print(best_mod.params)

# Tune GBM model
gbm_params = {'learn_rate': [0.1, 0.01, 0.001, 0.001], 'sample_rate': [i * 0.1 for i in range(5, 11)],
              'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
gbm_grid = H2OGradientBoostingEstimator(ntrees=10000, max_depth=17, learn_rate_annealing=0.99, seed=123,
                                        score_tree_interval=10, stopping_rounds=5, stopping_metric='rmsle',
                                        stopping_tolerance=1e-4)

grid = H2OGridSearch(gbm_grid, gbm_params, grid_id='pussy', search_criteria={'strategy': 'Cartesian'})
grid.train(x=X, y=Y, training_frame=housing, nfolds=10)
# Get the grid results, sorted
print('gbm ranking')
grid_res = grid.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best GBM model # best so far max_depth: 17, learn_rate: 0.1, sample_rate: 0,5, col_sample_rate: 0.4
#  with rmsle 0.12970952986414705
best_mod = grid_res.models[-1]
print('Best GBM:')
print(best_mod.params)


# Tune RF model
rf_params = {'max_depth': [75, 80, 90, 100, 150]}
rf_grid = H2ORandomForestEstimator(ntrees=500)
grid = H2OGridSearch(rf_grid, hyper_params=rf_params)
grid.train(x=X, y=Y, training_frame=data, nfolds=10)
# Get the grid results, sorted
print('rf ranking')
grid_res = grid.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best RF model # best so far ntrees: 500, max_depth: 80 with RMSLE: 0.1381765158317337
best_mod = grid_res.models[-1]
print('Best RF:')
print(best_mod.params)


# Tune GLM model
glm_params = {'lambda': [0, 1, 0.5, 0.1, 0.001, 0.0001, 0.00001] + [x / 10 for x in range(1, 11)], 'alpha': [x / 10 for x in range(1, 11)]}
glm = H2OGeneralizedLinearEstimator(family='gaussian')
grid = H2OGridSearch(model=glm, hyper_params=glm_params, search_criteria={'strategy': 'Cartesian'})
grid.train(list(set(housing.columns) - {'C1', 'Ids'}), y='SalePrice', training_frame=housing, nfolds=10)
# Get the grid results, sorted
print('GLM ranking')
grid_res = grid.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best GLM model # best so far alpha: 0.4, lambda: 1 with RMSLE: 0.1472874227501013
best_mod = grid_res.models[-1]
print('Best GLM:')
print(best_mod.params)
print('Best Lambda: {}'.format(best_mod.actual_params['lambda']))


# Tune XBG model
#xgb_params = {'max_depth': [x for x in range(3, 11, 2)],
#              'min_child_weight': [x for x in range(1, 6, 2)]}
xgb_params = {'learn_rate': [0.01, 0.001],
              'ntrees': [5000, 10000]}
xgb = H2OXGBoostEstimator(learn_rate=0.01, ntrees=5000, colsample_bytree=0.81,
                          subsample=0.89, gamma=0.4, max_depth=4, min_child_weight=4)
grid = H2OGridSearch(model=xgb, hyper_params=xgb_params, search_criteria={'strategy': 'Cartesian'})
grid.train(list(set(housing.columns) - {'C1', 'Ids'}), y='SalePrice', training_frame=housing, nfolds=10)
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


# Tune PCA model
pca_params = {'transform': ['None', 'Standardize', 'Normalize', 'Demean', 'Descale'],
              'k': [i for i in range(1, 10)],
              'max_iterations': [pow(10, i) for i in range(1, 6)]}
pca = H2OPrincipalComponentAnalysisEstimator(seed=seed)
grid = H2OGridSearch(model=pca, hyper_params=pca_params, search_criteria={'strategy': 'Cartesian'})
grid.train(list(set(housing.columns) - {'C1', 'Ids'}), y='SalePrice', training_frame=housing, nfolds=10)
# Get the grid results, sorted
print('PCA ranking')
grid_res = grid.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best PCA model # best so far learn_rate:0.01, ntrees:5000, colsample_bytree:0.81, subsample:0.88999,
# gamma:0.4, max_depth:4, min_child_weight:4
# with RMSLE: 0.12430670916111265
best_mod = grid_res.models[-1]
print('Best PCA:')
print(best_mod.params)



# Random set of GBM
gbm_params2 = {'learn_rate': [i * 0.01 for i in range(1, 11)],
               'max_depth': [x for x in range(2, 11)],
               'sample_rate': [i * 0.1 for i in range(5, 11)],
               'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 100, 'seed': 3211}
gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid2',
                          hyper_params=gbm_params2,
                          search_criteria=search_criteria)
gbm_grid2.train(x=X, y=Y, nfolds=folds, seed=seed, ntrees=100, training_frame=housing)
# Get the grid results, sorted
print('GBM random grid search (RGS) ranking')
grid_res = gbm_grid2.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best GBM model # best so far learn_rate=0.09, max_depth=6, sample_rate=0.9, col_sample_rate=0.4
# with RMSLE: 0.12937550279587925
best_mod = grid_res.models[-1]
print('Best GBM:')
print(best_mod.params)


# Random set of PCA
pca_params = {'learn_rate': [i * 0.01 for i in range(1, 11)],
               'max_depth': [x for x in range(2, 11)],
               'sample_rate': [i * 0.1 for i in range(5, 11)],
               'col_sample_rate': [i * 0.1 for i in range(1, 11)]}
search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 100, 'seed': 3211}
gbm_grid2 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid2',
                          hyper_params=pca_params,
                          search_criteria=search_criteria)
gbm_grid2.train(x=X, y=Y, nfolds=folds, seed=seed, ntrees=100, training_frame=housing)
# Get the grid results, sorted
print('GBM random grid search (RGS) ranking')
grid_res = gbm_grid2.get_grid(sort_by='rmsle', decreasing=True)
print(grid_res)
# Get best XGB model # best so far learn_rate=0.09, max_depth=6, sample_rate=0.9, col_sample_rate=0.4
# with RMSLE: 0.12937550279587925
best_mod = grid_res.models[-1]
print('Best GBM:')
print(best_mod.params)
