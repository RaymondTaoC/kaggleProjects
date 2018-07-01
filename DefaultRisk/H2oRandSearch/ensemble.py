import h2o
import pandas as pd
from os import listdir
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from kaggleProjects.DefaultRisk.H2oRandSearch import config
from kaggleProjects.directory_table import get_paths
from kaggleProjects.logger_factory import get_logger

# Import directories
paths = get_paths(station=config.WORK_STATION)
data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
h2o_rand_dir, log_dir = paths['h2o_rand_search'], paths['logs']
# Initiate logger
logger = get_logger('ensemble', log_dir)


def load_models_from_dir(saved_models_dir):
    models = []
    for model_path in listdir(saved_models_dir):
        models += [h2o.load_model(model_path)]
    return models


ensemble_models = load_models_from_dir(h2o_rand_dir)

h2o.init(**config.H2O_INIT_SETTINGS)
# Load data
meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')
h2o.init(**config.H2O_INIT_SETTINGS)
logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
credit_data = h2o.upload_file(pkl_dir + "/train_imp_na_df.csv")
logger.info("Loaded data into cluster")

# Grid searching parameters
X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
Y = 'TARGET'
credit_data[Y] = credit_data[Y].asfactor()
del meta

# Train a stacked ensemble using the GBM and GLM above
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                       base_models=ensemble_models)
ensemble.train(x=X, y=Y, training_frame=credit_data)
