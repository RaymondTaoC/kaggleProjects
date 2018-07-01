import h2o
import pandas as pd
from os import listdir
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from kaggleProjects.DefaultRisk.H2oRandSearch import config
from kaggleProjects.directory_table import get_paths
from kaggleProjects.logger_factory import get_logger


def load_models_from_dir(saved_models_dir):
    models = []
    for model_dir in listdir(saved_models_dir):
        directory = saved_models_dir + '/' + model_dir + '/'
        models += [h2o.load_model(directory + listdir(directory)[0])]
    return models


# Import directories
paths = get_paths(station=config.WORK_STATION)
pkl_dir, submit_dir = paths['pkl_dir'], paths['submissions']
h2o_rand_dir, log_dir = paths['h2o_rand_search'], paths['logs']
# Initiate logger
logger = get_logger('ensemble', log_dir)

h2o.init(**config.H2O_INIT_SETTINGS)

ensemble_models = load_models_from_dir(h2o_rand_dir)
# Load data
meta = pd.read_pickle(pkl_dir + '/meta_df.pkl')
logger.info("Started new H2o session " + str(h2o.cluster().cloud_name))
credit_data = h2o.upload_file(pkl_dir + "/train_imp_na_df.csv")
predict_me = h2o.upload_file(pkl_dir + '/predict_df.npy')
logger.info("Loaded data into cluster")

# Grid searching parameters
X = set(credit_data.columns) - {'TARGET'} - set(meta.columns)
Y = 'TARGET'
credit_data[Y] = credit_data[Y].asfactor()
del meta

meta_algo_param = {
    'family': 'binomial',
    'lambda_search': True
}

# Train a stacked ensemble using the GBM and GLM above
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                       base_models=ensemble_models,
                                       metalearner_params=meta_algo_param,
                                       metalearner_algorithm='glm')
logger.info('Training Ensemble model...')
ensemble.train(x=X, y=Y, training_frame=credit_data)
logger.info('Completed training model')

logger.info('generating predictions')
submission = ensemble.predict(predict_me)
h2o.export_file(submission, path=submit_dir + '/' + 'ens.csv')
logger.info('wrote submission to file')

logger.info("Completed ensembling. Shutting down cluster " + str(h2o.cluster().cloud_name))
h2o.cluster().shutdown()
