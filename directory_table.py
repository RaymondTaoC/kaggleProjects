SUBGRAPH_PATH = '/home/user/Documents/Kaggle/CreditDefaultRisk/'
WINDOWS_PATH = 'C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/'

_file_systems = {
    "Subgraph": {
        'path': SUBGRAPH_PATH,
        'h2o_rand_search': SUBGRAPH_PATH + 'H2oRandSearchModels',
        'data_dir': SUBGRAPH_PATH + 'RawDataSets',
        'pkl_dir': SUBGRAPH_PATH + 'EngineeredData',
        'logs': SUBGRAPH_PATH + 'Logs',
        'submissions': SUBGRAPH_PATH + 'Submissions',
        'h2o_cont_search': SUBGRAPH_PATH + 'H2oContSearchModels',
	'lgbm_cont_search': SUBGRAPH_PATH + 'LgbmContSearchModels',
        'temp': SUBGRAPH_PATH + 'temp'
    },
    "Windows": {
        'path': WINDOWS_PATH,
        'h2o_rand_search': WINDOWS_PATH + 'H2oRandSearchModels',
        'data_dir': WINDOWS_PATH + 'RawData',
        'pkl_dir': WINDOWS_PATH + 'NN-pkl',
        'logs': WINDOWS_PATH + 'Logs',
        'submissions': WINDOWS_PATH + 'Submissions',
        'h2o_cont_search': WINDOWS_PATH + 'H2oContSearchModels',
	'lgbm_cont_search': WINDOWS_PATH + 'LgbmContSearchModels',
        'temp': WINDOWS_PATH + 'temp'
    }
}


def get_paths(station):
    return _file_systems[station]
