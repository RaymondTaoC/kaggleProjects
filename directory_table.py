SUBGRAPH_PATH = '/home/user/Documents/Kaggle/CreditDefaultRisk/'
WINDOWS_PATH = 'C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/'

_file_systems = {
    "Subgraph": {
        'path': SUBGRAPH_PATH,
        'h2o_rand_search': SUBGRAPH_PATH + 'H2oRandSearchModels',
        'data_dir': SUBGRAPH_PATH + 'RawDataSets',
        'pkl_dir': SUBGRAPH_PATH + 'EngineeredData',
        'logs': SUBGRAPH_PATH + 'Logs'
    },
    "Windows": {
        'path': WINDOWS_PATH,
        'h2o_rand_search': WINDOWS_PATH + 'H2oRandSearchModels',
        'data_dir': WINDOWS_PATH + 'RawData',
        'pkl_dir': WINDOWS_PATH + 'NN-pkl',
        'logs': WINDOWS_PATH + 'Logs'
    }
}


def get_paths(station):
    return _file_systems[station]
