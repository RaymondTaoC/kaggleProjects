def get_paths(station='Subgraph'):
    if station == 'Windows':
        win_data_path = r"C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/RawData"
        win_pkl_dir = r'C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/NN-pkl'
        return win_data_path, win_pkl_dir
    else:
        sub_data_path = "/home/user/Documents/Kaggle/CreditDefaultRisk/RawDataSets"
        sub_pkl_dir = '/home/user/Documents/Kaggle/CreditDefaultRisk/EngineeredData'
        return sub_data_path, sub_pkl_dir
