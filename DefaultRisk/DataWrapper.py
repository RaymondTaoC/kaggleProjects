from pandas import read_csv, read_pickle
from numpy import load
from sklearn.preprocessing import LabelEncoder

# Capture other categorical features not as object data types:
non_obj_categoricals = [
        'FONDKAPREMONT_MODE',
        'HOUR_APPR_PROCESS_START',
        'HOUSETYPE_MODE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'WALLSMATERIAL_MODE',
        'WEEKDAY_APPR_PROCESS_START',
        'NAME_CONTRACT_TYPE_BAVG',
        'WEEKDAY_APPR_PROCESS_START_BAVG',
        'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS',
        'NAME_PAYMENT_TYPE',
        'CODE_REJECT_REASON',
        'NAME_TYPE_SUITE_BAVG',
        'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO',
        'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE',
        'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP',
        'PRODUCT_COMBINATION',
        'NAME_CONTRACT_STATUS_CCAVG',
        'STATUS',
        'NAME_CONTRACT_STATUS_CAVG'
    ]


class HomeCreditDataWrapper:

    def __init__(self, data_dir, pkl_dir, n_rows=None):
        app_train_df = read_csv(data_dir + '/application_train.csv')
        self.n_rows = len(app_train_df)
        del app_train_df
        if n_rows:
            assert 0 < n_rows <= self.n_rows, "Training data set doesn't have {} rows".format(n_rows)
            self.n_rows = n_rows
        meta = read_pickle(pkl_dir + '/meta_df.pkl')
        self.target = 'TARGET'
        self.data_set, self.categorical_feats, self.encoder_dict = process_dataframe(
            input_df=read_csv(pkl_dir + '/train.csv', nrows=self.n_rows))
        self.categorical_feats = self.categorical_feats + non_obj_categoricals
        self.predictors = list(set(self.data_set.columns) - set(meta.columns) - {self.target, 'Unnamed: 0'})
        del meta

        self.data_path = pkl_dir + '/train.csv'

        self.cont_feats_idx = load(pkl_dir + '/cont_feats_idx.npy')
        self.cat_feats_idx = load(pkl_dir + '/cat_feats_idx.npy')

    def manage_na(self, null_thresh=0.8, verbose=False):
        null_counts = self.data_set.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        null_ratios = null_counts / len(self.data_set)

        # Drop columns over x% null
        null_cols = null_ratios[null_ratios > null_thresh].index
        self.data_set.drop(null_cols, axis=1, inplace=True)
        if verbose:
            print('Columns dropped for being over {}% null:'.format(100 * null_thresh))
        for col in null_cols:
            print(col)
            if col in self.categorical_feats:
                self.categorical_feats.pop(col)

        # Fill the rest with the mean (TODO: do something better!)
        # self.data_set.fillna(self.data_set.median(), inplace=True)
        self.data_set.fillna(0, inplace=True)


def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder

    return input_df, categorical_feats.tolist(), encoder_dict
