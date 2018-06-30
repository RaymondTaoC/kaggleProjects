import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

'''
# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.7592969729234852
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    SelectPercentile(score_func=f_classif, percentile=39),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=0.25, min_samples_leaf=11,
                               min_samples_split=6, n_estimators=100, subsample=0.7000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
'''

data_dir = r"C:/Users/dean_/.kaggle/competitions/home-credit-default-risk/RawData"
pickle_dir = r'C:\Users\dean_\.kaggle\competitions\home-credit-default-risk\NN-pkl'
training_df = np.load(pickle_dir + r'\train_df.npy')
predicting_df = np.load(pickle_dir + r'\predict_df.npy')
target = np.load(pickle_dir + r'\target.npy')

meta_df = pd.read_pickle(pickle_dir + r'\meta_df.pkl')

app_train_df = pd.read_csv(data_dir + '/application_train.csv')
len_train = len(app_train_df)
del app_train_df


# Score on submission set was:0.504
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    SelectPercentile(score_func=f_classif, percentile=39),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=0.25, min_samples_leaf=11,
                               min_samples_split=6, n_estimators=100, subsample=0.7000000000000001)
)

exported_pipeline.fit(training_df, target[:, 1])
results = exported_pipeline.predict(predicting_df)

out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': results})
out_df.to_csv('tpot_submission.csv', index=False)

