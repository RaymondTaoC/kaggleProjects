from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import numpy as np

pkl_dir = '/home/user/Documents/Kaggle/CreditDefaultRisk/EngineeredData'
# _, pkl_dir = directory_table.get_paths(station='Subgraph')
training_df = np.load(pkl_dir + r'\train_df.npy')
target = np.load(pkl_dir + r'\target.npy')
predicting_df = np.load(pkl_dir + r'\predict_df.npy')

model = lda()
model.fit(training_df, target[:, 0])
pred = model.predict_proba(predicting_df)

