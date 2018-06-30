"""
TPOT is built on top of several existing Python libraries, including:
NumPy, SciPy, scikit-learn, DEAP,update_checker, tqdm, stopit, pandas
and xgboost.
"""
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from kaggleProjects.directory_table import get_paths

seed = 321

paths = get_paths(station='Subgraph')
data_dir, pkl_dir = paths['data_dir'], paths['pkl_dir']
train_df = np.load(pkl_dir + r'\train_df.npy')
target = np.load(pkl_dir + r'\target.npy')
predicting_df = np.load(pkl_dir + r'\predict_df.npy')

# Create a validation set to check training performance
X_train, X_valid, y_train, y_valid = train_test_split(train_df, target, test_size=0.1, random_state=seed,
                                                      stratify=target[:, 0])

y_train = y_train[:, 0]
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, scoring='roc_auc', n_jobs=-1, random_state=seed)
tpot.fit(X_train, y_train)
print(tpot.score(X_valid, y_valid))
tpot.export('tpot_mnist_pipeline.py')
