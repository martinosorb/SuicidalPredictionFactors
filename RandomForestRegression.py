import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# #### PARAMETERS - CAN BE SET #### #
FILE = "dataset_cleaned.csv"
N_FOLDS = 20  # number of train/test splits
TEST_SIZE = 0.2  # What fraction of the dataset is saved for testing
COL_TARGET = "SPS"  # Which column we're trying to predict


# Data loading
data = pd.read_csv(FILE)

# Definition of the model which will be used below: this is the most important bit
# see the scikit-learn documentation for the meaning of these hyperparameters
model = RandomForestRegressor(
    n_jobs=4,
    n_estimators=100,
    criterion='mse',
    max_features='sqrt',
    max_depth=10,
    min_samples_split=15
)


# #### MODEL FITTING #### #
# allocate variables
scores_test = np.empty(N_FOLDS)
scores_train = np.empty(N_FOLDS)
importances = np.empty([N_FOLDS, len(data.columns) - 1])

for iteration in range(N_FOLDS):
    # Selection of train/test split and of target column
    train_df, test_df = train_test_split(data, shuffle=True,
                                         test_size=TEST_SIZE)
    # removing the feature we're trying to predict...
    train_features = train_df.drop(COL_TARGET, 1)
    # ... and putting it into another variable
    train_labels = train_df[COL_TARGET]
    # same as above, for the test set
    test_features = test_df.drop(COL_TARGET, 1)
    test_labels = test_df[COL_TARGET]

    # Fitting and predictions
    model.fit(train_features, train_labels)

    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Evaluation
    scores_train[iteration] = model.score(train_features, train_labels)
    scores_test[iteration] = model.score(test_features, test_labels)

    # Importance of variables
    importances[iteration] = model.feature_importances_
