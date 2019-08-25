import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import plot_importances

# #### PARAMETERS - CAN BE SET #### #
FILE = "dataset_cleaned.csv"
N_FOLDS = 20  # number of train/test splits
TEST_SIZE = 0.2  # What fraction of the dataset is saved for testing
COL_TARGET = "SPS"  # Which column we're trying to predict
COL_DROP = ["SI_ever"]  # we want to remove this


# Data loading
data = pd.read_csv(FILE)
data.drop(COL_DROP, 1, inplace=True)

# Definition of the model which will be used below: this is the most important bit
# see the scikit-learn documentation for the meaning of these hyperparameters
model = RandomForestRegressor(
    n_jobs=4,
    n_estimators=100,
    criterion='mse',
    max_features='sqrt',
    max_depth=10,
    min_samples_split=15  # regularizes a bit
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


print(model)
print("Median scores on training set:", np.median(scores_train))
print("Median scores on test set:", np.median(scores_test))


# #### PLOT FIT QUALITY #### #
plt.title("Regression of SPS by Random Forest")

plt.gca().set_aspect('equal')
plt.plot(train_labels, train_predictions, '+', alpha=.8,
         color='C8', label="Training set")
plt.plot(test_labels, test_predictions, 'x', alpha=.8,
         color='C9', label="Test set")
plt.legend()
plt.xlabel(COL_TARGET + " true value")
plt.ylabel(COL_TARGET + " predicted value")
plt.plot([0, 25], [0, 25], 'k--', alpha=.5)

plt.savefig("RF_SPS_metrics.pdf", bbox_inches="tight")


# #### PLOT IMPORTANCES #### #
plt.figure(figsize=(4, 5.5))
plt.title("Regression of 'SPS': features")
plt.xlabel("Random forest's feature importance")

plot_importances(importances, column_names=train_features.columns, color='C9')

plt.savefig("RF_SPS_features.pdf", bbox_inches="tight")
