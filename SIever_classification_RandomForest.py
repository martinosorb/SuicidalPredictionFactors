from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from utils import plot_importances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# #### PARAMETERS - CAN BE SET #### #
FILE = "dataset_cleaned.csv"
N_FOLDS = 20  # number of train/test splits
TEST_SIZE = 0.2  # What fraction of the dataset is saved for testing
COL_TARGET = "SI_ever"  # Which column we're trying to predict
COL_DROP = ["SPS"]  # we want to remove this


# Data loading
data = pd.read_csv(FILE)
data.drop(COL_DROP, 1, inplace=True)

# Definition of the model which will be used below: this is the most important bit
# see the scikit-learn documentation for the meaning of these hyperparameters
model = RandomForestClassifier(
    n_jobs=3,
    n_estimators=200,
    max_features='sqrt',
    class_weight='balanced',
    criterion='entropy'
)


# #### MODEL FITTING #### #
# allocate variables
accuracies, precisions, recalls, specificities, aucs = np.empty((5, N_FOLDS))
# fprs, tprs = [], []
importances = np.empty([N_FOLDS, len(data.columns) - 1])

for iteration in range(N_FOLDS):
    # Selection of train/test split and of target column
    train_df, test_df = train_test_split(data, shuffle=True, test_size=TEST_SIZE)

    train_features = train_df.drop(COL_TARGET, 1)
    train_labels = pd.get_dummies(
        train_df[COL_TARGET], drop_first=True, prefix=COL_TARGET).iloc[:, 0]

    test_features = test_df.drop(COL_TARGET, 1)
    test_labels = pd.get_dummies(
        test_df[COL_TARGET], drop_first=True, prefix=COL_TARGET).iloc[:, 0]

    # Fitting and predictions
    model.fit(train_features, train_labels)
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    test_pred_proba = model.predict_proba(test_features)[:, 1]

    # Evaluation
    # In binary classification, the count of true negatives is C00,
    # false negatives is C10, true positives C11 is and false positives is C01.
    c = confusion_matrix(test_labels, test_predictions)
    accuracies[iteration] = (c[0, 0] + c[1, 1]) / c.sum()
    precisions[iteration] = c[1, 1] / c[:, 1].sum()
    recalls[iteration] = c[1, 1] / c[1, :].sum()
    specificities[iteration] = c[0, 0] / (c[0, 1] + c[0, 0])

    # Compute Receiver operated characteristic (ROC) and Area under curve (AUC)
    # false_positive_rate, true_positive_rate, _ = roc_curve(
    #     test_labels, test_pred_proba)
    aucs[iteration] = roc_auc_score(test_labels, test_pred_proba)
    # fprs.append(false_positive_rate)
    # tprs.append(true_positive_rate)

    # Importance of variables
    importances[iteration] = model.feature_importances_


print(model)
print("Median accuracy:", np.median(accuracies))
print("Median precision:", np.median(precisions))
print("Median recall:", np.median(recalls))
print("Median specificity:", np.median(specificities))
print("Median AUC:", np.median(aucs))

# #### PLOT FIT QUALITY #### #
plt.title("Prediction metrics, RF classification")
plt.grid(axis='y')
bp = plt.boxplot([accuracies, precisions, recalls, specificities, aucs], widths=0.5)
plt.setp(bp['medians'], color='C4', linewidth=2)
plt.ylim([0.5, 1])
plt.xticks(range(1, 6), ["Accuracy", "Precision", "Recall", "Specificity", "AUC"])
plt.ylabel("Score")

plt.savefig("RF_SIever_metrics.pdf")


# #### PLOT IMPORTANCES #### #
plt.figure(figsize=(4, 5.5))
plt.title("Classification of 'SI_ever': features")
plt.xlabel("Feature importance in random forest")

plot_importances(importances, train_features.columns, color='C4')

plt.savefig("RF_SIever_features.pdf", bbox_inches="tight")
