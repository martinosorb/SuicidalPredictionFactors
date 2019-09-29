import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# #### PARAMETERS - CAN BE SET #### #
FILE = "dataset_cleaned.csv"
TEST_SIZE = 0.2
COL_TARGET = "Lifetime suicidal ideation"  # Which column we're trying to predict
COL_DROP = ["Current suicidal ideation"]  # we want to remove this
C_values = np.logspace(-2.5, 1, 50)

# Data loading
data = pd.read_csv(FILE, index_col="Part_ID")
data.drop(COL_DROP, 1, inplace=True)


importances = np.empty([len(C_values), len(data.columns) - 1])

# choose the model and its parameters. This can be changed and experimented with
# note the L1 penalty, this is essential
model = LogisticRegression(
    solver='saga',
    tol=1e-6,
    max_iter=int(1e6),
    warm_start=False,
    penalty='l1'
)


# We don't do any testing here.
train_df, test_df = train_test_split(data, shuffle=True, test_size=TEST_SIZE)

train_features_prescaling = train_df.drop(COL_TARGET, 1)
train_labels = pd.get_dummies(
    train_df[COL_TARGET], drop_first=True, prefix=COL_TARGET).iloc[:, 0]
test_features_prescaling = test_df.drop(COL_TARGET, 1)
test_labels = pd.get_dummies(
    test_df[COL_TARGET], drop_first=True, prefix=COL_TARGET).iloc[:, 0]

# scaling is caring (necessary to interpret weights as importances)
scaler = StandardScaler(copy=True)
train_features = scaler.fit_transform(train_features_prescaling)
test_features = scaler.transform(test_features_prescaling)

# Fit the model for each value of C, which is the L1 regularisation weight
accuracies, precisions, recalls, specificities = [], [], [], []
for iteration, C in enumerate(C_values):
    # Fitting and predictions
    model.set_params(C=C)
    model.fit(train_features, train_labels)

    train_predictions = model.predict(train_features)

    # Evaluation
    # In binary classification, the count of true negatives is C00,
    # false negatives is C10, true positives C11 is and false positives is C01.
    test_predictions = model.predict(test_features)
    test_pred_proba = model.predict_proba(test_features)[:, 1]
    c = confusion_matrix(test_labels, test_predictions)
    accuracies.append((c[0, 0] + c[1, 1]) / c.sum())
    precisions.append(c[1, 1] / c[:, 1].sum())
    recalls.append(c[1, 1] / c[1, :].sum())
    specificities.append(c[0, 0] / (c[0, 1] + c[0, 0]))

    # I determined the importance of variables for Logistic Regression
    # just using the absolute value of the weights.
    importances[iteration] = model.coef_


fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
plt.sca(ax[0])
plt.title('Feature weight as a function of regularization')
plt.plot(C_values, importances, linewidth=0.7)
for i, varname in enumerate(train_features_prescaling.columns):
    plt.annotate(varname,
                 (C_values[-1] + 1, importances[-1, i] - 0.01),
                 color=f'C{i}', fontsize=7)
plt.xlim([C_values[0] * 0.9, C_values[-1] * 30.0])
plt.xscale('log')
plt.ylabel('Feature weight in logistic regression')

plt.sca(ax[1])
plt.plot(C_values, accuracies, label='Accuracy')
plt.plot(C_values, precisions, label='Precision')
plt.plot(C_values, recalls, label='Recall')
plt.plot(C_values, specificities, label='Specificity')
plt.xscale('log')
plt.ylabel('Metric')
plt.xlabel('L1 regularization factor (stronger to weaker)')
plt.legend()
plt.ylim([0, 1])

plt.savefig("figures/L1RegularizationPath.pdf")
