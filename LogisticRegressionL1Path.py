import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# #### PARAMETERS - CAN BE SET #### #
FILE = "dataset_cleaned.csv"
COL_TARGET = "SI_ever"  # Which column we're trying to predict
COL_DROP = ["SPS"]  # we want to remove this
C_values = np.logspace(-2.5, 1, 50)

# Data loading
data = pd.read_csv(FILE)
data.drop(COL_DROP, 1, inplace=True)


importances = np.empty([len(C_values), len(data.columns) - 1])

# choose the model and its parameters. This can be changed and experimented with
# note the L1 penalty, this is essential
model = LogisticRegression(
    solver='saga',
    tol=1e-6,
    max_iter=int(1e6),
    warm_start=True,
    penalty='l1'
)


# We don't do any testing here.
train_features_prescaling = data.drop(COL_TARGET, 1)
train_labels = pd.get_dummies(
    data[COL_TARGET], drop_first=True, prefix=COL_TARGET).iloc[:, 0]
# scaling is caring (necessary to interpret weights as importances)
scaler = StandardScaler(copy=True)
train_features = scaler.fit_transform(train_features_prescaling)

# Fit the model for each value of C, which is the L1 regularisation weight
for iteration, C in enumerate(C_values):
    # Fitting and predictions
    model.set_params(C=C)
    model.fit(train_features, train_labels)

    train_predictions = model.predict(train_features)

    # I determined the importance of variables for Logistic Regression
    # just using the absolute value of the weights.
    importances[iteration] = model.coef_


plt.figure()
plt.title('Feature weight as a function of regularization')
plt.plot(C_values, importances)
for i, varname in enumerate(train_features_prescaling.columns):
    plt.annotate(varname,
                 (C_values[-1] + 1, importances[-1, i] - 0.01),
                 color=f'C{i}')
plt.xlim([C_values[0] * 0.9, C_values[-1] * 10.0])
plt.xscale('log')
plt.ylabel('Feature weight in logistic regression')
plt.xlabel('L1 regularization factor (stronger to weaker)')

plt.savefig("figures/L1RegularizationPath.pdf")