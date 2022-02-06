from email import header
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import pandas as pd
import os

data_path = "./prog4/code_linear_regression/houseprice"

X_train = pd.read_csv(os.path.join(data_path, "x_train.csv")).to_numpy()
X_test = pd.read_csv(os.path.join(data_path, "x_test.csv")).to_numpy()
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).to_numpy()
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).to_numpy()

lr = LinearRegression()

# set this parameter
max_degree = 4

degree_x = []
mse_y = []
prediction_y = []

for r in range(1, max_degree + 1):  # 1-based indexing
    lr.fit(X=X_train, y=y_train, CF=True, degree=r)

    mse = lr.error(X=X_test, y=y_test)
    print(f"MSE for degree {r}: {mse}")

    y_hat = lr.predict(X=X_test)

    degree_x.append(r)
    mse_y.append(mse)
    prediction_y.append((r, y_hat))   # store degree and prediction

# plot to examine the true y vs. predicted y for each degree
fig, axes = plt.subplots(len(prediction_y))

# x-vals for fig2 subplots
fig2_x = np.arange(y_test.shape[0])

for i, (degree, prediction) in enumerate(prediction_y):
    ax_pred = axes[i]

    ax_pred.set_title(f"Degree: {degree}")
    ax_pred.set_ylabel("House price (y)")

    # plot truth and prediction
    ax_pred.plot(fig2_x, y_test.flatten())
    ax_pred.plot(fig2_x, prediction.flatten())
    ax_pred.legend(["truth", "prediction"])

plt.show()
