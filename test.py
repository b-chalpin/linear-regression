from email import header
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from misc
import pandas as pd
import os

data_path = "./prog4/code_linear_regression/houseprice"

X_train = pd.read_csv(os.path.join(data_path, "x_train.csv")).to_numpy()
X_test = pd.read_csv(os.path.join(data_path, "x_test.csv")).to_numpy()
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).to_numpy()
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).to_numpy()

lr = LinearRegression()

# configure hyperparameters
max_degree = 4
gd = True
# lam = 1
epochs = 1000
eta = 0.01

degree_x = [] # store each degree
mse_val_y = [] # store the validation error for each degree
mse_train_y = [] # store the training error for each degree
prediction_y = [] # store the predictions for each degree

for r in range(1, max_degree + 1):  # 1-based indexing
    if gd: 
        lr.fit(X=X_train, y=y_train, CF=False, epochs=epochs, eta=eta, degree=r)
    else:
        lr.fit(X=X_train, y=y_train, CF=True, degree=r)

    validaiton_mse = lr.error(X=X_test, y=y_test)
    training_mse = lr.error(X=X_train, y=y_train)
    
    print(f"MSE for degree {r}:\nTest: {validaiton_mse}\nTrain: {training_mse}\n")

    y_hat = lr.predict(X=X_test)

    degree_x.append(r)
    mse_val_y.append(validaiton_mse)
    mse_train_y.append(training_mse)
    prediction_y.append(y_hat) 