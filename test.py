from email import header
import numpy as np
from linear_regression import LinearRegression
import pandas as pd
import os

data_path = "./prog4/code_linear_regression/houseprice"

X_train = pd.read_csv(os.path.join(data_path, "x_train.csv"))
X_test = pd.read_csv(os.path.join(data_path, "x_test.csv"))
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))

lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

y_hat = lr.predict(X=X_test)
mse = lr.error(X=X_test, y=y_test)
print(mse)
