########## Blake Chalpin 00864973


# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 


import numpy as np
import math
import sys

sys.path.append("..")

from misc.utils import MyUtils


class LinearRegression:
    def __init__(self):
        self.w = None  # The (d+1) x 1 numpy array weight matrix
        self.degree = 1

    def fit(self, X, y, CF=True, lam=0, eta=0.01, epochs=1000, degree=1):
        """ Find the fitting weight vector and save it in self.w.

            parameters:
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        """
        self.degree = degree
        X = MyUtils.z_transform(X, degree=self.degree)

        if CF:
            self._fit_cf(X, y, lam)
        else:
            self._fit_gd(X, y, lam, eta, epochs)

    def _fit_cf(self, X, y, lam=0):
        """ Compute the weight vector using the clsoed-form method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample.
        """
        X_bias = self._add_bias_column(X)

        _, d = X_bias.shape

        self._init_w_vector_if_no_exist(d)

        self.w = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def _fit_gd(self, X, y, lam=0, eta=0.01, epochs=1000):
        """ Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample.
        """
        X_bias = self._add_bias_column(X)

        n, d = X_bias.shape

        self._init_w_vector_if_no_exist(d)

        # we can calculate these outside of our training loop since they are independent from self.w
        w_coefficient = np.eye(d) - (2 * eta / n) * X_bias.T @ X_bias
        w_intercept = (2 * eta / n) * X_bias.T @ y

        prev_epoch_error = math.inf

        while epochs >= 0:
            self.w = w_coefficient @ self.w + w_intercept

            next_epoch_error = self.error(X_bias[:, 1:], y)  # remove bias vector from X_bias

            # Repeat the following until the decrease of the value of E(w) at a round
            # becomes too small, or we have finished a certain number of epochs, or
            # ∇E(w) becomes nearly 0.
            if next_epoch_error == 0.0 or next_epoch_error >= prev_epoch_error:  # TODO: revise this condition
                break

            prev_epoch_error = next_epoch_error

            epochs -= 1

        print(f"Only required {epochs} epochs.")

    def predict(self, X):
        """ parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        """
        X_bias = self._add_bias_column(X)

        _, d = X_bias.shape

        self._init_w_vector_if_no_exist(d)

        return X_bias @ self.w

    def error(self, X, y):
        """ parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return:
                the MSE for this test set (X,y) using the trained model
        """
        y_hat = self.predict(X)

        return np.square(np.subtract(y_hat, y)).mean()

    def _init_w_vector_if_no_exist(self, d):
        """ parameters:
                d: scalar, representing number of features in our samples

                if self.w does not exist, it is initialized
        """
        if self.w is None:
            self.w = np.zeros((d, 1))

    def _add_bias_column(self, X):
        """ parameters:
                X: n x d matrix of future samples

            return:
                X: n x (d+1) matrix, with added bias column
        """
        return np.insert(X, 0, 1, axis=1)
