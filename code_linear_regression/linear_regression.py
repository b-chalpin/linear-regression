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
        self._init_w_vector(d)

        self.w = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def _fit_gd(self, X, y, lam=0, eta=0.01, epochs=1000):
        """ Compute the weight vector using the gradient desecent based method.
            Save the result in self.w
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample.
        """
        X_bias = self._add_bias_column(X)
        n, d = X_bias.shape
        self._init_w_vector(d)

        # we can calculate these outside of our training loop since they are independent from self.w
        w_coefficient = np.eye(d) - (2 * eta / n) * (X_bias.T @ X_bias)
        w_intercept = (2 * eta / n) * (X_bias.T @ y)

        prev_epoch_error = math.inf

        while epochs >= 0:
            self.w = w_coefficient @ self.w + w_intercept

            next_epoch_error = self._error_z(X_bias, y)
            
            prev_epoch_error = next_epoch_error
            epochs -= 1
            
    def fit_metrics(self, X, y, X_test, y_test, lam=0, eta=0.01, epochs=1000, degree=1):
        """ A method used for model performance analysis purposes. Internal use only.
            parameters:
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                X_test: n x d matrix of validation samples
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
            returns:
                train_mse: epochs x 1 array of training MSE values
                test_mse: epochs x 1 array of validation MSE values
        """
        self.degree = degree
        X = MyUtils.z_transform(X, degree=self.degree)
        
        train_mse = []
        test_mse = []
        
        X_bias = self._add_bias_column(X)
        n, d = X_bias.shape
        self._init_w_vector(d)

        # we can calculate these outside of our training loop since they are independent from self.w
        w_coefficient = np.eye(d) - (2 * eta / n) * (X_bias.T @ X_bias)
        w_intercept = (2 * eta / n) * (X_bias.T @ y)

        prev_epoch_error = math.inf

        while epochs >= 0:
            self.w = w_coefficient @ self.w + w_intercept

            next_epoch_error = self._error_z(X_bias, y)
            
            train_mse.append(next_epoch_error)
            test_mse.append(self.error(X_test, y_test))

            prev_epoch_error = next_epoch_error
            epochs -= 1
            
        return (train_mse, test_mse)

    def predict(self, X):
        """ parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        """
        Z = MyUtils.z_transform(X, degree=self.degree)  # Z-transform to match self.w dimension
        Z_bias = self._add_bias_column(Z)

        return Z_bias @ self.w  # it is assumed that self.w is already trained

    def error(self, X, y):
        """ parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return:
                the MSE for this test set (X,y) using the trained model
        """
        y_hat = self.predict(X)

        return self._calculate_mse(y_hat, y)

    def _error_z(self, Z, y):
        """ An internal helper method to calculate MSE for already
            Z-tranformed samples
            parameters:
                Z: n x d matrix of Z-transformed samples (INCLUDING BIAS)
                y: n x 1 matrix of labels
            return:
                the MSE for this test set (Z,y) using the trained model
        """
        y_hat = Z @ self.w

        return self._calculate_mse(y_hat, y)

    def _init_w_vector(self, d):
        """ If self.w does not exist, it is initialized
            parameters:
                d: scalar, representing number of features in our Z-tranformed samples
        """
        self.w = np.zeros((d, 1))

    def _add_bias_column(self, X):
        """ parameters:
                X: n x d matrix of future samples
            return:
                X: n x (d+1) matrix, with added bias column
        """
        return np.insert(X, 0, 1, axis=1)

    def _calculate_mse(self, y_hat, y):
        """ parameters:
                y_hat: n x 1 vector, predicted labels for samples
                y: n x 1 vector, true labels for samples
            return:
                MSE between the predicted labels and true labels
        """
        return np.square(np.subtract(y_hat, y)).mean()
