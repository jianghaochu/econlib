"""
linearmodel.py
===============================================
This module defines the class of linear models.
"""

import pandas as pd
import numpy as np

from econlib.result import Result


class LinearModel:
    """The class stores the basic information of a linear model.

    :param y: an array of the exogenous variable
    :type y: pandas.Series or numpy.array
    :param x: an array of the endogenous variables
    :type x: pandas.DataFrame or numpy.array
    """

    def __init__(self, y=None, x=None):
        """The constructor function"""
        self.y = y
        self.x = x

    def ols(self, y=None, x=None):
        """Ordinary least squares regression of a linear model.

        :param y: an array of the exogenous variable
        :type y: pandas.Series or numpy.array
        :param x: an array of the endogenous variables
        :type x: pandas.DataFrame or numpy.array
        """
        if y is None:
            y = self.y.copy()
        if x is None:
            x = self.x.copy()
        result = Result()  # container to store regression result
        # convert x and y to np.array if needed
        if type(y) is pd.DataFrame:
            result.y_name = list(y.columns)
            y = np.array(y)
        else:
            result.y_name = 'y'
        if type(x) is pd.DataFrame:
            result.x_name = list(x.columns)
            x = np.array(x)
        else:
            result.x_name = ['x'+str(i) for i in range(x.shape[1])]
        # linear algebra for OLS
        beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.array(x).transpose(), np.array(x))), x.transpose()), y)
        # store result
        result.coefficient = beta_hat
        result.y = y
        result.x = x
        result.residual = y - np.matmul(x, beta_hat)
        result.r_squared = 1 - sum(result.residual ** 2) / sum((y - y.mean()) ** 2)
        result.adjusted_r_squared = 1 - (sum(result.residual ** 2) / (x.shape[0] - x.shape[1])) / (sum((y - y.mean()) ** 2) / (x.shape[0] - 1))
        result.sigma_squared = sum(result.residual ** 2) / (x.shape[0] - x.shape[1])  # error variance
        result.covariance = result.sigma_squared * np.linalg.inv(np.matmul(np.array(x).transpose(), np.array(x)))
        result.standard_error = np.sqrt(result.covariance.diagonal())
        return result


# unit testing
if __name__ == '__main__':
    n, k = 100, 2
    beta = np.array([1, 1, 10])
    xx = np.concatenate([np.ones((n, 1)), np.random.randn(n, k)], axis=1)
    yy = np.matmul(xx, beta) + np.random.randn(n)
    linear_model = LinearModel(y=yy, x=xx)
    ols_result = linear_model.ols()
    ols_result.summary()
