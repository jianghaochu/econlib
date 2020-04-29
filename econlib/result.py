from prettytable import PrettyTable, NONE
from scipy.stats import t, f
import numpy as np
import math


class Result:
    """Store result of linear regression."""

    def __init__(self):
        #: :obj:`numpy.array`: Endogenous variable
        self.y = None
        #: :obj:`numpy.array`: Exogenous variable
        self.x = None
        #: :obj:`numpy.array`: Estimated coefficient
        self.coefficient = None
        #: :obj:`numpy.array`: Regression residual
        self.residual = None
        #: :obj:`str`: Name of endogenous variable
        self.y_name = None
        #: :obj:`list` of :obj:`str`: Name of exogenous variable
        self.x_name = None

    @property
    def r_squared(self):
        """:obj:`float`: R squared"""
        return 1 - sum(self.residual ** 2) / sum((self.y - self.y.mean()) ** 2)

    @property
    def adjusted_r_squared(self):
        """:obj:`float`: Adjusted R squared"""
        return 1 - (sum(self.residual ** 2) / (self.x.shape[0] - self.x.shape[1])) / (sum((self.y - self.y.mean()) ** 2) / (self.x.shape[0] - 1))

    @property
    def sigma_squared(self):
        """:obj:`float` : Variance of error term under homoskedasticity"""
        return sum(self.residual ** 2) / (self.x.shape[0] - self.x.shape[1])

    @property
    def covariance(self):
        """:obj:`numpy.array`: Covariance under homoskedasticity"""
        return self.sigma_squared * np.linalg.inv(np.matmul(np.array(self.x).transpose(), np.array(self.x)))

    @property
    def robust_covariance(self):
        """:obj:`numpy.array`: Robust estimate of covariance matrix"""
        txx_inv = np.linalg.inv(np.matmul(self.x.transpose(), self.x))
        return np.matmul(np.matmul(txx_inv, np.matmul(np.matmul(self.x.transpose(), np.diag(self.residual ** 2)), self.x)), txx_inv)

    @property
    def robust_standard_error(self):
        """:obj:`numpy.array`: robust standard error of estimator"""
        return np.sqrt(self.robust_covariance.diagonal())

    @property
    def standard_error(self):
        """:obj:`numpy.array`: Standard error of estimator"""
        return np.sqrt(self.covariance.diagonal())

    @property
    def f_statistic(self):
        """:obj:`numpy.float`: F-statistic of the regression"""
        return (1 / (1 - self.r_squared) - 1) * (self.x.shape[0] - self.x.shape[1]) / (self.x.shape[1] - 1)

    @property
    def loglikelihood(self):
        """:obj:`float`: log likelihood of the linear regression under normality"""
        return -(self.x.shape[0] / 2) * math.log(2 * math.pi) - (self.x.shape[0] / 2) * math.log(sum(self.residual ** 2) / self.x.shape[0]) - (self.x.shape[0] / 2)

    @property
    def aic(self):
        """:obj:`float`: aic of the model on the data"""
        return 2 * self.x.shape[1] - 2 * self.loglikelihood

    @property
    def bic(self):
        """:obj:`float`: bic of the model on the data"""
        return self.x.shape[1] * math.log(self.x.shape[0]) - 2 * self.loglikelihood

    def summary(self, robust=False, tol=4):
        """Print a summary of the result.

        Parameters
        ----------
        robust : boolean, optional
            True if to use robust standard error for testing coefficient. Defaults to False.
        tol : int, optional
            Degree of precision for showing float number. Defaults to 4.
        """
        if robust:
            standard_error = self.robust_standard_error
        else:
            standard_error = self.standard_error
        f_stat = (1 / (1 - self.r_squared) - 1) * (self.x.shape[0] - self.x.shape[1]) / (self.x.shape[1] - 1)
        table = PrettyTable()
        table.title = 'OLS Regression Result'
        table.header = False
        table.vrules = NONE
        table.add_row(['Dependent Variable:', self.y_name, 'R-squared', self.r_squared])
        table.add_row(['No. Observations:', self.x.shape[0], 'Adjusted R-squared', self.adjusted_r_squared])
        table.add_row(['Degrees of Freedom:', self.x.shape[0] - self.x.shape[1], 'F-Statistic', f_stat])
        table.add_row(['Covariance Type:', 'Robust' if robust else 'Non-Robust' , 'Pr(>F-Statistic)', 1-f.cdf(f_stat, self.x.shape[1]-1, self.x.shape[0]-self.x.shape[1])])
        table.add_row(['Log-Likelihood:', self.loglikelihood, 'AIC', self.aic])
        table.add_row(['', '', 'BIC', self.bic])
        table.align['Field 1'] = 'l'
        table.align['Field 2'] = 'r'
        table.align['Field 3'] = 'l'
        table.align['Field 4'] = 'r'
        table.float_format = "8."+str(tol)
        table1 = table.get_string()
        table = PrettyTable()
        table.add_column('Variable', self.x_name)
        table.add_column('Coefficient', self.coefficient)
        table.add_column('Standard Error', standard_error)
        table.add_column('t', self.coefficient / standard_error)
        fun = lambda x: t.cdf(x, df=self.x.shape[0] - self.x.shape[1])
        table.add_column('Pr(>|t|)', 1-2*np.abs(0.5 - np.vectorize(fun)(self.coefficient / standard_error)))
        table.align['Variable'] = 'l'
        table.align['Coefficient'] = 'r'
        table.align['Standard Error'] = 'r'
        table.align['t'] = 'r'
        table.float_format = "8."+str(tol)
        table.vrules = NONE
        table2 = table.get_string()
        return table1 + '\n' + table2


if __name__ == '__main__':
    result = Result()
    result.y = np.array([1, 2, 3, 4, 5])
    result.x = np.array([[1, 2], [1, 4], [1, 1], [1, 9], [1, 3]])
    y = result.y
    x = result.x
    result.coefficient = np.array([1, 1])
    result.residual = None
    result.y_name = 'y'
    result.x_name = ['x1', 'x2']
    beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.array(x).transpose(), np.array(x))), x.transpose()), y)
    result.coefficient = beta_hat
    result.residual = y - np.matmul(x, beta_hat)
    print(result.summary(robust=True))
    print(result.summary(robust=False))
