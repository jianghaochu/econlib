from prettytable import PrettyTable, NONE
from scipy.stats import t
import numpy as np

class Result:
    """The class is used to store result from linear regressions."""

    def __init__(self):
        self.y = None
        self.x = None
        self.coefficient = None
        self.residual = None
        self.y_name = None
        self.x_name = None
        self.r_squared = None
        self.adjusted_r_squared = None
        self.covariance = None
        self.sigma_squared = None
        self.standard_error = None

    def summary(self, tol=3):
        """
        Function that prints a summary table of the result.

        Args:
            tol (int): tolerance for inaccuracy (or precision of output numbers)
        """
        table = PrettyTable()
        table.title = 'OLS Regression Result'
        table.header = False
        table.vrules = NONE
        table.add_row(['Dependent Variable:', self.y_name, 'R-squared', self.r_squared])
        table.add_row(['No. Observations:', self.x.shape[0], 'Adjusted R-squared', self.adjusted_r_squared])
        table.add_row(['Degrees of Freedom:', self.x.shape[0] - self.x.shape[1], 'F-Statistic', self.adjusted_r_squared])
        table.add_row(['Residual Std. Error:', np.sqrt(self.sigma_squared), 'Pr(>F-Statistic)', self.adjusted_r_squared])
        table.align['Field 1'] = 'l'
        table.align['Field 2'] = 'r'
        table.align['Field 3'] = 'l'
        table.align['Field 4'] = 'r'
        table.float_format = "8.4"
        print(table)
        table = PrettyTable()
        table.add_column('variable', self.x_name)
        table.add_column('coefficient', self.coefficient)
        table.add_column('standard_error', self.standard_error)
        f = lambda x: t.cdf(x, df=self.x.shape[0] - self.x.shape[1])
        table.add_column('t', self.coefficient / self.standard_error)
        table.add_column('Pr(>|t|)', 1-2*np.abs(0.5 - np.vectorize(f)(self.coefficient / self.standard_error)))
        table.align['variable'] = 'l'
        table.align['coefficient'] = 'r'
        table.align['standard_error'] = 'r'
        table.align['t'] = 'r'
        table.float_format = "8.4"
        table.vrules = NONE
        print(table)


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
    result.r_squared = 1 - sum(result.residual ** 2) / sum((y - y.mean()) ** 2)
    result.adjusted_r_squared = 1 - (sum(result.residual ** 2) / (x.shape[0] - x.shape[1])) / (
                sum((y - y.mean()) ** 2) / (x.shape[0] - 1))
    result.sigma_squared = sum(result.residual ** 2) / (x.shape[0] - x.shape[1])
    result.covariance = result.sigma_squared * np.linalg.inv(np.matmul(np.array(x).transpose(), np.array(x)))
    result.standard_error = np.sqrt(result.covariance.diagonal())
    result.summary()