from econlib.linearmodel import LinearModel
import pandas as pd

y = pd.DataFrame([1, 2, 3, 4, 5])
x = pd.DataFrame([1, 3, 3, 3, 5])

result = fit(y, x)

col_name = list(X.columns)
X = np.array(X)
y = np.array(y)
beta_hat_ridge = np.matmul(np.matmul(np.linalg.inv(
                 np.matmul(np.array(X).transpose(), np.array(X)) + ridge_para*np.diag(np.ones(X.shape[1]))),
                      X.transpose()), y)
pred_y = np.matmul(X, beta_hat_ridge)
coefficient = pd.DataFrame(beta_hat_ridge, index=col_name)