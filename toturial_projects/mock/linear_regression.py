import sys
from functools import reduce
import numpy as np

class LinearRegression:

    def __init__(self, alpha = 0.01):
        self.weights = []
        self.alpha = alpha

    def fit(self, X, y):
        self._bgd(X, y)
        return self

    def _bgd(self, X, y):
        self.weights = [1.0] * (len(X[0]) + 1)  # bias
        loss = sys.float_info.max
        while loss != 0:
            pre_loss = loss
            loss = self._one_bgd(X, y)
            if abs(loss - pre_loss) < 1e-15:
                break


    def _one_bgd(self, X, y):
        X_y = np.hstack((X, y.reshape(-1, 1)))
        gradient_loss = map(self._get_loss_gradient, X_y)
        reduce_gradient_loss = reduce(self._reduce_lost_function, gradient_loss)
        gradient = np.divide(reduce_gradient_loss[0], len(X))
        self.weights = np.subtract(self.weights, gradient * self.alpha)
        return reduce_gradient_loss[1]

    def _get_loss_gradient(self, x_y):
        x = x_y[:-1]
        y = x_y[-1]
        x_bias = np.append(x, 1)
        loss = np.dot(x_bias, self.weights) - y
        gradient = np.multiply(loss, x_bias)
        return (gradient, loss)

    def _reduce_lost_function(self, x, y):
        gradient = x[0] + y[0]
        loss = x[1] + y[1]
        return (gradient, loss)

    def predict(self, X):
        X_bais = np.hstack((X, np.ones((len(X), 1))))
        return np.fromiter(map(lambda x: np.dot(x, self.weights), X_bais), dtype=np.double)


X = np.array([[1], [2], [3], [4], [5]])
# y是训练集对应的lable，一维数组，长度是训练集合的长度。
y = np.array([2, 4, 6, 8, 10])

reg = LinearRegression().fit(X, y)
print(reg.predict(np.array([[9], [10]])))
