import sys
from functools import reduce
import numpy as np

class LinearRegression:

    # 初始化一个空的weights，在fit中定义长度，alpha学习速率
    def __init__(self, alpha=0.01):
        self.weights = []
        self.alpha = alpha

    def fit(self, X, y):
        self.weights = [1.0] * (len(X[0]) + 1)  # bias
        self._bgd(X, y)
        return self

    def _sgd(self, X, y):
        loss = sys.float_info.max
        while loss != 0:
            pre_loss = loss
            self._epoch_sgd(X, y)
            loss = self._get_loss(X, y)
            if abs(loss - pre_loss) < 1e-20:
                break

    def _epoch_sgd(self, X, y):
        X_y = np.hstack((X, y.reshape(-1, 1)))
        for x_y in X_y:
            self._batch_sgd(x_y)

    def _batch_sgd(self, x_y):
        gradient_loss = self._get_loss_gradient(x_y)
        self.weights = np.subtract(self.weights, gradient_loss[0] * self.alpha)

    def _bgd(self, X, y):
        loss = sys.float_info.max
        # loss为0或者优化不到
        while loss != 0:
            pre_loss = loss
            loss = self._one_bgd(X, y)
            if abs(loss - pre_loss) < 1e-15:
                break

    def _one_bgd(self, X, y):
        # 将x和y拼在一起
        X_y = np.hstack((X, y.reshape(-1, 1)))
        gradient_loss = map(self._get_loss_gradient, X_y)
        reduce_gradient_loss = reduce(self._reduce_lost_function, gradient_loss)
        gradient = np.divide(reduce_gradient_loss[0], len(X))
        self.weights = np.subtract(self.weights, gradient * self.alpha)
        return reduce_gradient_loss[1]

    # 参考损失函数的导函数
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

    def _get_loss(self, X, y):
        loss = map(lambda x, y: (x - y) * (x - y), self.predict(X), y)
        loss = reduce(lambda x, y: x +y, loss)
        loss = loss / (2*len(X))
        return loss;

    def predict(self, X):
        X_bais = np.hstack((X, np.ones((len(X), 1))))
        # np.fromiter将map后的结果转为1-dimensional array
        return np.fromiter(map(lambda x: np.dot(x, self.weights), X_bais), dtype=np.double)


X1 = np.array([[1], [2], [3], [4], [5]])
y1 = np.array([4, 7, 10, 13, 16])

X2 = np.array([[1], [2], [3], [4], [5]])
y2 = np.array([2, 4, 6, 8, 10])

X3 = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y3 = np.dot(X3, np.array([1, 2])) + 3

reg = LinearRegression().fit(X1, y1)
print(reg.predict(np.array([[9], [10]])))

reg = reg.fit(X2, y2)
print(reg.predict(np.array([[9], [10]])))

reg = reg.fit(X3, y3)
print(reg.predict(np.array([[3, 5]])))
