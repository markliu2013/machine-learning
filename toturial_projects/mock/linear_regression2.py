from functools import reduce
import numpy as np
import math


class LinearRegression:

    # 初始化一个空的weights，在fit中定义长度，alpha学习速率
    def __init__(self, alpha=0.01):
        self.weights = []
        self.alpha = alpha

    def fit(self, X, y):
        self.weights = [1.0] * (len(X[0]) + 1)  # bias
        self._bgd(X, y)
        return self

    def _bgd(self, X, y, epoch=20):
        # for e in range(epoch):
        #     self._one_bgd(X, y)
        #     #print("Epoch: %03d | Loss: %.3f" % (e, self._get_loss(X, y)))
        loss = self._get_loss(X, y)
        # loss为0或者优化不到
        while loss != 0:
            pre_loss = loss
            self._one_bgd(X, y)
            loss = self._get_loss(X, y)
            if abs(loss - pre_loss) < 1e-25:
                break

    '''
        一次批量梯度下降
    '''
    def _one_bgd(self, X, y):
        gradient = self._get_gradient(X, y)
        self.weights = np.subtract(self.weights, gradient * self.alpha)

    '''
        损失函数，传入单个的样本x，计算该样本的损失
    '''
    def _loss_function(self, x, y):
        predict_y = np.dot(np.append(x, 1), self.weights)
        return math.pow(predict_y-y, 2) / 2

    '''
        总样本的损失
    '''
    def _get_loss(self, X, y):
        return np.sum(list(map(self._loss_function, X, y)))

    '''
        处理单个样本，供整体gradient用
    '''
    def _gradient_function(self, x, y):
        loss = np.dot(np.append(x, 1), self.weights) - y
        return np.multiply(loss, np.append(x, 1))

    '''
        总体梯度
    '''
    def _get_gradient(self, X, y):
        # TODO 为什么这个是错误的
        # return np.divide(np.sum(list(map(self._gradient_function, X, y))), len(X))
        return np.divide(reduce(lambda x, y: x + y, map(self._gradient_function, X, y)), len(X))

    def predict(self, X):
        X_bais = np.hstack((X, np.ones((len(X), 1))))
        # np.fromiter将map后的结果转为1-dimensional array
        return np.fromiter(map(lambda x: np.dot(x, self.weights), X_bais), dtype=np.double)

