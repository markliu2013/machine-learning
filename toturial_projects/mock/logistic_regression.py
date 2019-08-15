import math
import numpy as np
from functools import reduce

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

class LogisticRegression:

    def __init__(self, alpha=0.01):
        self.weights = []
        self.alpha = alpha

    def fit(self, X, y):
        self.weights = [1.0] * (len(X[0]) + 1)  # bias
        self._bgd(X, y)
        return self

    def predict(self, X):
        X_bais = np.hstack((X, np.ones((len(X), 1))))
        # expit=sigmoid, sigmod之后四舍五入
        return np.fromiter(map(lambda x: round(sigmoid(np.dot(x, self.weights))), X_bais), dtype=np.int)

    def _bgd(self, X, y, epoch=30):
        for e in range(epoch):
            self._one_bgd(X, y)
            print("Epoch: %03d | Loss: %.3f" % (e, self._get_loss(X, y)))
        # loss = self._get_loss(X, y)
        # while loss != 0:
        #     pre_loss = loss
        #     self._one_bgd(X, y)
        #     loss = self._get_loss(X, y)
        #     if abs(loss - pre_loss) < 1e-3:
        #         break

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
        predict_y = sigmoid(np.dot(np.append(x, 1), self.weights))
        if y == 1:
            return -(math.log(predict_y))
        if y == 0:
            return -(math.log(1-predict_y))

    '''
        总样本的损失
    '''
    def _get_loss(self, X, y):
        # return np.sum(list(map(self._loss_function, X, y)))
        return reduce(lambda x, y: x + y, map(self._loss_function, X, y))

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
        # return np.divide(np.sum(list(map(self._gradient_function, X, y))), len(X))
        return np.divide(reduce(lambda x, y: x + y, map(self._gradient_function, X, y)), len(X))


from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
y = np.vectorize(lambda x: 1 if x!=0 else x)(y)
clf = LogisticRegression().fit(X, y)
prediction = clf.predict([[5.3, 3.9, 1.2, 0.1], [1.3, 1.9, 0.2, 0.1], [11.3, 1.9, 0.2, 0.1]])
print(prediction)
