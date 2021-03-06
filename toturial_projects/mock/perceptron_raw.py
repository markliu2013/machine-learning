import numpy as np
from utils import sign

'''
实现 Perceptron 原始形式的算法
'''
class PerceptronRaw(object):

    def __init__(self, learning_rate=1, max_epoch=10000):
        self.learning_rate = learning_rate
        # 最大迭代多少次数
        self.max_epoch = max_epoch

    '''
    SGD算法，参考算法2.1，公式2.6，2.7
    '''
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        epoch = 1
        m = len(X) # 样本集个数
        print('参数初始化为w=%s, b=%.2f，开始训练。' % (self.w, self.b))
        while epoch <= self.max_epoch:
            i = 0
            while i < m:
                xi = X[i]
                yi = y[i]
                # 判断是否为误分类
                if yi * self._get_linear_x(xi) <= 0:
                    self.w = self.w + self.learning_rate * yi * xi
                    self.b = self.b + self.learning_rate * yi
                    print('误分类点x%d=%s, 更新后的参数为：w=%s, b=%.2f' % (i + 1, xi, self.w, self.b))
                    # 更新参数后，回到第一个样本，继续
                    i = 0
                    epoch = epoch + 1 # 这个算一次epoch
                    if epoch > self.max_epoch:
                        break
                    else:
                        continue
                else:
                    # 当前的样本不是误差类，则继续下一个
                    i = i+1
                    continue
            # 尝试max_epoch次之后
            if epoch > self.max_epoch: # 线性不可分
                print('线性不可分')
                break
            else:
                print('训练完成, epock=%d' % (epoch))
                break
        return self

    def predict(self, X):
        return np.fromiter(map(self._predict_x, X), dtype=np.int)

    def _predict_x(self, x):
        return sign(self._get_linear_x(x))

    def _get_linear_x(self, x):
        return np.dot(x, self.w) + self.b
