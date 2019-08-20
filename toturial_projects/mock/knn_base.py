import numpy as np
from utils import distance
'''
https://github.com/SmallVagetable/machine_learning_python/blob/master/knn/knn_base.py
https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py
'''
class KNN(object):

    def __init__(self, k=20, p=2):
        self.k = k
        self.p = p


    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.fromiter(map(self._predict_x, X), dtype=np.int)

    def _predict_x(self, x):
        print(x)
        train_length = len(self.X)
        ## i个样本的距离
        distList = [0] * train_length
        # 遍历训练集中所有的样本点，计算与x的距离
        for i in range(train_length):
            distList[i] = distance(self.X[i], x, self.p)
        # 对距离列表进行排序
        topKList = np.argsort(np.array(distList))[:self.k]
        labelList = [0] * len(np.unique(self.y))
        for index in topKList:
            labelList[int(self.y[index])] += 1
        return labelList.index(max(labelList))