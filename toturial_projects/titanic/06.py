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
        print(self.weights)
        # expit=sigmoid, sigmod之后四舍五入
        return np.fromiter(map(lambda x: round(sigmoid(np.dot(x, self.weights))), X_bais), dtype=np.int)

    def _bgd(self, X, y, epoch=300):
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


# 逻辑回归算法

import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 加载训练集和测试集数据
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_combined_train_test = df_train.append(df_test)
train_size = df_train.shape[0]

predictors = ["Pclass", "Age", "Embarked", "Sex"]
df_combined_train_test = df_combined_train_test[predictors]

df_combined_train_test["Age"] = df_combined_train_test['Age'].fillna(df_combined_train_test['Age'].median())

dummies_df_Pclass = pd.get_dummies(df_combined_train_test['Pclass'], prefix=df_combined_train_test[['Pclass']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Pclass], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Pclass'],axis=1)

# 使用众数填充Embarked
df_combined_train_test['Embarked'].fillna(df_combined_train_test['Embarked'].mode().iloc[0],inplace=True)
# 为了后面的特征分析，这里我们将Embarked特征进行factorizing
df_combined_train_test['Embarked'] = pd.factorize(df_combined_train_test['Embarked'])[0]
# 使用pd.get_dummies获取one-hot编码
dummies_df_Embarked = pd.get_dummies(df_combined_train_test['Embarked'], prefix=df_combined_train_test[['Embarked']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Embarked], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Embarked'],axis=1)

df_combined_train_test['Sex'] = pd.factorize(df_combined_train_test['Sex'])[0]
dummies_df_Sex = pd.get_dummies(df_combined_train_test['Sex'], prefix=df_combined_train_test[['Sex']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Sex], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Sex'],axis=1)

X = df_combined_train_test[:train_size].to_numpy()
y = df_train.Survived.to_numpy()
clf = LogisticRegression().fit(X, y)

X_Test = df_combined_train_test[train_size:].to_numpy()
Y_Pred = clf.predict(X_Test)

FinalResult = pd.DataFrame({'PassengerId':df_test["PassengerId"], 'Survived':Y_Pred.astype(int)})
FinalResult.to_csv('titanic-submission2.csv', index=False)
