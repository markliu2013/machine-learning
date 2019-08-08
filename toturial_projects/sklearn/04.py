# 预测的线性模型为： y = x的平方 这是一个多变量多参数模型

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

# X新增一个特征，x的平方
X = np.concatenate((X, np.square(X)), axis=1)

reg = LinearRegression().fit(X, y)

# 预测的也必须是二维
print(reg.predict(np.array([[9, 81]])))
