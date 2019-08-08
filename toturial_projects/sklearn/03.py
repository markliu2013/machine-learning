# 预测的线性模型为： y = x0 + 2x1 + 3 这是一个多变量多参数模型

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

reg = LinearRegression().fit(X, y)
print(reg.predict(np.array([[3, 5]])))
