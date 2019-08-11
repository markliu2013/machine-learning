# 预测的线性模型为： y=3x+1 这是一个单变量多参数模型

import numpy as np
from sklearn.linear_model import LinearRegression

X1 = np.array([[1], [2], [3], [4], [5]])
y1 = np.array([4, 7, 10, 13, 16])

X2 = np.array([[1], [2], [3], [4], [5]])
y2 = np.array([2, 4, 6, 8, 10])

reg = LinearRegression().fit(X1, y1)
print(reg.predict(np.array([[9], [10]])))

reg = reg.fit(X2, y2)
print(reg.predict(np.array([[9], [10]])))
