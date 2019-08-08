# 预测的线性模型为： y=3x+1 这是一个单变量多参数模型

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([4, 7, 10, 13, 16])

reg = LinearRegression().fit(X, y)
# 预测两条
print(reg.predict(np.array([[9], [10]])))
