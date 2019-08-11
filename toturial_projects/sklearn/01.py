# 预测的线性模型为： y=2x 这是一个单变量单参数模型

import numpy as np
from sklearn.linear_model import LinearRegression

# 注意X是二维的，第一维表示训练集，长度是训练集合的长度。第二维表示训练集中的单条数据，长度是feature的个数。
X = np.array([[1], [2], [3], [4], [5]])
# y是训练集对应的lable，一维数组，长度是训练集合的长度。
y = np.array([2, 4, 6, 8, 10])

reg = LinearRegression().fit(X, y)
print(reg.predict(np.array([[9], [10]])))
