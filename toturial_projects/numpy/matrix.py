import numpy as np
A = np.random.uniform(-0.1, 0.1, (3, 4))
B = np.array([[5,7], [6,8]])

x = np.array([5, 6,7,8])
x1 = np.ones(3)

# 矩阵乘以向量
# print(np.dot(x, A))
print("----")
print(np.dot(A, x))
print(np.dot(A, x)+x1)
# 矩阵乘以矩阵
# print(np.dot(A, B))
