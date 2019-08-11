import numpy as np

a = np.zeros((6,2))
b = np.ones((len(a),1))
c = np.hstack((a,b))

# print(c)
# print(np.hsplit(c, np.array([2, 3])))
# class_label = c[:, -1]
# print(class_label.reshape(-1,1))
# class_label = c[:, :-1]
# print(class_label)

# class_label = dataset[:, -1] # for last column
# dataset = dataset[:, :-1] # for all but last column

# m = np.array([[1, 2, 3],
#               [5, 6, 7]])
#
#
# v = map(lambda x_y: np.append(x_y, 1), m)
# print(list(v))

# A = np.array( [10e121,2,3] )
# print(A[-1])
# print(A[:-1])

d = 1e-5
print(d)