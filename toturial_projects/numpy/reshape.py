import numpy as np

m = np.zeros(27)

print(m.reshape(3,3,3))
print("-----")
print(m.reshape(-1,3,3,3))

# print(z.reshape(4, -1))
# print(z.reshape(4, 3))
# print(z.reshape(-1))
# print(z.reshape(-1, 1))

