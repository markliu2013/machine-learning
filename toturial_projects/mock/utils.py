import numpy as np

'''
Perceptron的sign函数。参考李航2.2
'''
def sign(x):
    return -1 if x < 0 else 1

'''
一般的sigmoid函数在输入大整数时会返回1，导致log操作时报错。经过我的测试这个方法最好使。
https://stackoverflow.com/questions/52423163/sigmoid-function-returns-1-for-large-positive-inputs
'''
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

'''
https://blog.csdn.net/qq_19707521/article/details/78479532
'''
def distance(x1, x2, l=2):
    # 欧氏距离
    # return np.sqrt(np.sum(np.square(x1 - x2)))
    # 曼哈顿距离
    # np.sum(np.abs(x1 - x2))
    return np.linalg.norm(x1-x2, l)

def relu(x):
    return max(0, x)
