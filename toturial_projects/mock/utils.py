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