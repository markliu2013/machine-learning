'''
全连接神经网络
'''
import numpy as np
from utils import sigmoid

class LayerConnect(object):

    def __init__(self, input_size, output_size):
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # self.W = np.ones((output_size, input_size))
        self.b = np.ones(output_size)

    def get_output(self, input):
        return sigmoid(np.dot(self.W, input) + self.b)

    def cal_output(self, input):
        self.input = input
        self.output = self.get_output(input)
        return self.output

    def cal_grad(self, delta_array):
        self.delta = self.input * (1 - self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array.reshape(-1, 1), self.input.reshape(1, -1))
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

class FNN(object):

    def __init__(self, learning_rate=0.1, layers=[]):
        self.learning_rate = learning_rate
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(LayerConnect(layers[i-1], layers[i]))

    def fit(self, X, y, epoch=10):
        for i in range(epoch):
            for j in range(len(X)):
                for k in range(len(self.layers)):
                    self._train_one_sample(X[j], y[j])
            print('epoch %d 结束' % i)

    def _forword(self, x):
        input = x
        for i in range(len(self.layers)):
            input = self.layers[i].cal_output(input)

    def _train_one_sample(self, x, y):
        self._forword(x)
        self.calc_gradient(y)
        self.update_weight()

    def calc_gradient(self, y):
        # 输出层delta
        delta = self.layers[-1].output * (1-self.layers[-1].output) * (y - self.layers[-1].output)
        # ::-1 倒序
        for layer in self.layers[::-1]:
            layer.cal_grad(delta)
            delta = layer.delta
        return delta

    def update_weight(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def predict(self, X):
        return np.array(list(map(self._predict_x, X)))

    def _predict_x(self, x):
        pre_output = x
        for i in range(len(self.layers)):
            pre_output = self.layers[i].get_output(pre_output)
        return pre_output