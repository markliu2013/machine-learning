import numpy as np
from utils import sign
from perceptron_raw import PerceptronRaw

'''
实现 Perceptron 对偶形式的算法
'''
class PerceptronDuality(PerceptronRaw):

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.alpha = [0] * n_samples

        gram_matrix = np.dot(X, X.T)

        i = 0
        while i < n_samples:
            inner_product = gram_matrix[i]
            distance = y[i] * (np.sum(self.alpha * y * inner_product) + self.b)
            if distance <= 0:
                self.alpha[i] += self.learning_rate
                self.b += self.learning_rate * y[i]
                i = 0
            else:
                i += 1
        for j in range(n_samples):
            self.w += self.alpha[j] * X[j] * y[j]
        return self


p = PerceptronDuality(max_epoch=5000);
X = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])
p.fit(X, y)
print(p.predict(np.array([[1, 2]])))
