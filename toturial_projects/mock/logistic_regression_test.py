from logistic_regression import LogisticRegression
import numpy as np

def test1():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    y = np.vectorize(lambda x: 1 if x!=0 else x)(y)
    clf = LogisticRegression().fit(X, y)
    prediction = clf.predict([[5.3, 3.9, 1.2, 0.1], [1.3, 1.9, 0.2, 0.1], [11.3, 1.9, 0.2, 0.1]])
    print(prediction)

test1()