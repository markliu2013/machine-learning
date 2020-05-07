from perceptron_raw import PerceptronRaw
import numpy as np
import pandas as pd

p = PerceptronRaw()

def test1():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    p.fit(X, y)
    print(p.predict(np.array([[1, 2]])))

def test2():
    data_path = '../mnist/data/'
    df_train = pd.read_csv(data_path + "train.csv")
    X = df_train.iloc[:, 1:].to_numpy()
    y = df_train.iloc[:, 0].to_numpy()
    y = np.where(y == 0, 1, -1)

    p.fit(X, y)

    test = pd.read_csv(data_path + "test.csv").to_numpy()
    predictions = p.predict(test)
    submission = pd.DataFrame({
        "ImageId": range(1, 1 + len(predictions)),
        "Label": predictions
    })
    submission.to_csv("mnist-submission.csv", index=False)



# 线性不可分
def test3():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, -1, -1, 1])
    p.fit(X, y)
    print(p.predict(np.array([[1, 1]])))

test1()