from perceptron_raw import PerceptronRaw
import numpy as np
import pandas as pd

def test1():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    p = PerceptronRaw()
    p.fit(X, y)
    print(p.predict(np.array([[1, 2]])))

def test2():
    data_path = '../mnist/data/'
    df_train = pd.read_csv(data_path + "train.csv")
    X = df_train.iloc[:, 1:].to_numpy()
    y = df_train.iloc[:, 0].to_numpy()

test2()