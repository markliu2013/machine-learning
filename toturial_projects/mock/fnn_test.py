from fnn import FNN
import numpy as np
import pandas as pd
from tensorflow import keras

fnn = FNN(layers=[784, 300, 10])

def test1():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, 0])
    fnn.fit(X, y)
    print(fnn.predict(np.array([[1, 2], [3, 5]])))

def test2():
    data_path = '../mnist/data/'
    df_train = pd.read_csv(data_path + "train.csv")
    X = df_train.iloc[:, 1:].to_numpy()
    y = df_train.iloc[:, 0].to_numpy()
    y = keras.utils.to_categorical(y, num_classes=10)
    fnn.fit(X, y, epoch=20)
    test = pd.read_csv(data_path + "test.csv").to_numpy()
    predictions = fnn.predict(test)
    results = np.argmax(predictions, axis = 1)
    results = pd.Series(results, name='Label', dtype='int32')
    submission = pd.concat([pd.Series(range(1, results.size+1), name='ImageId', dtype='int32'), results], axis=1)
    submission.to_csv("MNIST_Dataset_Submissions.csv", index=False)

test2()

