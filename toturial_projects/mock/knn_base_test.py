from knn_base import KNN
import numpy as np
import pandas as pd

knn = KNN()

def test2():
data_path = '../mnist/data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:].to_numpy()
y = df_train.iloc[:, 0].to_numpy()

knn.fit(X, y)

test = pd.read_csv(data_path + "test.csv").to_numpy()
predictions = knn.predict(test)
submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": predictions
})
submission.to_csv("mnist-submission.csv", index=False)

test2()

