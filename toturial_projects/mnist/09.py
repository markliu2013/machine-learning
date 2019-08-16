# 使用sklearn svm
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn import svm

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')
classifier.fit(X, y)

df_test = pd.read_csv(data_path + "test.csv")
predictions = classifier.predict(df_test)

submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": predictions
})
submission.to_csv("mnist-submission.csv", index=False)