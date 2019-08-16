# keras 全连接网络
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "test.csv")

train_images = df_train.iloc[:, 1:]
train_labels = df_train.iloc[:, 0]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

predictions = model.predict(df_test)

submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": list(map(np.argmax, predictions))
})
submission.to_csv("mnist-submission.csv", index=False)
