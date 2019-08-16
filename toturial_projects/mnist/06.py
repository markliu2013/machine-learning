# https://www.kaggle.com/raahat98/digit-classification-with-keras-99-accuracy
# keras RNN 神经网络
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

data_path = 'data/'
train = pd.read_csv(data_path + "train.csv")
test = pd.read_csv(data_path + "test.csv")

y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)
del train

X_train = X_train/255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test/255.0
test = test.values.reshape(-1, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes = 10)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])

model.fit(X_train, y_train, epochs=20, batch_size = 128)

results = model.predict(test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)
submission.to_csv("MNIST_Dataset_Submissions.csv", index = False)
