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

train_images = train_images
test_images = df_test

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images.iloc[i].values.reshape(28,-1), cmap=plt.cm.binary)
#     plt.xlabel(train_labels[i])
# plt.show()

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

predictions = model.predict(test_images)

submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": list(map(np.argmax, predictions))
})
submission.to_csv("mnist-submission7.csv", index=False)
