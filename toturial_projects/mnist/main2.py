import numpy as np
import matplotlib.pyplot as plt

with open('data/train-images-idx3-ubyte', 'rb') as imgpath:
    imgpath.read(16)
    X_train = np.fromfile(imgpath, dtype=np.uint8).reshape(60000, 784)

with open('data/train-labels-idx1-ubyte', 'rb') as lbpath:
    lbpath.read(8)
    y_train = np.fromfile(lbpath, dtype=np.uint8)


fig, ax = plt.subplots(
    nrows=5,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 2][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

