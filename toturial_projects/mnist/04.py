import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

plt.axis('off')
img = X.iloc[2].values.reshape(28,-1)
plt.imshow(img, cmap=cm.binary)
plt.show()


