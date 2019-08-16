# matplotlib 显示数字图片
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

# plt.axis('off')
# img = X.iloc[2].values.reshape(28,-1)
# plt.imshow(img, cmap=plt.cm.binary)
# plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X.iloc[i].values.reshape(28,-1), cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()

