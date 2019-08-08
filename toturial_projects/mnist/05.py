#
import pandas as pd
import tensorflow as tf

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:, 0]


# 训练集占位符：28*28=784
x = tf.placeholder(tf.float32, [None, 784])
# 初始化参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 输出结果
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# 计算交叉熵
crossEntropy = -tf.reduce_sum(y_ * tf.log(y))
# 训练策略
trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
# 初始化参数值
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


sess.run(trainStep, feed_dict={x: X_train, y_: y_train})
