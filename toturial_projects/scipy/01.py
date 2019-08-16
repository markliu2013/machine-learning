#coding:utf-8
from scipy.stats import binom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

fig,ax = plt.subplots(1,1)
n = 100
p = 0.5
#平均值, 方差, 偏度, 峰度
mean,var,skew,kurt = binom.stats(n,p,moments='mvsk')
print(mean,var,skew,kurt)
#ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.arange(binom.ppf(0.01, n, p),binom.ppf(0.99, n, p))
ax.plot(x, binom.pmf(x, n, p),'o')
plt.title(u'二项分布概率质量函数')
plt.show()