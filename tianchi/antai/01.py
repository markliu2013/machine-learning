import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import gc
# 禁用科学计数法
sns.set()
pd.set_option('display.float_format',lambda x : '%.2f' % x)
plt.style.use('seaborn-dark') 
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['figure.figsize'] = (10.0, 5.0)
plt.rcParams['font.sans-serif'] = ['SimHei']

path = 'data/'
item = pd.read_csv(path+'Antai_AE_round1_item_attr_20190626.csv')
train = pd.read_csv(path+'Antai_AE_round1_train_20190626.csv')
test = pd.read_csv(path+'Antai_AE_round1_test_20190626.csv')

