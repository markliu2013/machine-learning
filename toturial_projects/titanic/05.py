# 逻辑回归算法

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')

data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

data_train["Age"] = data_train['Age'].fillna(data_train['Age'].median())
data_train.loc[data_train["Sex"] == "male", "Sex"] = 0
data_train.loc[data_train["Sex"] == "female", "Sex"] = 1

predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]

# 初始化逻辑回归算法
LogRegAlg = LogisticRegression(random_state=1)
re = LogRegAlg.fit(data_train[predictors], data_train["Survived"])

# 使用sklearn库里面的交叉验证函数获取预测准确率分数
scores = model_selection.cross_val_score(LogRegAlg, data_train[predictors], data_train["Survived"], cv=3)
# 使用交叉验证分数的平均值作为最终的准确率
print("准确率为: ", scores.mean())

# 测试test.csv中的数据
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())

data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].max())
data_test.loc[data_test["Sex"] == "male", "Sex"] = 0
data_test.loc[data_test["Sex"] == "female", "Sex"] = 1

# 构造测试集的Survived列，
data_test["Survived"] = -1
test_predictors = data_test[predictors]
data_test["Survived"] = LogRegAlg.predict(test_predictors)

submission = pd.DataFrame({
    "PassengerId": data_test["PassengerId"],
    "Survived": data_test["Survived"]
})
submission.to_csv("titanic-submission.csv", index=False)
