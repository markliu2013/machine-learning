# 线性回归算法

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

data_train["Age"] = data_train['Age'].fillna(data_train['Age'].median())
data_train.loc[data_train["Sex"] == "male", "Sex"] = 0
data_train.loc[data_train["Sex"] == "female", "Sex"] = 1

# 选取简单的可用输入特征
predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]

alg = LinearRegression()

kf = KFold(n_splits=3, shuffle=False, random_state=1)

predictions = []
# 样本平均分成3份，3折交叉验证
for train, test in kf.split(data_train):
    # The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
    train_predictors = (data_train[predictors].iloc[train, :])
    # The target we're using to train the algorithm.
    train_target = data_train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(data_train[predictors].iloc[test, :])
    predictions.append(test_predictions)

# The predictions are in three aeparate numpy arrays.	Concatenate them into one.
# We concatenate them on axis 0,as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes(only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions == data_train["Survived"]) / len(predictions)
print("准确率为: ", accuracy)

# 测试test.csv中的数据
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())

data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].max())
data_test.loc[data_test["Sex"] == "male", "Sex"] = 0
data_test.loc[data_test["Sex"] == "female", "Sex"] = 1

test1_predictions = alg.predict(data_test[predictors])
data_test.loc[test1_predictions > .5, 'Survived'] = '1'
data_test.loc[test1_predictions <= .5, 'Survived'] = '0'
submission = pd.DataFrame({
    "PassengerId": data_test["PassengerId"],
    "Survived": data_test["Survived"]
})
submission.to_csv("titanic-submission.csv", index=False)
