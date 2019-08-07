# 逻辑回归算法

import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# 加载训练集和测试集数据
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
df_combined_train_test = df_train.append(df_test)
train_size = df_train.shape[0]

predictors = ["Pclass", "Age", "Embarked", "Sex"]
df_combined_train_test = df_combined_train_test[predictors]

df_combined_train_test["Age"] = df_combined_train_test['Age'].fillna(df_combined_train_test['Age'].median())

dummies_df_Pclass = pd.get_dummies(df_combined_train_test['Pclass'], prefix=df_combined_train_test[['Pclass']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Pclass], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Pclass'],axis=1)

# 使用众数填充Embarked
df_combined_train_test['Embarked'].fillna(df_combined_train_test['Embarked'].mode().iloc[0],inplace=True)
# 为了后面的特征分析，这里我们将Embarked特征进行factorizing
df_combined_train_test['Embarked'] = pd.factorize(df_combined_train_test['Embarked'])[0]
# 使用pd.get_dummies获取one-hot编码
dummies_df_Embarked = pd.get_dummies(df_combined_train_test['Embarked'], prefix=df_combined_train_test[['Embarked']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Embarked], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Embarked'],axis=1)

df_combined_train_test['Sex'] = pd.factorize(df_combined_train_test['Sex'])[0]
dummies_df_Sex = pd.get_dummies(df_combined_train_test['Sex'], prefix=df_combined_train_test[['Sex']].columns[0])
df_combined_train_test = pd.concat([df_combined_train_test, dummies_df_Sex], axis=1)
df_combined_train_test=df_combined_train_test.drop(['Sex'],axis=1)

X = df_combined_train_test[:train_size]
y = df_train.Survived
clf = LogisticRegression(random_state=1).fit(X, y)

X_Test = df_combined_train_test[train_size:]
Y_Pred = clf.predict(X_Test)

FinalResult = pd.DataFrame({'PassengerId':df_test["PassengerId"], 'Survived':Y_Pred.astype(int)})
FinalResult.to_csv('titanic-submission.csv', index=False)
