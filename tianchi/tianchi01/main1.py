import pandas as pd
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv("data/t1.csv")
df_test = pd.read_csv("data/pre.csv")


X = df_train.drop(['USER_ID', 'ITEM_ID', 'CHECK_DATE', 'IS_BUY'], axis=1)
y = df_train['IS_BUY']
clf = LogisticRegression(random_state=1).fit(X, y)

X_Test = df_test.drop(['USER_ID', 'ITEM_ID', 'CHECK_DATE'], axis=1)
Y_Pred = clf.predict(X_Test)

PreResult = pd.DataFrame({
    'user_id': df_test["USER_ID"],
    'item_id': df_test["ITEM_ID"],
    'is_buy': Y_Pred.astype(int)
})

PreResult.to_csv('tianchi_mobile_recommendation_predict.csv', index=False)