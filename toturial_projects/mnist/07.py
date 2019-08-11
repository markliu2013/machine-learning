import pandas as pd
# from sklearn.linear_model import LinearRegression
from linear_regression import LinearRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:].to_numpy()
y = df_train.iloc[:, 0].to_numpy()
lg = LinearRegression().fit(X, y)

df_test = pd.read_csv(data_path + "test.csv")
predictions = lg.predict(df_test)
submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": list(map(lambda x: int(round(x)), predictions))
})
submission.to_csv("mnist-submission9.csv", index=False)
