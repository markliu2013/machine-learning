# sklearn LogisticRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression

data_path = 'data/'
df_train = pd.read_csv(data_path + "train.csv")
X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]
logisticRegr = LogisticRegression(solver='lbfgs').fit(X, y)

df_test = pd.read_csv(data_path + "test.csv")
predictions = logisticRegr.predict(df_test)
submission = pd.DataFrame({
    "ImageId": range(1, 1 + len(predictions)),
    "Label": predictions
})
# submission.insert(0, 'ImageId', range(1, 1 + len(submission)))
submission.to_csv("mnist-submission5.csv", index=False)
