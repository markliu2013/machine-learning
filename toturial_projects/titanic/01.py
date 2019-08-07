# 所有人都生还

import pandas as pd

df_test = pd.read_csv("data/test.csv")

FinalResult = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": '0'
})

FinalResult.to_csv("titanic-submission.csv", index=False)
