# 所有女性都生还

import pandas as pd

df_test = pd.read_csv("data/test.csv")

FinalResult = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": pd.factorize(df_test['Sex'])[0]
})

FinalResult.to_csv("titanic-submission.csv", index=False)
