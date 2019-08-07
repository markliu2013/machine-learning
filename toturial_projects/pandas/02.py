import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/tips.csv'
df_tips = pd.read_csv(url)

print(df_tips.loc[1:3])

