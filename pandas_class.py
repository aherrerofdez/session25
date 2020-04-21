import numpy as np
import pandas as pd

s = pd.Series([1, 3.5, 5, np.nan, 6, 8])
print(s.T)

dates = pd.date_range('20200420', periods=7)
print(dates)

df = pd.DataFrame(np.random.randn(7, 4), index=dates, columns=["AA", "BB", "CC", "DD"])
print(df)

print("using loc")
print(df.loc[dates[2]:dates[4], ["BB", "CC"]])
df.loc[dates[3], ["BB"]] = 77
print("with iloc")
print(df.iloc[2:5, [1, 3]])
df.iloc[0, 0] = 111
df.to_csv("my_csv.csv")

df2 = pd.read_csv("my_csv.csv")
print(df2)