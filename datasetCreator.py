import pandas as pd

df = pd.read_csv('data/SPY_max_daily.csv')
#df = df.set_index('Date')
print(df.iloc[:,0])

print(df.iloc[:int(len(df)*0.8),0])



