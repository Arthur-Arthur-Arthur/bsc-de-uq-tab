import pandas as pd
import numpy as np

#df_full = pd.read_csv('./data/loan.csv')
#print(df_full.shape)
#df_cent=df_full.iloc[:(int(df_full.shape[0]/10000))]
#df_cent.to_csv('./data/loan_tiny.csv')
df_cent= pd.read_csv('./data/loan_tiny.csv')
print(df_cent.shape)

df_cent = df_cent.dropna(axis='columns')

df_y = df_full.iloc[:,'SalePrice']

df_y = df_full.ix[:,'SalePrice']
df_x = df_full.ix[:,['YearBuilt', 'OverallQual']]

df_x = df_full.ix[:100, 'Col']
df_x = df_full.ix[:100] # whole rows

df_y = df_full.loc['SalePrice']

np_dataset_x = df_x.values