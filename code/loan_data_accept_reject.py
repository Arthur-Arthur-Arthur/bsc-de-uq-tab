#%%
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def shrink_data():
    df_full = pd.read_csv('./data/accepted_2007_to_2018Q4.csv')

    df_cent=df_full.iloc[:(int(df_full.shape[0]/10000))]
    df_cent.to_csv('./data/accepted_tiny.csv')

    df_full = pd.read_csv('./data/rejected_2007_to_2018Q4.csv')

    df_cent=df_full.iloc[:(int(df_full.shape[0]/10000))]
    df_cent.to_csv('./data/rejected_tiny.csv')
def type_sort(df):
        # Get the data types of each column
    data_types = df.dtypes

    # Sort the data types
    sorted_data_types = data_types.sort_values()

    # Reorder the columns based on the sorted data types
    sorted_df = df[sorted_data_types.index]
    return sorted_df
def remove_singular(df):
    df_unique = df[[c for c
        in list(df)
        if df[c].nunique() > 1]]
    return df_unique
def remove_all_unique(df):
    df_unique = df[[c for c
        in list(df)
        if df[c].nunique() < (df[c].count()*0.5) or df[c].dtype !=('object')]]
    return df_unique
def remove_too_frequent(df):
    # Calculate the most frequent value in each column
    most_frequent = df.mode(axis=0).iloc[0]

    # Calculate the percentage of the most frequent value in each column
    percentage_most_frequent = (df == most_frequent).sum(axis=0) / df.count()

    # Define the threshold (e.g., 90%)
    threshold = 0.9

    # Drop columns where the percentage of the most frequent value is above the threshold
    df = df.loc[:, percentage_most_frequent < threshold]
    return df

def strings_to_labels(df_strings):
    #df_strings contains only objects
    df_strings=df_strings.astype('str',errors='raise')
    df_idx=df_strings.copy()
    labels=[]
    for column in range(len(df_strings.columns)):
        series=df_strings.iloc[:,column]
        freqs=series.value_counts()
        these_labels=freqs.index.tolist()
        labels.append(these_labels)
        indexes=df_strings.iloc[:,column].apply(lambda x:these_labels.index(x) ) 
        df_idx.iloc[:,column]=indexes
    return df_idx, labels 

df_small= pd.read_csv('./data/accepted_tiny.csv')

df_small = df_small.dropna(axis='columns')

df_unique = remove_singular(df_small)

df_unique=type_sort(df_unique)

df_unique=remove_too_frequent(df_unique)
df_unique=remove_all_unique(df_unique)
df_unique=df_unique.drop(columns=['id','Unnamed: 0'])
temp_cols=df_unique.columns.tolist()
index=df_unique.columns.get_loc("loan_status")
new_cols=temp_cols[0:index] + temp_cols[index+1:]+temp_cols[index:index+1] 
df_unique=df_unique[new_cols]
df_unique.iloc[:,62:],labels=strings_to_labels(df_unique.iloc[:,62:])
df_x=(df_unique.drop(columns='loan_status')).to_numpy()
df_y=df_unique['loan_status'].to_numpy()
np_unique=df_unique.to_numpy()
np_unique
myvar=(df_x,df_y,labels)
with open('./data/loan_small.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(myvar, file) 

#df_unique.to_csv('./data/accept_small_clean.csv')
#df_unique.to_pickle('./data/small_loan.pkl')    #to save the dataframe, df to 123.pkl

describe=df_unique.describe(include='all')
describe
#df_unique['pct_tl_nvr_dlq'].plot.hist( bins=10, color='blue')
#df_unique['pct_tl_nvr_dlq']

df_unique['loan_status'].value_counts().plot(kind='bar')
print(df_unique['pct_tl_nvr_dlq'].describe())
plt.title('Histogram of Column Name')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.show()
# %%
