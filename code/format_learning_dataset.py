import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def shrink_data(df_full,divide):
    df_small=df_full.iloc[:(int(df_full.shape[0]/divide))]
    return df_small
def type_sort(df):
    data_types = df.dtypes
    sorted_data_types = data_types.sort_values()
    sorted_df = df[sorted_data_types.index]
    return sorted_df
def remove_singular(df):
    df = df[[c for c
        in list(df)
        if df[c].nunique() > 1]]
    return df
def remove_too_sparse(df:pd.DataFrame,min_real):
    df = df[[c for c
        in list(df)
        if (df[c].count()-df[c].isin([0.0]).sum())/df.shape[0] > min_real]]
    return df
def limit_embedding_count(df,max_embeddings):
    df = df[[c for c
        in list(df)
        if df[c].nunique() < max_embeddings or df[c].dtype !=('object')]]
    return df
def strings_to_labels(df:pd.DataFrame):
    df_strings = df.select_dtypes(include=['object'])
    df_strings=df_strings.fillna("Missing Value")
    df_idx=df_strings.copy()
    labels=[]
    for column in range(len(df_strings.columns)):
        series=df_strings.iloc[:,column]
        freqs=series.value_counts()
        these_labels=freqs.index.tolist()
        labels.append(these_labels)
        indexes=df_strings.iloc[:,column].apply(lambda x:these_labels.index(x) ) 
        df_idx.isetitem(column,indexes)
    df.update(df_idx)
    return df, labels
def merge_other(df:pd.DataFrame,min_size): #min_size is minimum portion of avarage category size that a category can be before getting merged
    df_strings = df.select_dtypes(include=['object'])
    df_idx=df_strings.copy()
    for column in df_strings.columns:
        series=df_strings.loc[:,column]
        freqs=series.value_counts()
        avg_freq=freqs.mean()
        for freq in freqs.items():
            if freq[1]<avg_freq*min_size:
                df_strings[column]=df_strings[column].replace(freq[0],"Other")
    df.update(df_strings)
    return df
def move_y_to_last(df,y_name):
    temp_cols=df.columns.tolist()
    index=df.columns.get_loc(y_name)
    new_cols=temp_cols[0:index] + temp_cols[index+1:]+temp_cols[index:index+1] 
    df=df[new_cols]
    return df
def split_xy(df,y):
    df_x=(df.drop(columns=y)).to_numpy()
    df_y=df[y].to_numpy()
    return df_x,df_y
def remove_outliers(df:pd.DataFrame):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(df[column] >= upper+1e8)[0]
        lower_array = np.where(df[column] <= lower-1e8)[0]
        
        # Removing the outliers
        df.drop(index=upper_array, inplace=True)
        df.drop(index=lower_array, inplace=True)
    return df
def format_dataset(df:pd.DataFrame, y_name):
    df = df.dropna(axis='columns',how='all')
    df=df.drop(columns='id')
    df=remove_singular(df)
    df=limit_embedding_count(df,1000)
    df=merge_other(df,0.1)
    df=remove_too_sparse(df,0.7)
    df.update(df.select_dtypes(include=['object']).fillna("Missing Value"))
    df.update(df.select_dtypes(include=['number']).fillna(df.select_dtypes(include=['number']).mean()))
    df=remove_outliers(df)
    df=type_sort(df)
    df=move_y_to_last(df,y_name)

    return df
def table_to_learn(df,y_name):
    df,labels=strings_to_labels(df)
    df_x,df_y=split_xy(df,y_name)

    return df_x,df_y,labels
csv_load_path="./data/accepted_2007_to_2018Q4.csv"
pkl_load_path="./data/accepted_pickle.pkl"
csv_save_path="./data/accepted_clean.csv"
pkl_save_path="./data/loan_ml100.pkl"
load_from_csv=False
save_to_csv=False
if load_from_csv:
    df= pd.read_csv(csv_load_path,low_memory=False)  
else:
    with open(f"{pkl_load_path}", "rb") as fp:
        df = pickle.load(fp)
df=shrink_data(df,100)
df_formed=format_dataset(df,'loan_status') 
if save_to_csv:
    df_formed.to_csv(csv_save_path,index=False)
dset=table_to_learn(df=df_formed,y_name='loan_status')

with open(pkl_save_path, 'wb') as file: 
    pickle.dump(dset, file) 



