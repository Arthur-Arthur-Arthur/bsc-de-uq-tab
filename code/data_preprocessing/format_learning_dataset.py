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
        lower = Q1 - 3*IQR
        upper = Q3 + 3*IQR
        
        df_column=df[column]
        upper_array = np.where(df_column.values >= upper+1e-8)[0]
        lower_array=(np.where(df_column.values <= lower-1e-8)[0])     
        drop_array=np.concatenate([lower_array,upper_array],axis=None)   
        # Removing the outliers
        df.drop(index=drop_array, inplace=True)
        df.reset_index(drop=True,inplace=True)

    return df
def winsorize_outliers(df:pd.DataFrame):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3*IQR
        upper = Q3 + 3*IQR
        
        df_column=df[column]
        #upper_array = np.where(df_column.values >= upper+1e-8)[0]
        df_column.where(df_column.values <= upper,upper,inplace=True)
        #lower_array=(np.where(df_column.values <= lower-1e-8)[0])   
        df_column.where(df_column.values >=lower,lower,inplace=True)  
        #drop_array=np.concatenate([lower_array,upper_array],axis=None)   
        # Removing the outliers
        #df.drop(index=drop_array, inplace=True)
        df.reset_index(drop=True,inplace=True)

    return df
def format_dataset(df:pd.DataFrame, y_name):
    df.dropna(axis='columns',how='all',inplace=True)
    df.dropna(subset='loan_status',inplace=True)
    #df=df.drop(columns='id')
    df=remove_singular(df)
    df=limit_embedding_count(df,1000)
    df=remove_too_sparse(df,0.5)
    df=merge_other(df,0.1)
    df.update(df.select_dtypes(include=['object']).fillna(df.select_dtypes(include=['object']).mode()))
    df.update(df.select_dtypes(include=['number']).fillna(df.select_dtypes(include=['number']).mean()))
    #df=remove_outliers(df)
    #df=winsorize_outliers(df)
    df=type_sort(df)
    df=move_y_to_last(df,y_name)

    return df
def table_to_learn(df,y_name):
    df,labels=strings_to_labels(df)
    df_x,df_y=split_xy(df,y_name)

    return df_x,df_y,labels

def specific_cleanup(df:pd.DataFrame):
    bad_loan = [ "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period",
            "Late (16-30 days)", "Late (31-120 days)"]
    good_loan="Does not meet the credit policy. Status:Fully Paid"
    df['loan_status'].replace(bad_loan,"Charged Off",inplace=True)
    df['loan_status'].replace(good_loan,"Fully Paid",inplace=True)
    emp_length_mapping = {
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0.5,
    'n/a': 0
}
    df['emp_length_float'] = df['emp_length'].map(emp_length_mapping)
    df.drop(['emp_length'], axis=1, inplace=True)
    term_mapping = {' 36 months':36.0,' 60 months':60.0}
    df['term_float'] = df['term'].map(term_mapping)
    df.drop(['term'], axis=1, inplace=True)

    direct_indicators = [
        'collection_recovery_fee',
        'last_pymnt_amnt',
        'out_prncp',
        'out_prncp_inv',
        'recoveries',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_int',
        'total_rec_late_fee',
        'total_rec_prncp'
    ]
    df.drop(direct_indicators, axis=1, inplace=True)
    
    df.drop(['id','emp_title','url','title','zip_code','grade','desc'], axis=1, inplace=True)

    df['sub_grade']=df['sub_grade'].apply(sub_grade_to_num)
    for date in ['issue_d','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','debt_settlement_flag_date','settlement_date','sec_app_earliest_cr_line','hardship_start_date','hardship_end_date','payment_plan_start_date']:
        df[date]=df[date].apply(date_to_num)
    print(df.select_dtypes(include=["object"]).nunique())
    return df



def sub_grade_to_num(letter_and_num):
    if not isinstance(letter_and_num,str):
        return np.nan
    return (ord(letter_and_num[0]) - 64)*5+int(letter_and_num[1])

def date_to_num(date:str):
    if not isinstance(date,str):
        return np.nan
    month,year=(*date.split('-'),)
    month_mapping = {
    'Jan': 0,
    'Feb': 1,
    'Mar': 2,
    'Apr': 3,
    'May': 4,
    'Jun': 5,
    'Jul': 6,
    'Aug': 7,
    'Sep': 8,
    'Oct': 9,
    'Nov': 10,
    'Dec': 11
}
    return (int(year))-2000+month_mapping[month]/12.0

csv_load_path="./data/accepted_2007_to_2018Q4.csv"
pkl_load_path="./data/accepted_pickle.pkl"
csv_save_path="./data/clean.csv"
pkl_save_path="./data/loan_squeak.pkl"
load_from_csv=False
save_to_csv=True
if load_from_csv:
    df= pd.read_csv(csv_load_path,low_memory=False)  
else:
    with open(f"{pkl_load_path}", "rb") as fp:
        df = pickle.load(fp)

df=shrink_data(df,1)
print(df.head())
df=specific_cleanup(df)
df_formed=format_dataset(df,'loan_status') 
print(df_formed.select_dtypes(include=["object"]).nunique())
if save_to_csv:
    df_formed.to_csv(csv_save_path,index=False)
dset=table_to_learn(df=df_formed,y_name='loan_status')

with open(pkl_save_path, 'wb') as file: 
    pickle.dump(dset, file) 



