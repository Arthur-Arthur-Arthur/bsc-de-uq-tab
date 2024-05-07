import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

csv_path="./data/clean_outliers.csv"
pkl_path="./data/loan_ml.pkl"
df= pd.read_csv(csv_path,low_memory=False)  
#plt.show()
for column in df.select_dtypes(include=['number']).columns:
    df[column].plot.hist()
    plt.title('Histogram of '+str(column))
    plt.tight_layout()
    plt.savefig("./figs/wins/float_"+str(column)+'_hist.png')
    plt.clf()
#plt.show()
for column in df.select_dtypes(exclude=['number']).columns:
    if df[column].nunique()<1000:
        df[column].value_counts().plot(kind='bar')
        plt.title('Category distribution of '+str(column))
        plt.tight_layout()
        plt.savefig("./figs/wins/orig_string_"+str(column)+'_hist.png')
        plt.clf()




