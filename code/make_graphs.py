import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

csv_path="./data/accepted_2007_to_2018Q4.csv"
pkl_path="./data/loan_ml.pkl"
df= pd.read_csv(csv_path,low_memory=False)  
plt.show()
for column in range(0):# df.select_dtypes(include=['number']).columns:
    df[column].plot.hist()
    plt.title('Histogram of '+str(column))
    plt.savefig("./figs/"+str(column)+'_hist.png')
    plt.clf()
plt.show()
for column in df.select_dtypes(exclude=['number']).columns:
    df[column].value_counts().plot(kind='bar')
    plt.title('Category distribution of '+str(column))
    plt.savefig("./figs/string"+str(column)+'_hist.png')
    plt.clf()




