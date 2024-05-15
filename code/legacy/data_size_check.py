import pandas as pd
csv_load_path="./data/squeaky_clean2.csv"
df= pd.read_csv(csv_load_path,low_memory=False)  
print(df.shape)