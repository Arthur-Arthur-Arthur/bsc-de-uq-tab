import pandas as pd


paths = [
    "data\experiments_better\search9M_WIDE_20.011024_grid_results.csv",
    "data\experiments_better\search9M_WIDE_20.0011024_grid_results.csv",
    "data\experiments_better\search9M_WIDE_20.00011024_grid_results.csv",
    "data\experiments_better\search9M_WIDE_20.014096_grid_results.csv",
    "data\experiments_better\search9M_WIDE_20.0014096_grid_results.csv",
    "data\experiments_better\search9M_WIDE_20.00014096_grid_results.csv",
]
df_array=[]
for path in paths:
    df = pd.read_csv(
        path, low_memory=False
    )
    df['path']=path
    df_array.append(df)
combo_df=pd.concat(df_array)
combo_df=combo_df.sort_values("loss")
print(combo_df)