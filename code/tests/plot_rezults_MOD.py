import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

experiments_00 = [
    "data/experiments_better/best_0e+00_Mask_14-05-2024_21-15-39_.pth_distance__results.csv",
"data/experiments_better/best_0e+00_Mask_14-05-2024_21-15-39_.pth_noise__results.csv",
"data/experiments_better/best_0e+00_Mask_15-05-2024_03-30-00_.pth_distance__results.csv",
"data/experiments_better/best_0e+00_Mask_15-05-2024_03-30-00_.pth_noise__results.csv"
]
experiments_05 = [
    "data/experiments_better/best_5e-01_Mask_14-05-2024_23-22-07_.pth_distance__results.csv",
"data/experiments_better/best_5e-01_Mask_14-05-2024_23-22-07_.pth_noise__results.csv",
"data/experiments_better/best_5e-01_Mask_15-05-2024_05-00-33_.pth_distance__results.csv",
"data/experiments_better/best_5e-01_Mask_15-05-2024_05-00-33_.pth_noise__results.csv"]
experiments_10 = [
"data/experiments_better/best_1e+00_Mask_15-05-2024_01-32-32_.pth_distance__results.csv",
"data/experiments_better/best_1e+00_Mask_15-05-2024_01-32-32_.pth_noise__results.csv",
"data/experiments_better/best_1e+00_Mask_15-05-2024_06-33-09_.pth_distance__results.csv",
"data/experiments_better/best_1e+00_Mask_15-05-2024_06-33-09_.pth_noise__results.csv"
]

for type in ["noise", "distance"]:
    experiment_dfs = []
    df_means_arr = []
    for i, experiment_group in enumerate([experiments_00,experiments_05,experiments_10]):
        experiment_dfs.append([])
        for experiment_path in experiment_group:
            exp_df = pd.read_csv(experiment_path)
            if type in experiment_path:
                experiment_dfs[i].append(exp_df)
        df_concat = pd.concat(experiment_dfs[i])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means_arr.append(by_row_index.mean())
    stats = ["ECE", "loss", "accuracy", "confidence"]
    cmap = plt.get_cmap(name="hsv", lut=15)
    for i, stat in enumerate(stats):
        plt.figure(figsize=[12.8,9.6])
        for j,overlap in enumerate([0.0,0.5,1.0]):
            sub_j=0
            for column in df_means_arr[j].drop(columns=["time", "best_epoch"]).columns:
                if stat in column:
                    sub_j+=1
                    sub_j%=3
                    x = df_means_arr[j].index
                    y = df_means_arr[j][column]
                    plt.plot(x, y, label=column+" with "+str(overlap) +"overlap", color=cmap(5*j+sub_j), linewidth=2, marker="o")
                    for exp_df in experiment_dfs[j]:
                        if column in exp_df.columns:
                            y_i = exp_df[column]
                            plt.plot(x, y_i, color=cmap(5*j+sub_j), alpha=0.5, linewidth=0.7)
        plt.title(stat+" using "+type)
        plt.legend()
        
        plt.savefig(fname="./figs/experiments/mask"+type+"_"+stat+".png")
        plt.show()
       
