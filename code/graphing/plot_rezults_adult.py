import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

bad_experiments = [
"data/experiments_better/best_10_adult_15-05-2024_15-26-11_.pth_distance__results.csv",
"data/experiments_better/best_10_adult_15-05-2024_15-26-11_.pth_noise__results.csv",
"data/experiments_better/best_10_adult_15-05-2024_15-35-40_.pth_distance__results.csv",
"data/experiments_better/best_10_adult_15-05-2024_15-35-40_.pth_noise__results.csv",
"data/experiments_better/best_10_adult_15-05-2024_15-54-12_.pth_distance__results.csv",
"data/experiments_better/best_10_adult_15-05-2024_15-54-12_.pth_noise__results.csv"
]
good_experiments = [
]

for type in ["noise", "distance"]:
    experiment_dfs = []
    df_means_arr = []
    for experiment_path in bad_experiments:
        exp_df = pd.read_csv(experiment_path)
        if type in experiment_path:
            experiment_dfs.append(exp_df)
    df_concat = pd.concat(experiment_dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means=by_row_index.mean()

    stats = ["ECE", "loss", "accuracy", "confidence"]
    cmap = plt.get_cmap(name="hsv", lut=5)
    for i, stat in enumerate(stats):
        plt.figure(figsize=[12.8,9.6])

        sub_i = -1
        for column in df_means.drop(columns=["time", "best_epoch"]).columns:
            if stat in column:
                sub_i += 1
                sub_i %= 4
                x = df_means.index
                y = df_means[column]
                plt.plot(x, y, label=column, color=cmap(sub_i), linewidth=2, marker="o")
                for exp_df in experiment_dfs:
                    if column in exp_df.columns:
                        y_i = exp_df[column]
                        plt.plot(x, y_i, color=cmap(sub_i), alpha=0.5, linewidth=0.7)

        plt.title(stat+" using "+type)
        plt.legend()
        plt.savefig(fname="./figs/experiments/adult_"+type+"_"+stat+".png")
        plt.show()
        
