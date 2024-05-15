import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

bad_experiments = [
    "data/experiments_better/best_10_Bad_13-05-2024_21-41-33_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_13-05-2024_21-41-33_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_13-05-2024_22-00-45_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_13-05-2024_22-00-45_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_13-05-2024_23-29-13_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_13-05-2024_23-29-13_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_00-56-38_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_00-56-38_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_02-24-00_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_02-24-00_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_03-51-57_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_03-51-57_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_05-19-16_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_05-19-16_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_06-46-42_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_06-46-42_.pth_noise__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_08-13-51_.pth_distance__results.csv",
    "data/experiments_better/best_10_Bad_14-05-2024_08-13-51_.pth_noise__results.csv",
]
good_experiments = [
    "data/experiments_better/best_10_Well_13-05-2024_01-23-45_.pth_distance__results.csv",
    "data/experiments_better/best_10_Well_13-05-2024_01-23-45_.pth_noise__results.csv",
    "data/experiments_better/best_10_Well_13-05-2024_04-07-36_.pth_distance__results.csv",
    "data/experiments_better/best_10_Well_13-05-2024_04-07-36_.pth_noise__results.csv",
]

for type in ["noise", "distance"]:
    experiment_dfs = []
    df_means_arr = []
    for i, experiment_group in enumerate([bad_experiments,good_experiments]):
        experiment_dfs.append([])
        for experiment_path in experiment_group:
            exp_df = pd.read_csv(experiment_path)
            if type in experiment_path:
                experiment_dfs[i].append(exp_df)
        df_concat = pd.concat(experiment_dfs[i])
        by_row_index = df_concat.groupby(df_concat.index)
        df_means_arr.append(by_row_index.mean())
    df_means = df_means_arr[0].join(df_means_arr[1], how="outer", rsuffix="_good")
    stats = ["ECE", "loss", "accuracy", "confidence"]
    cmap = plt.get_cmap(name="hsv", lut=5)
    for i, stat in enumerate(stats):
        sub_i = -1
        for column in df_means.drop(columns=["time", "best_epoch"]).columns:
            if stat in column and "T_scaled" not in column:
                sub_i += 1
                sub_i %= 4
                x = df_means.index
                y = df_means[column]
                plt.plot(x, y, label=column, color=cmap(sub_i), linewidth=2, marker="o")
                for exp_df in experiment_dfs[0]:
                    if column in exp_df.columns:
                        y_i = exp_df[column]
                        plt.plot(x, y_i, color=cmap(sub_i), alpha=0.5, linewidth=0.7)
                for exp_df in experiment_dfs[1]:
                    if column in exp_df.columns:
                        y_i = exp_df[column]
                        plt.plot(
                            x, y_i, color=cmap(sub_i + 2), alpha=0.5, linewidth=0.5
                        )
        plt.title(stat+" using "+type)
        plt.legend()
        plt.savefig(fname="./figs/experiments/"+type+"_"+stat+".png")
        plt.show()
       
