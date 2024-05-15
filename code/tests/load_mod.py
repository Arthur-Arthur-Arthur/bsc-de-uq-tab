import torch
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data/experiments_better/training_plot_1e-08_MOD_14-05-2024_23-49-59_.csv")
plt.plot(df.index,df["training_loss"],label="training_loss")
plt.plot(df.index,df["validation_loss"],label="validation_loss")
plt.legend()
plt.show()
best_model = torch.load("models/MOD/best_1e-08_MOD_14-05-2024_21-20-56_.pth")
print(best_model["epoch"])

data = pd.DataFrame()
data.at[0, "best_epoch"] = best_model["epoch"]