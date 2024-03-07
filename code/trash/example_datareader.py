# %%
import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7
EMBEDDING_SIZE = 8
HUBER_DELTA= 1


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset_2.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/cardekho_india_dataset_2.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X, self.Y, self.labels = pickle.load(fp)

        X = np.array(X)
        self.X_classes = np.array(X[:, :4])

        self.X = np.array(X[:, 4:]).astype(np.float32)  # VERY IMPORTANT OTHERWISE NOT ENOUGH CAPACITY
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std
        self.X = self.X.astype(np.float32)

        # x_brands,
        # x_fuel,
        # x_transmission,
        # x_seller_type,

        # x_year,
        # x_km_driven,

        self.Y = np.array(self.Y).astype(np.float64)
        Y_mean = np.mean(self.Y, axis=0)
        Y_std = np.std(self.Y, axis=0)
        self.Y = (self.Y - Y_mean) / Y_std
        self.Y = self.Y.astype(np.float32)

        # y_owner,
        # y_selling_price

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y[idx]

dataset_full = Dataset()

train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train)%BATCH_SIZE==1)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=2+4*EMBEDDING_SIZE,out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64,out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32,out_features=2),
        )
        self.deviation= torch.nn.Sequential(
            torch.nn.Linear(in_features=4+4*EMBEDDING_SIZE,out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64,out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32,out_features=2),
        )
        self.embs=torch.nn.ModuleList()
        for i in range(4):
            # x_brands, x_fuel, x_transmission, x_seller_type
            self.embs.append(
                torch.nn.Embedding(embedding_dim=EMBEDDING_SIZE,num_embeddings=len(dataset_full.labels[i]))
            )

    def forward(self, x, x_classes):
        x_enc=[]
        for i in range(4):
            x_enc.append(self.embs[i].forward(x_classes[:,i]))
        x_enc.append(x)
        x_cat=torch.cat((x_enc),dim=1)
        y_prim = self.layers.forward(x_cat)
        y_std=self.deviation.forward(torch.cat((x_cat,y_prim),dim=1))
        #TODO
        return y_prim,y_std


model = Model()
optimizer = torch.optim.SGD(
    model.parameters(),
    LEARNING_RATE
)

class LossMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        loss = torch.mean((y_prim-y)**2)
        return loss
class LossUQ(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y,y_prim,y_std):
        loss = torch.abs(torch.mean(torch.abs(y_prim-y)-y_std))
        return loss
class LossHuber(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        loss = torch.mean((HUBER_DELTA**2)*torch.sqrt(1+((y_prim-y)/HUBER_DELTA)**2)-1)
        return loss
loss_fn_mean=LossMSE()
loss_fn_std = LossUQ()

loss_plot_train = []
loss_plot_test = []
loss_plot_train_std = []
loss_plot_test_std = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader == dataloader_test:
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            model = model.train()
            torch.set_grad_enabled(True)

        losses_mean = []
        losses_std = []
        for x, x_classes, y in dataloader:
            y_prim,y_std= model.forward(x,x_classes)
            loss_mean=loss_fn_mean.forward(y_prim,y)
            loss_std=loss_fn_std.forward(y,y_prim,y_std)
            loss=loss_mean+loss_std
            losses_mean.append(loss_mean.item())
            losses_std.append(loss_std.item())
            if dataloader==dataloader_train:
                loss_mean.backward()
                optimizer.step() # W=W-dW*learning_rate
                optimizer.zero_grad()# resets d values to 0

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses_mean))
            loss_plot_train_std.append(np.mean(losses_std))
        else:
            loss_plot_test.append(np.mean(losses_mean))
            loss_plot_test_std.append(np.mean(losses_std))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}loss_train_std: {loss_plot_train_std[-1]} loss_test_std: {loss_plot_test_std[-1]}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax1.plot(loss_plot_train_std, 'g-', label='train_std')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax2.plot(loss_plot_test_std, 'b-', label='test_std')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()
# %%
