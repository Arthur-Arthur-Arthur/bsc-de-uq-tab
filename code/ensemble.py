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
from collections import Counter


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
        path_dataset = './data/loan_small.pkl'
        
        with open(f'{path_dataset}', 'rb') as fp:
            X, self.Y, self.labels = pickle.load(fp)

        X = np.array(X)

        self.Y_idx = self.Y
        
        self.Y_labels = self.labels[-1]
        self.Y_len = len(self.Y_labels)

        Y_counter = Counter(self.Y)
        Y_counter_val = np.array(list(Y_counter.values()))
        self.Y_weights = (10 / Y_counter_val) * np.sum(Y_counter_val)

        self.Y_idx=F.one_hot(torch.LongTensor(self.Y))

        self.X_classes = np.array(X[:, 62:]).astype(np.int32)

        self.X = np.array(X[:, :62]).astype(np.float32)  # VERY IMPORTANT OTHERWISE NOT ENOUGH CAPACITY
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


        # y_owner,
        # y_selling_price

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y_idx[idx]

dataset_full = Dataset()
check=dataset_full.X.shape[1]+dataset_full.X_classes.shape[1]*EMBEDDING_SIZE

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
            torch.nn.Linear(in_features=dataset_full.X.shape[1]+dataset_full.X_classes.shape[1]*EMBEDDING_SIZE,out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64,out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32,out_features=len(dataset_full.labels[-1])),
            torch.nn.Softmax()
        )

        self.embs=torch.nn.ModuleList()
        for i in range(len(dataset_full.labels)):
            # x_brands, x_fuel, x_transmission, x_seller_type
            self.embs.append(
                torch.nn.Embedding(embedding_dim=EMBEDDING_SIZE,num_embeddings=len(dataset_full.labels[i]))
            )

    def forward(self, x, x_classes):
        x_enc=[]
        for i in range(x_classes.shape[1]):
            x_enc.append(self.embs[i].forward(x_classes[:,i]))
        x_enc.append(x)
        x_cat=torch.cat((x_enc),dim=1)
        y_prim = self.layers.forward(x_cat)

        #TODO
        return y_prim


model = Model()
optimizer = torch.optim.SGD(
    model.parameters(),
    LEARNING_RATE
)


class LossBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        w=torch.Tensor(dataset_full.Y_weights)
        loss = -torch.mean( w*y*torch.log(y_prim+1e-8)+(1-y)*torch.log(1-y_prim+1e-8))
        return loss

class LossCCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        loss=-torch.mean(y*torch.log(y_prim+1e-8))
        return loss
loss_fn=LossBCE()


loss_plot_train = []
loss_plot_test = []

acc_plot_train = []
acc_plot_test = []

f1_plot_train = []
f1_plot_test = []

conf_matrix_train = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
conf_matrix_test = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader == dataloader_test:
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            model = model.train()
            torch.set_grad_enabled(True)

        losses = []
        accs = []

        conf_matrix = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
        for x, x_classes, y_idx in dataloader:
            
            y_prim_idx = model.forward(x, x_classes)

            
            loss=loss_fn.forward(y_prim=y_prim_idx,y=y_idx)
            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            y_idx = np.argmax(y_idx.data.numpy(),axis=1).astype(int)
            y_prim_idx = np.argmax(y_prim_idx.data.numpy(),axis=1).astype(int)
            acc = np.mean((y_idx == y_prim_idx) * 1.0)
            accs.append(acc)

            for idx in range(len(y_prim_idx)):
                conf_matrix[y_prim_idx[idx], y_idx[idx]] += 1

       


        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            conf_matrix_train = conf_matrix

        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            conf_matrix_test = conf_matrix


    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]} '

    )

    if epoch % 10 == 0:

        plt.tight_layout(pad=0)

        fig, axes = plt.subplots(nrows=2, ncols=2)

        ax1 = axes[0, 0]
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[0, 1]
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        
        for ax, conf_matrix in [(axes[1, 0], conf_matrix_train), (axes[1, 1], conf_matrix_test)]:
            ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
            ax.set_xticks(np.arange(dataset_full.Y_len), dataset_full.Y_labels, rotation=45)
            ax.set_yticks(np.arange(dataset_full.Y_len), dataset_full.Y_labels)
            for x in range(dataset_full.Y_len):
                for y in range(dataset_full.Y_len):
                    perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix))
                    ax.annotate(
                        str(int(conf_matrix[x, y])),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor=(1., 1., 1., 0.),
                        color='black' if perc < 50 else 'white',
                        fontsize=10
                    )
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')


        plt.show()
# %%
