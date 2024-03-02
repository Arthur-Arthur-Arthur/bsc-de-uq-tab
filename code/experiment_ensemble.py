# %%
import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import ensemble
import dataset
import metrics
from tqdm import tqdm

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.5
EMBEDDING_SIZE = 8
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


dataset_full = dataset.Dataset("./data/loan_ml100.pkl")

train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0),
)
y_train_indices = dataset_train.indices

y_train = [dataset_full.Y_idx[i] for i in y_train_indices]

class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)

sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1),
    sampler=sampler
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
model_ensemble = ensemble.Ensemble(
    input_size=(
        dataset_full.X.shape[1] + dataset_full.X_classes.shape[1] * EMBEDDING_SIZE
    ),
    output_size=len(dataset_full.labels[-1]),
    depths=[16,8,4],
    widths=[8,16,32],
    n_members=3,
    device=DEVICE
)
model = ensemble.EmbAndEnsemble(
    dataset_labels=dataset_full.labels, embedding_size=EMBEDDING_SIZE, ensemble=model_ensemble
)
optimizer = torch.optim.RAdam(model.parameters(), LEARNING_RATE)


class LossShared(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        loss = -torch.mean(y * torch.log(y_prim + 1e-8))
        return loss
    
class LossSeperate(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prims, y):
        y=y.unsqueeze(dim=0)
        y=y.expand(y_prims.size())
        loss = -torch.mean(y * torch.log(y_prims + 1e-8))
        return loss

loss_fn = LossSeperate()



loss_plot_train = []
loss_plot_test = []

acc_plot_train = []
acc_plot_test = []

ece_plot_train = []
ece_plot_test = []

std_plot_train = []
std_plot_test = []

conf_matrix_train = np.zeros((len(dataset_full.Y_labels), len(dataset_full.Y_labels)))
conf_matrix_test = np.zeros((len(dataset_full.Y_labels), len(dataset_full.Y_labels)))
for epoch in range(1, 1000):
    epoch_start=time.time()
    for dataloader in [dataloader_train, dataloader_test]:

        if dataloader == dataloader_test:
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            model = model.train()
            torch.set_grad_enabled(True)

        losses = []
        accs = []
        stds=[]
        y_prims_for_ece=np.zeros((1,4))
        y_for_ece=np.zeros((1,4))
        conf_matrix = np.zeros((len(dataset_full.Y_labels), len(dataset_full.Y_labels)))
        for x, x_classes, y in tqdm(iter(dataloader)):

            y_prims = model.forward(x, x_classes) #x izmērs ir 48, x_classes 20, kā jābūt, iepriekš bija kļūda

            loss = loss_fn.forward(y_prims, y)
            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            y_idx = np.argmax(y.data.numpy(), axis=1).astype(int)
            y_prim_idx = np.argmax(torch.mean(y_prims,0).data.numpy(), axis=1).astype(int)
            acc = np.mean((y_idx == y_prim_idx) * 1.0)
            accs.append(acc)
            y_for_ece=np.append(y_for_ece,y.data.numpy(),0)
            y_prims_for_ece=np.append(y_prims_for_ece,torch.mean(y_prims,0).data.numpy(),0)
            stds.append(np.mean(np.std(y_prims.data.numpy(),1)))
            for idx in range(len(y_prim_idx)):
                conf_matrix[y_prim_idx[idx], y_idx[idx]] += 1

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            std_plot_train.append(np.mean(stds))
            ece_plot_train.append(metrics.calc_ece(y_for_ece,y_prims_for_ece,10))
            conf_matrix_train = conf_matrix


        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            std_plot_test.append(np.mean(stds))
            ece_plot_test.append(metrics.calc_ece(y_for_ece,y_prims_for_ece,10))
            conf_matrix_test = conf_matrix

    epoch_end=time.time()
    print(
        f"epoch: {epoch} "
        f"loss_train: {loss_plot_train[-1]} "
        f"loss_test: {loss_plot_test[-1]} "
        f"acc_train: {acc_plot_train[-1]} "
        f"acc_test: {acc_plot_test[-1]} "
        f"std_train: {std_plot_train[-1]} "
        f"std_test: {std_plot_test[-1]} "
        f"ece_train: {ece_plot_train[-1]} "
        f"ece_test: {ece_plot_test[-1]} "

        f"epoch_time: {epoch_end-epoch_start} "
    )

    if epoch % 2 == 0:


        fig, axes = plt.subplots(nrows=3, ncols=2)

        ax1 = axes[0,0]
        ax1.plot(loss_plot_train, "r-", label="train")
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, "c-", label="test")
        ax1.legend()
        ax2.legend(loc="upper left")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[0,1]
        ax1.plot(acc_plot_train, "r-", label="train")
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, "c-", label="test")
        ax1.legend()
        ax2.legend(loc="upper left")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")

        ax1 = axes[1,0]
        ax1.plot(std_plot_train, "b-", label="train")
        ax2 = ax1.twinx()
        ax2.plot(std_plot_test, "v-", label="test")
        ax1.legend()
        ax2.legend(loc="upper left")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("STD")

        ax1 = axes[1,1]
        ax1.plot(ece_plot_train, "g-", label="train")
        ax2 = ax1.twinx()
        ax2.plot(ece_plot_test, "y-", label="test")
        ax1.legend()
        ax2.legend(loc="upper left")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("ECE")

        for ax, conf_matrix in [(axes[2, 0], conf_matrix_train), (axes[2, 1], conf_matrix_test)]:
            ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
            ax.set_xticks(np.arange(len(dataset_full.Y_labels)), dataset_full.Y_labels, rotation=45)
            ax.set_yticks(np.arange(len(dataset_full.Y_labels)), dataset_full.Y_labels)
            for x in range(len(dataset_full.Y_labels)):
                for y in range(len(dataset_full.Y_labels)):
                    perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix))
                    ax.annotate(
                        str(int(conf_matrix[x, y])),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor=(1., 1., 1., 0.),
                        color='black' if perc < 20 else 'white',
                        fontsize=10
                    )
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
        plt.show()
# %%
