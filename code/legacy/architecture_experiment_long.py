import ensemble
import torch
import dataset
import numpy as np
import torch.utils.data
import noiser
import metrics
import time
import loss_sep_sha
from tqdm import tqdm
import pandas as pd
import random


if __name__ == '__main__':
    LEARNING_RATE = 1e-1
    BATCH_SIZE = 2048
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.1
    EMBEDDING_SIZE = 2
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"


    dataset_full = dataset.Dataset("./data/loan_squeak.pkl")
    dataset_full.Y.to(DEVICE)

    dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(
        dataset_full,
        [TRAIN_SPLIT, VALIDATION_SPLIT, 1 - TRAIN_SPLIT - VALIDATION_SPLIT],
        generator=torch.Generator().manual_seed(0),
    )


    def class_weights():
        y_train_indices = dataset_train.indices

        y_train = [dataset_full.Y_idx[i] for i in y_train_indices]

        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
        )

        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight


    samples_weight = class_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight.type("torch.FloatTensor"), len(samples_weight)
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=2,
        drop_last=True,
        sampler=sampler,
        #shuffle=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=False,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
    )
    count=4.0
    width=64.0
    depth=2.0
    epochs=10.0
    prev_count=0.0
    prev_width=0.0
    prev_depth=0.0
    prev_best_epoch=0.0
    prev_loss=1e8
    while True:
        if epochs<5: epochs=5.0
        if count<1:count=1.0
        if width<1:width=1.0
        if depth<1:depth=1.0
        n_members=int(count)
        depths=[int(depth)]*n_members #constant depth experiment
        widths=[int(width)]*n_members
        modes=["res"]*n_members
        model_name="architecture"+str(n_members)+"M_"+str(depths[0])+"X"+str(widths[0])
        model_ensemble = ensemble.Ensemble(
            input_size=(
                dataset_full.X.shape[1] + dataset_full.X_classes.shape[1] * EMBEDDING_SIZE
            ),
            output_size=len(dataset_full.labels[-1]),
            depths=depths,
            widths=widths,
            modes=modes,
            n_members=n_members,
            device=DEVICE,
        )
        model = ensemble.EmbAndEnsemble(
            dataset_labels=dataset_full.labels,
            embedding_size=EMBEDDING_SIZE,
            ensemble=model_ensemble,
        )
        model.to(DEVICE)



        loss_fn = loss_sep_sha.LossSeperate().to(DEVICE)
        optimizer = torch.optim.RAdam(model.parameters(), LEARNING_RATE)
        lowest_validation_loss = 1e16
        
        model_path = "./models/"+model_name+".pth"
        # TRAINING
        for epoch in range(1, int(epochs)):
            epoch_validation_losses = []
            for dataloader in [dataloader_train, dataloader_valid]:

                if dataloader == dataloader_valid:
                    model = model.eval()
                    torch.set_grad_enabled(False)
                else:
                    model = model.train()
                    torch.set_grad_enabled(True)
                for x, x_classes, y in tqdm(iter(dataloader)):

                    y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))

                    loss = loss_fn.forward(y_prims, y.to(DEVICE))
                    if dataloader == dataloader_train:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    elif dataloader == dataloader_valid:
                        epoch_validation_losses.append(loss.item())
                if dataloader == dataloader_valid:
                    epoch_loss = np.mean(epoch_validation_losses)
                    epoch_validation_losses.clear()
                    if epoch_loss < lowest_validation_loss:
                        lowest_validation_loss = epoch_loss
                        best_epoch=epoch
                        print(f"New best model {model_name}- loss:{epoch_loss} epoch:{epoch}")
                        torch.save(
                            {"model": model.state_dict(), "epoch": epoch, "loss": epoch_loss},
                            model_path,
                        )
        #MODEL PARAMETER COMPARSION
        if lowest_validation_loss<prev_loss:
            best_model_path=model_path  # Update path to be unique for each model type
            best_count=count
            count+=(count-prev_count)+random.random()
            prev_count=best_count
            best_depth=depth
            depth+=(depth-prev_depth)+random.random()
            prev_depth=best_depth
            best_width=width
            width+=(width-prev_width)+random.random()
            prev_width=best_width
            epochs+=(best_epoch-prev_best_epoch)+random.random()
            prev_best_epoch=best_epoch
        elif lowest_validation_loss>prev_loss:
            count-=(count-prev_count)
            depth-=(depth-prev_depth)
            width-=(width-prev_width)
            epochs-=(best_epoch-epochs)




        