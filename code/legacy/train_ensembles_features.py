import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm
import pandas as pd

import ensemble
import dataset
import noiser
import metrics
import time
import loss_sep_sha


if __name__ == "__main__":
    NUM_TRAININGS = 5
    N_MEMBERS = 10
    MAX_EPOCHS = 10
    LEARNING_RATE = 1e-1
    BATCH_SIZE = 2048
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.1
    EMBEDDING_SIZE = 2
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"
    for training in range(NUM_TRAININGS):
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
            batch_size=BATCH_SIZE*N_MEMBERS, #required that batch is divisible by member amount
            num_workers=2,
            drop_last=True,
            sampler=dataset.equal_sampler(dataset_train, dataset_full),
            # shuffle=True
        )
        dataloader_valid = torch.utils.data.DataLoader(
            dataset=dataset_valid,
            batch_size=BATCH_SIZE,
            num_workers=2,
            sampler=dataset.equal_sampler(dataset_valid, dataset_full),
            # shuffle=False
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
        )

        n_members = N_MEMBERS
        depths = [16] * n_members  # constant depth experiment
        widths = [64] * n_members
        modes = ["res"] * n_members
        model_name = (
            str(training)
            + "RUN_FEx"
            + str(n_members)
            + "M_"
            + str(depths[0])
            + "X"
            + str(widths[0])
        )
        model_ensemble = ensemble.Ensemble(
            input_size=(
                dataset_full.X.shape[1]
                + dataset_full.X_classes.shape[1] * EMBEDDING_SIZE
            ),
            output_size=len(dataset_full.labels[-1]),
            depths=depths,
            widths=widths,
            modes=modes,
            n_members=n_members,
            device=DEVICE,
        )
        model_ensemble.feature_mask = ensemble.make_feature_mask(
            model_ensemble, 0.2*training
        ).to(DEVICE)
        model = ensemble.EmbAndEnsemble(
            dataset_labels=dataset_full.labels,
            embedding_size=EMBEDDING_SIZE,
            ensemble=model_ensemble,
        )
        model.to(DEVICE)

        loss_fn = loss_sep_sha.LossNLL().to(DEVICE)
        optimizer = torch.optim.RAdam(model.parameters(), LEARNING_RATE)
        lowest_validation_loss = 1e16

        best_model_path = "./models/best" + model_name + ".pth"
        last_model_path = "./models/last" + model_name + ".pth"
        # TRAINING
        training_plot = []
        for epoch in range(MAX_EPOCHS):
            epoch_validation_losses = []
            epoch_training_losses = []
            for dataloader in [dataloader_train, dataloader_valid]:
                if dataloader == dataloader_valid:
                    model = model.eval()
                    torch.set_grad_enabled(False)
                else:
                    model = model.train()
                    torch.set_grad_enabled(True)
                for x, x_classes, y in tqdm(iter(dataloader)):
                    y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
                    loss = loss_fn.forward(y_prim, y.to(DEVICE))
                    if dataloader == dataloader_train:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        epoch_training_losses.append(loss.item())
                    elif dataloader == dataloader_valid:
                        epoch_validation_losses.append(loss.item())
            valid_loss = np.mean(epoch_validation_losses)
            epoch_validation_losses.clear()
            train_loss = np.mean(epoch_training_losses)
            epoch_training_losses.clear()
            training_plot.append((train_loss, valid_loss))
            if valid_loss < lowest_validation_loss:
                lowest_validation_loss = valid_loss
                print(
                    f"New best model {model_name}- loss:{valid_loss} epoch:{epoch}"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "loss": valid_loss,
                    },
                    best_model_path,
                )
                # Update path to be unique for each model type

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "loss": valid_loss},
            last_model_path,
        )
        df = pd.DataFrame(training_plot, columns =['training_loss','validation_loss'])
        df.to_csv("./data/experiments_better/training_plot"+str(training)+".csv",)
