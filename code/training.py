import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils.data

import loss_sep_sha
import ensemble
import dataset
import noiser


def training_base(
    model: ensemble.EmbAndEnsemble,
    dataloaders_train,
    dataloader_valid,
    model_name,
    device,
    learning_rate,
    max_epochs,
):
    loss_fn = loss_sep_sha.LossNLL().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), learning_rate)

    lowest_validation_loss = 1e16  # for fininding best model
    best_model_path = (
        "./models/best_" + model_name + ".pth"
    )  # for best model throughout learning, by validation loss
    last_model_path = "./models/last_" + model_name + ".pth"
    # TRAINING
    training_plot = []
    for epoch in range(max_epochs):
        epoch_validation_losses = []
        epoch_training_losses = []

        for i, dataloader in enumerate(
            [*dataloaders_train, dataloader_valid]
        ):  # iterates through one training cycle for each member and one validation cycle
            model.ensemble.current_model_train = i  # specifies that only the ith member is to be trained (if validation, this doesn't matter anyway)

            if dataloader == dataloader_valid:
                model = model.eval()
                torch.set_grad_enabled(False)
            else:
                model = model.train()
                torch.set_grad_enabled(True)

            for x, x_classes, y in tqdm(iter(dataloader)):
                y_prim, y_prims = model.forward(x.to(device), x_classes.to(device))
                loss = loss_fn.forward(y_prim, y.to(device))

                if dataloader in dataloaders_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_training_losses.append(loss.item())
                elif dataloader == dataloader_valid:
                    epoch_validation_losses.append(loss.item())
        # TRACKING LOSSES
        valid_loss = np.mean(epoch_validation_losses)
        epoch_validation_losses.clear()
        train_loss = np.mean(epoch_training_losses)
        epoch_training_losses.clear()
        training_plot.append((train_loss, valid_loss))

        # CHECKING/SAVING BEST MODEL
        if valid_loss < lowest_validation_loss:
            lowest_validation_loss = valid_loss
            print(f"New best model {model_name}- loss:{valid_loss} epoch:{epoch}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "loss": valid_loss,
                },
                best_model_path,
            )

    torch.save(
        {"model": model.state_dict(), "epoch": epoch, "loss": valid_loss},
        last_model_path,
    )
    df = pd.DataFrame(training_plot, columns=["training_loss", "validation_loss"])
    df.to_csv(
        "./data/experiments_better/training_plot_" + model_name + ".csv",
    )


def training_bad(
    model: ensemble.EmbAndEnsemble,
    dataloader_train,
    dataloader_valid,
    model_name,
    device,
    learning_rate,
    max_epochs,
):
    model.ensemble.seperate_batches = False
    loss_fn = loss_sep_sha.LossSeperate().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), learning_rate)

    lowest_validation_loss = 1e16  # for fininding best model
    best_model_path = (
        "./models/best_" + model_name + ".pth"
    )  # for best model throughout learning, by validation loss
    last_model_path = "./models/last_" + model_name + ".pth"
    # TRAINING
    training_plot = []
    epoch = 0
    for epoch in range(max_epochs):
        epoch_validation_losses = []
        epoch_training_losses = []

        for i, dataloader in enumerate(
            [dataloader_train, dataloader_valid]
        ):  # iterates through one training cycle for each member and one validation cycle
            model.ensemble.current_model_train = i  # specifies that only the ith member is to be trained (if validation, this doesn't matter anyway)

            if dataloader == dataloader_valid:
                model = model.eval()
                torch.set_grad_enabled(False)
            else:
                model = model.train()
                torch.set_grad_enabled(True)

            for x, x_classes, y in tqdm(iter(dataloader)):
                y_prim, y_prims = model.forward(x.to(device), x_classes.to(device))
                loss = loss_fn.forward(y_prims, y.to(device))

                if dataloader == dataloader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_training_losses.append(loss.item())
                elif dataloader == dataloader_valid:
                    epoch_validation_losses.append(loss.item())
        # TRACKING LOSSES
        valid_loss = np.mean(epoch_validation_losses)
        epoch_validation_losses.clear()
        train_loss = np.mean(epoch_training_losses)
        epoch_training_losses.clear()
        training_plot.append((train_loss, valid_loss))

        # CHECKING/SAVING BEST MODEL
        if valid_loss < lowest_validation_loss:
            lowest_validation_loss = valid_loss
            print(f"New best model {model_name}- loss:{valid_loss} epoch:{epoch}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "loss": valid_loss,
                },
                best_model_path,
            )

    torch.save(
        {"model": model.state_dict(), "epoch": epoch, "loss": valid_loss},
        last_model_path,
    )
    df = pd.DataFrame(training_plot, columns=["training_loss", "validation_loss"])
    df.to_csv(
        "./data/experiments_better/training_plot_" + model_name + ".csv",
    )


def training_MOD(
    model: ensemble.EmbAndEnsemble,
    dataloader_train,
    dataloader_valid,
    model_name,
    device,
    learning_rate,
    max_epochs,
    diversity_weight,
    dataset_full,
):
    model.ensemble.seperate_batches = False
    loss_fn = loss_sep_sha.LossSeperate().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), learning_rate)

    lowest_validation_loss = 1e16  # for fininding best model
    best_model_path = (
        "./models/MOD/best_" + model_name + ".pth"
    )  # for best model throughout learning, by validation loss
    last_model_path = "./models/MOD/last_" + model_name + ".pth"
    # TRAINING
    training_plot = []
    epoch = 0
    for epoch in range(max_epochs):
        epoch_validation_losses = []
        epoch_training_losses = []

        for i, dataloader in enumerate(
            [dataloader_train, dataloader_valid]
        ):  # iterates through one training cycle for each member and one validation cycle
            model.ensemble.current_model_train = i  # specifies that only the ith member is to be trained (if validation, this doesn't matter anyway)

            if dataloader == dataloader_valid:
                model = model.eval()
                torch.set_grad_enabled(False)
            else:
                model = model.train()
                torch.set_grad_enabled(True)

            for x, x_classes, y in tqdm(iter(dataloader)):
                y_prim, y_prims = model.forward(x.to(device), x_classes.to(device))
                target_loss = loss_fn.forward(y_prims, y.to(device))

                ood_x = noiser.uniform_feature_noise(x.shape, stretch=1e1).to(
                    torch.float32
                )
                random_classes = noiser.randomise_labels(
                    x_classes, 1, labels=dataset_full.labels[:-1]
                )
                ood_y_mean, ood_ys = model.forward(
                    ood_x.to(device), random_classes.to(device)
                )
                diversity_loss = torch.mean(torch.std(ood_ys, dim=0))

                loss = target_loss - diversity_weight * diversity_loss

                if dataloader == dataloader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_training_losses.append(loss.item())
                elif dataloader == dataloader_valid:
                    epoch_validation_losses.append(loss.item())
        # TRACKING LOSSES
        valid_loss = np.mean(epoch_validation_losses)
        epoch_validation_losses.clear()
        train_loss = np.mean(epoch_training_losses)
        epoch_training_losses.clear()
        training_plot.append((train_loss, valid_loss))

        # CHECKING/SAVING BEST MODEL
        if valid_loss < lowest_validation_loss:
            lowest_validation_loss = valid_loss
            print(f"New best model {model_name}- loss:{valid_loss} epoch:{epoch}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "loss": valid_loss,
                },
                best_model_path,
            )
    torch.save(
        {"model": model.state_dict(), "epoch": epoch, "loss": valid_loss},
        last_model_path,
    )
    df = pd.DataFrame(training_plot, columns=["training_loss", "validation_loss"])
    df.to_csv(
        "./data/experiments_better/training_plot_" + model_name + ".csv",
    )
