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
    data_combo = []
    for lr in range(1, 3):
        for bs in range(4, 5):
            LEARNING_RATE = 10**-lr
            BATCH_SIZE = 4**bs
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
            n_members = 4
            widths = [32, 32, 64, 64]  # constant depth experiment
            depths = [8,  16, 8,  16]
            modes = ["res"] * n_members
            model_name = "search" + str(n_members) + "M_WIDE_2"
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
                seperate_batches=False,
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

            model_path = "./models/" + model_name + ".pth"
            # TRAINING
            for epoch in range(20):
                epoch_validation_losses = []
                for dataloader in [dataloader_train, dataloader_valid]:

                    if dataloader == dataloader_valid:
                        model = model.eval()
                        torch.set_grad_enabled(False)
                    else:
                        model = model.train()
                        torch.set_grad_enabled(True)
                    for x, x_classes, y in tqdm(iter(dataloader)):

                        y_prim, y_prims = model.forward(
                            x.to(DEVICE), x_classes.to(DEVICE)
                        )

                        loss = loss_fn.forward(y_prims, y.to(DEVICE))
                        if dataloader == dataloader_train:
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        elif dataloader == dataloader_valid:
                            epoch_validation_losses.append(loss.item())
                    print(loss)
                    if dataloader == dataloader_valid:
                        epoch_loss = np.mean(epoch_validation_losses)
                        epoch_validation_losses.clear()
                        if epoch_loss < lowest_validation_loss:
                            lowest_validation_loss = epoch_loss
                            print(
                                f"New best model {model_name}- loss:{epoch_loss} epoch:{epoch}"
                            )
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "epoch": epoch,
                                    "loss": epoch_loss,
                                },
                                model_path,
                            )  # Update path to be unique for each model type

            # TESTING

            best_model = torch.load(model_path)
            model.load_state_dict(best_model["model"])
            model.eval()
            data = pd.DataFrame(
                columns=[
                    "member",
                    "depth",
                    "width",
                    "loss",
                    "accuracy",
                    "confidence",
                    "ece",
                ]
            )
            
            losses = []
            model_losses = []
            accs = []
            confs = []
            stds = []
            y_prims_for_ece = np.zeros(
                (len(model_ensemble.members), 1, len(dataset_full.Y_labels))
            )
            y_for_ece = np.zeros((1, len(dataset_full.Y_labels)))
            for x, x_classes, y in tqdm(iter(dataloader_test)):

                y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
                y = y.cpu()
                y_prims = y_prims.cpu()

                losses.append(metrics.nll(y_prims.detach().numpy(), y.detach().numpy()))
                model_losses.append(
                    [
                        metrics.nll(y_prim_member.data.numpy(), y.data.numpy())
                        for y_prim_member in y_prims
                    ]
                )
                y_idx = np.argmax(y.data.numpy(), axis=-1).astype(int)
                y_prims_idx = np.argmax(y_prims.detach().numpy(), axis=-1).astype(int)
                acc = np.mean((y_idx == y_prims_idx) * 1.0, axis=-1)
                accs.append(acc)
                conf = np.mean(np.max(y_prims.detach().numpy(), axis=-1), axis=-1)
                confs.append(conf)
                y_for_ece = np.append(y_for_ece, y.data.numpy(), 0)
                y_prims_for_ece = np.append(
                    y_prims_for_ece, y_prims.detach().numpy(), 1
                )

            for i, member in enumerate(model_ensemble.members):
                data.loc[i, "member"] = i
                data.loc[i, "width"] = member.width
                data.loc[i, "depth"] = member.depth
                data.loc[i, "loss"] = np.mean(losses, axis=0)[i]
                data.loc[i, "accuracy"] = np.mean(accs, axis=0)[i]
                data.loc[i, "confidence"] = np.mean(confs, axis=0)[i]
                data.loc[i, "ece"] = metrics.calc_ece(
                    y_for_ece, y_prims_for_ece[i, :, :], 10
                )
            data["learning_rate"] = LEARNING_RATE
            data["batch_size"] = BATCH_SIZE
            data.to_csv(
                "./data/experiments_better/"
                + model_name
                + "Lr"
                + str(lr)
                + "Bs"
                + str(bs)
                + "_grid_results.csv",
                index=False,
            )
            data_combo.append(data)
    full_data = pd.concat(data_combo)
    full_data.to_csv(
        "./data/experiments_better/" + "combo_grid_results.csv",
        index=False,
    )
