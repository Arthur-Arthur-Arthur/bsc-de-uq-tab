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
import uqutils
import training
import random

if __name__ == "__main__":
    # HYPERPARAMETERS
    NUM_TRAININGS = 1 #9
    for run in range(NUM_TRAININGS):
        N_MEMBERS = 3
        MAX_EPOCHS = 0
        LEARNING_RATE = 1e-1
        BATCH_SIZE = 2048
        TRAIN_SPLIT = 0.1*(run+1)
        VALIDATION_SPLIT = 0.1
        EMBEDDING_SIZE = 2
        DEVICE = "cpu"
        if torch.cuda.is_available():
            DEVICE = "cuda"
    
        # DATASETS AND DATALOADERS
        dataset_full = dataset.Dataset("./data/loan_squeak.pkl")
        dataset_full.Y.to(DEVICE)

        dataset_train, dataset_valid, dataset_test = torch.utils.data.random_split(
            dataset_full,
            [TRAIN_SPLIT, VALIDATION_SPLIT, 1 - TRAIN_SPLIT - VALIDATION_SPLIT],
            generator=torch.Generator().manual_seed(0),
        )

        dataloaders_train = [
            torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=BATCH_SIZE,
                num_workers=2,
                drop_last=True,
                sampler=dataset.equal_sampler(dataset_train, dataset_full),
                # shuffle=True
            )
            for n in range(N_MEMBERS)
        ]
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

        # MODEL SPECIFICATION
        n_members = N_MEMBERS
        depths = [16] * n_members  # constant depth experiment
        widths = [64] * n_members
        modes = ["res"] * n_members
        model_name = (
            str(run)
            + "IGNORANT_RUN"
            + str(n_members)
            + "M_"
            + str(depths[0])
            + "X"
            + str(widths[0])
        )
        # MODEL CREATION
        model_ensemble = ensemble.Ensemble(
            input_size=(
                dataset_full.X.shape[1]  # skalārie
                + dataset_full.X_classes.shape[1] * EMBEDDING_SIZE  # embeddings
            ),
            output_size=len(dataset_full.labels[-1]),  # klases
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
        loss_fn=loss_sep_sha.LossNLL()
        training.training_base(
            model,
            dataloaders_train=dataloaders_train,
            dataloader_valid=dataloader_valid,
            model_name=model_name,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            max_epochs=MAX_EPOCHS,
        )
        # TESTING
        best_model_path = "./models/best_" + model_name + ".pth"
        best_model = torch.load(best_model_path)
        model.load_state_dict(best_model["model"])
        model.eval()
        x, x_classes, y = next(iter(dataloader_valid))
        y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
        temperature, _, _ = uqutils.perform_tempscale(y_prim, y.to(DEVICE, torch.float))

        data = pd.DataFrame(
            columns=[
                "members",
                "ensemble_loss",
                "tempered_loss",
                "accuracy",
                "confidence",
                "tempered_confidence",
                "std",
                "ece",
                "time",
                "best_epoch",
            ]
        )
        data.loc[0, "best_epoch"] = best_model["epoch"]
        start = time.process_time()
        for x, x_classes, y in tqdm(iter(dataloader_test)):
            y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
        end = time.process_time()
        epoch_time = end - start
        data.loc[0, "time"] = epoch_time
        for members in range(1, N_MEMBERS+1):
            losses = []
            model_losses = []
            temp_losses = []
            accs = []
            confs = []
            temp_confs = []
            stds = []
            y_prims_for_ece = np.zeros((1, len(dataset_full.Y_labels)))
            y_for_ece = np.zeros((1, len(dataset_full.Y_labels)))
            for permute in range(N_MEMBERS-members+1):
                options=list(range(N_MEMBERS))
                random.shuffle(options)
                model_ensemble.feature_mask = torch.tensor(
                np.ones((len(model_ensemble.members), model_ensemble.input_size))
            ).to(torch.float32).to('cuda')
                for i in options[:members]: model_ensemble.feature_mask[i,:]=0
                for x, x_classes, y in tqdm(iter(dataloader_test)):

                    y_prim, y_prims = model.forward(
                        x.to(DEVICE), x_classes.to(DEVICE)
                    )
                    loss = loss_fn.forward(y_prim, y.to(DEVICE))
                    y = y.cpu()
                    y_prims = y_prims.cpu()
                    y_prim = y_prim.cpu()
                    losses.append(loss.item())
                    model_losses.append(
                        [
                            metrics.nll(y_prim.data.numpy(), y.data.numpy())
                            for y_prim in y_prims
                        ]
                    )
                    temp_scaled_y_prim = uqutils.prob_power_t(y_prim, temperature)
                    temp_loss = loss_fn.forward(temp_scaled_y_prim, y)
                    temp_losses.append(temp_loss.item())
                    y_idx = np.argmax(y.data.numpy(), axis=1).astype(int)
                    y_prim_idx = np.argmax(
                        torch.mean(y_prims, 0).data.numpy(), axis=1
                    ).astype(int)
                    acc = np.mean((y_idx == y_prim_idx) * 1.0)
                    accs.append(acc)
                    conf = np.mean(np.max(torch.mean(y_prims, 0).data.numpy(), axis=1))
                    confs.append(conf)
                    temp_conf = np.mean(np.max(temp_scaled_y_prim.data.numpy(), axis=1))
                    temp_confs.append(temp_conf)
                    y_for_ece = np.append(y_for_ece, y.data.numpy(), 0)
                    y_prims_for_ece = np.append(
                        y_prims_for_ece, torch.mean(y_prims, 0).data.numpy(), 0
                    )
                    stds.append(np.mean(np.std(y_prims.data.numpy(), 0)))
            data.loc[members, "members"] = members
            data.loc[members, "ensemble_loss"] = np.mean(losses)
            #data.loc[members, "individual_losses"] = [np.mean(model_losses, axis=0)]
            data.loc[members, "tempered_loss"] = np.mean(temp_losses)
            data.loc[members, "accuracy"] = np.mean(accs)
            data.loc[members, "confidence"] = np.mean(confs)
            data.loc[members, "tempered_confidence"] = np.mean(temp_confs)
            data.loc[members, "std"] = np.mean(stds)
            data.loc[members, "ece"] = metrics.calc_ece(
                y_for_ece, y_prims_for_ece, 10
            )
        data.to_csv(
            "./data/experiments_better/memb" + model_name + "_results.csv", index=False
        )
