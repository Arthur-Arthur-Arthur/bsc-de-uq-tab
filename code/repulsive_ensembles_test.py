import ensemble
import torch
import dataset
import numpy as np
import torch.utils.data
import noiser
import metrics
import time
import repulsive.f_SVGD as f_SVGD
import repulsive.distributions as distributions
import repulsive.SSGE as SSGE
import repulsive.kernel as kernel
import loss_sep_sha
from tqdm import tqdm
import pandas as pd

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
        # shuffle=True
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
    for log_repulsion in range(-2,2):
        repulsion=10**log_repulsion
        log_count=4
        n_members=2**log_count
        depths=[8]*n_members #constant depth experiment
        widths=[int(1024/n_members)]*n_members
        modes=["res"]*n_members
        model_name=str(repulsion)+"shared"+str(n_members)+"M_"+str(depths[0])+"X"+str(widths[0])
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



        loss_fn = loss_sep_sha.LossRepulse(repulsion=repulsion).to(DEVICE)
        optimizer = torch.optim.RAdam(model.parameters(), LEARNING_RATE)
        lowest_validation_loss = 1e16
        
        model_path = "./models/"+model_name+".pth"
        # TRAINING
        for epoch in range(1, 10):
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
                        print(f"New best model {model_name}- loss:{epoch_loss} epoch:{epoch}")
                        torch.save(
                            {"model": model.state_dict(), "epoch": epoch, "loss": epoch_loss},
                            model_path,
                        )  # Update path to be unique for each model type

        # TESTING

        best_model = torch.load(model_path)
        model.load_state_dict(best_model["model"])
        model.eval()
        data = pd.DataFrame(
            columns=["corruption", "ensemble_loss", "individual_losses", "accuracy","confidence", "std", "ece","time"]
        )
        start=time.process_time()
        for x, x_classes, y in tqdm(iter(dataloader_test)):
                y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
        end =time.process_time()
        epoch_time=end-start
        data.loc[0,"time"] = epoch_time
        for corruption in range(0, 10):
            losses = []
            model_losses = []
            accs = []
            confs=[]
            stds = []
            y_prims_for_ece = np.zeros((1, len(dataset_full.Y_labels)))
            y_for_ece = np.zeros((1, len(dataset_full.Y_labels)))
            for x, x_classes, y in tqdm(iter(dataloader_test)):
                # NOISED
                x_corrputed_classes = noiser.randomise_labels(
                    x_classes, corruption / 10, dataset_full.labels[:-1]
                )
                # x_noised=noiser.random_noise(x,corruption/10)
                # DISTANCED
                x_distanced = noiser.random_offset(x, corruption / 2)
                # INFERENCE
                y_prims = model.forward(x_distanced.to(DEVICE), x_corrputed_classes.to(DEVICE))
                loss = loss_fn.forward(y_prims, y.to(DEVICE))
                y = y.cpu()
                y_prims = y_prims.cpu()
                losses.append(loss.item())
                model_losses.append(
                    [metrics.nll(y_prim.data.numpy(), y.data.numpy()) for y_prim in y_prims]
                )
                y_idx = np.argmax(y.data.numpy(), axis=1).astype(int)
                y_prim_idx = np.argmax(torch.mean(y_prims, 0).data.numpy(), axis=1).astype(int)
                acc = np.mean((y_idx == y_prim_idx) * 1.0)
                accs.append(acc)
                conf=np.mean(np.max(torch.mean(y_prims, 0).data.numpy(),axis=1))
                confs.append(conf)
                y_for_ece = np.append(y_for_ece, y.data.numpy(), 0)
                y_prims_for_ece = np.append(
                    y_prims_for_ece, torch.mean(y_prims, 0).data.numpy(), 0
                )
                stds.append(np.mean(np.std(y_prims.data.numpy(), 0)))
            data.loc[corruption,"corruption"] = corruption
            data.loc[corruption,"ensemble_loss"] = np.mean(losses)
            data.loc[corruption,"individual_losses"] = [np.mean(model_losses, axis=0)]
            data.loc[corruption,"accuracy"] = np.mean(accs)
            data.loc[corruption,"confidence"] = np.mean(confs)
            data.loc[corruption,"std"] = np.mean(stds)
            data.loc[corruption,"ece"] = metrics.calc_ece(y_for_ece, y_prims_for_ece, 10)
        data.to_csv("./data/"+model_name+"_results.csv",index=False)
