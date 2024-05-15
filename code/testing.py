import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm
import pandas as pd
import pathlib

import ensemble
import dataset
import noiser
import metrics
import time
import loss_sep_sha
import uqutils
import training

DEVICE = "cuda"
loss_nll = loss_sep_sha.LossNLL()
loss_sep = loss_sep_sha.LossSeperate()


def testing_base(model, model_name, dataloader_valid, dataloader_test, dataset_full):

    # TESTING
    best_model_path = "./models/best_" + model_name + ".pth"
    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model["model"])
    model.eval()
    x, x_classes, y = next(iter(dataloader_valid))
    y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
    temperature = temp_from_valid(best_model_path, model, dataloader_valid)
    testing_shorter(
        best_model_path, model, dataloader_test, temperature, dataset_full.labels[:-1]
    )


def temp_from_valid(model_path, model, dataloader_valid):

    best_model = torch.load(model_path)
    model.load_state_dict(best_model["model"])
    model.eval()
    x, x_classes, big_y = next(iter(dataloader_valid))
    big_y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
    big_y = big_y.detach().cpu()
    big_y_prim = big_y_prim.detach().cpu()
    for x, x_classes, y in dataloader_valid:
        y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
        big_y = torch.cat((big_y, y.detach().cpu()))
        big_y_prim = torch.cat((big_y_prim, y_prim.detach().cpu()))

    temperature, _, _ = uqutils.perform_tempscale(big_y_prim, big_y.to(torch.float))
    return temperature


def testing_shorter(
    model_path, model, dataloader_test, temperature, input_labels, save_name=None
):

    model_name = pathlib.PurePath(model_path).name
    if save_name == None:
        save_name = model_name
    best_model = torch.load(model_path)
    model.load_state_dict(best_model["model"])
    model.eval()

    data = pd.DataFrame()
    data.at[0, "best_epoch"] = best_model["epoch"]
    start = time.process_time()
    for x, x_classes, y in tqdm(iter(dataloader_test)):
        y_prim, y_prims = model.forward(x.to(DEVICE), x_classes.to(DEVICE))
    end = time.process_time()
    epoch_time = end - start
    data.at[0, "time"] = epoch_time

    for corruption_type in ["noise", "distance"]:

        for corruption in range(0, 10):
            ens_metrics = {
                "loss": [],
                "accuracy": [],
                "confidence": [],
                "ECE": [],
            }
            memb_metrics = {
                "loss": [],
                "accuracy": [],
                "confidence": [],
                "ECE": [],
            }
            temperature_metrics = {
                "loss": [],
                "accuracy": [],
                "confidence": [],
                "ECE": [],
            }
            for x, x_classes, y in tqdm(iter(dataloader_test)):
                # NOISED
                x_corrputed_classes = noiser.randomise_labels(
                    x_classes, corruption / 10, input_labels
                )
                x_noised = noiser.random_noise(x, corruption / 10)
                # DISTANCED
                x_distanced = noiser.random_offset(x, corruption / 5)
                # INFERENCE
                if corruption_type == "noise":
                    x_cor = x_noised
                elif corruption_type == "distance":
                    x_cor = x_distanced
                y_prim, y_prims = model.forward(
                    x_cor.to(DEVICE), x_corrputed_classes.to(DEVICE)
                )
                calc_metrics(
                    y_prim, y, ens_metrics
                )  # this is inplace, although it also returns the dict

                for each_y_prim in y_prims:
                    calc_metrics(each_y_prim, y, memb_metrics)

                # memb_metrics:dict = [
                #     calc_metrics(each_y_prim, y, memb_metrics) for each_y_prim in y_prims
                # ][0]
                temp_scaled_y_prim = uqutils.prob_power_t(y_prim, temperature)
                calc_metrics(temp_scaled_y_prim, y, temperature_metrics)

            for key in ens_metrics.keys():
                data.at[corruption, "ensemble_" + key] = np.mean(ens_metrics[key])
            for key in memb_metrics.keys():
                data.at[corruption, "individual_" + key] = np.mean(memb_metrics[key])
            for key in temperature_metrics.keys():
                data.at[corruption, "T_scaled_" + key] = np.mean(
                    temperature_metrics[key]
                )

        data.to_csv(
            "./data/experiments_better/"
            + save_name
            + "_"
            + corruption_type
            + "__results.csv",
            index=False,
        )


def calc_metrics(y_prim, y, metrics_dict: dict):
    y = y.cpu()
    y_prim = y_prim.cpu()
    loss = loss_nll.forward(y_prim, y)
    y_data = y.data.numpy()
    y_prim_data = y_prim.data.numpy()
    metrics_dict["loss"].append(loss.item())
    y_idx = np.argmax(y_data, axis=1).astype(int)
    y_prim_idx = np.argmax(y_prim_data, axis=1).astype(int)
    acc = np.mean((y_idx == y_prim_idx) * 1.0)
    metrics_dict["accuracy"].append(acc)
    conf = np.mean(np.max(y_prim_data, axis=1))
    metrics_dict["confidence"].append(conf)
    metrics_dict["ECE"].append(metrics.calc_ece(y_data, y_prim_data, 10))
    return metrics_dict
