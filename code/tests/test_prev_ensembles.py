import torch
import numpy as np
import torch.utils.data
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

import ensemble
import dataset
import noiser
import metrics
import time
import loss_sep_sha
import uqutils
import training
import testing

if __name__ == "__main__":
    # HYPERPARAMETERS
    NUM_TRAININGS = 10
    N_MEMBERS = 10
    MAX_EPOCHS = 0
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 1024
    TRAIN_SPLIT = 0.7
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

    # MODEL SPECIFICATION
    n_members = N_MEMBERS
    depths = [16] * n_members  # constant depth experiment
    widths = [64] * n_members
    modes = ["res"] * n_members

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
    for model_path in os.scandir("models\models_10M_16X64"):
        model_path=model_path.path
        temperature = testing.temp_from_valid(model_path, model, dataloader_valid)
        testing.testing_shorter(
            model=model,
            model_path=model_path,
            temperature=temperature,
            dataloader_test=dataloader_test,
            input_labels=dataset_full.labels[:-1],
            
        )
