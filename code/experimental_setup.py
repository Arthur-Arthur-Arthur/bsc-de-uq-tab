import ensemble
import torch
import dataset
import numpy as np
import torch.utils.data

LEARNING_RATE = 1e-4
BATCH_SIZE = 1024
TRAIN_TEST_SPLIT = 0.8
EMBEDDING_SIZE = 2
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


dataset_full = dataset.Dataset("./data/loan_squeak.pkl")
dataset_full.Y.to(DEVICE)


train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0),
)
y_train_indices = dataset_train.indices

y_train = [dataset_full.Y_idx[i] for i in y_train_indices]

class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
)

weight = 1.0 / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)

sampler = torch.utils.data.WeightedRandomSampler(
    samples_weight.type("torch.FloatTensor"), len(samples_weight)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1),
    sampler=sampler,
    # shuffle=True
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
    depths=[8, 16],
    widths=[1024, 256],
    modes=["res", "res"],
    n_members=2,
    device=DEVICE,
)
model = ensemble.EmbAndEnsemble(
    dataset_labels=dataset_full.labels,
    embedding_size=EMBEDDING_SIZE,
    ensemble=model_ensemble,
)
optimizer = torch.optim.RAdam(model.parameters(), LEARNING_RATE)
