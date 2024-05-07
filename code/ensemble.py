import torch
import numpy as np


class EmbAndEnsemble(torch.nn.Module):
    def __init__(self, dataset_labels, embedding_size, ensemble):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    num_embeddings=len(label), embedding_dim=embedding_size,dtype=torch.float32
                )
                for label in dataset_labels[:-1]
            ]
        )
        self.ensemble = ensemble

    def forward(self, x, x_classes):
        x_enc = [
            embedding(input)
            for embedding, input in zip(self.embeddings, torch.rot90(x_classes, k=-1))
        ]
        x_enc.append(x)
        x_cat = torch.cat((x_enc), dim=1)
        outs = self.ensemble.forward(x_cat)
        return outs


class Ensemble_Base(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        ready_members: torch.nn.ModuleList,
        feature_mask: np.ndarray = None,
        seperate_batches=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.members = ready_members
        if feature_mask is not None:
            self.feature_mask = feature_mask
        else:
            self.feature_mask = torch.tensor(
                np.ones((len(self.members), self.input_size))
            ).to(torch.float32).to('cuda')
        self.seperate_batches = seperate_batches
        self.current_model_train = 0

    def forward(self, x:torch.Tensor):
        if self.training and self.seperate_batches:
            x_chunk=x.chunk(len(self.members),0)
            outs = torch.stack(
                [
                    member.forward(x_chunk[i] * self.feature_mask[i, :])
                    for i, member in enumerate(self.members)
                ]
            )
            # out = self.members[self.current_model_train].forward(
            #     x * self.feature_mask[self.current_model_train, :]
            # )
            # self.current_model_train += 1
            # self.current_model_train %= len(self.members)
            # outs=out.unsqueeze(0)
            # outs = outs.expand((len(self.members),-1,-1))
            #previous code that passed batches to 1 member at a time instead of splitting. splitting is faster
            out=torch.reshape(outs,(-1,)+outs.shape[2:])
        else:
            outs = torch.stack(
                [
                    member.forward(x * self.feature_mask[i, :])
                    for i, member in enumerate(self.members)
                ]
            )
            out=outs.mean(0)
        return out,outs


class Ensemble(Ensemble_Base):
    def __init__(
        self,
        input_size,
        output_size,
        n_members,
        depths,
        widths,
        modes,
        device,
        feature_mask: np.ndarray = None,
        seperate_batches=True,
    ):

        ready_members = torch.nn.ModuleList(
            [
                Member(input_size, output_size, depths[n], widths[n], modes[n], device)
                for n in range(n_members)
            ]
        )
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            ready_members=ready_members,
            feature_mask=feature_mask,
            seperate_batches=seperate_batches
        )


def make_feature_mask(self, overlap=1.0):
    n_members = len(self.members)
    feature_mask = np.zeros((n_members, self.input_size))
    rng = np.random.default_rng()
    for feature in range(self.input_size):
        num_overlap = int((n_members - 1) * overlap + 1 + rng.random())
        use_feature = rng.choice(n_members, num_overlap, replace=False)
        feature_mask[use_feature, feature] = 1
    feature_mask = torch.tensor(feature_mask).to(dtype=torch.float32)
    return feature_mask


class Member(torch.nn.Module):
    def __init__(self, input_size, output_size, depth, width, mode, device):
        super().__init__()
        self.depth=depth
        self.width=width
        self.input_layer = Layer(input_size, width)
        if mode == "res":
            self.mode = 1
        elif mode == "dense":
            self.mode = 2
        else:
            self.mode = 0
        if self.mode == 0:
            self.hidden_layers = torch.nn.Sequential(
                *[Layer(width, width) for n in range(depth)]
            )
        elif self.mode == 1:
            self.hidden_layers = torch.nn.ModuleList(
                [Layer(2 * width, width) for n in range(depth)]
            )
        self.output_layer = torch.nn.Sequential(
            Layer(width, output_size), torch.nn.Softmax(-1)
        )

    def forward(self, x):
        out = self.input_layer.forward(x)
        if self.mode == 0:
            out = self.hidden_layers.forward(out)
        elif self.mode == 1:
            out1 = out
            for layer in self.hidden_layers:
                out = layer.forward(torch.cat((out, out1), dim=-1))
        out = self.output_layer.forward(out)
        return out


class Layer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size, dtype=torch.float32),
            torch.nn.BatchNorm1d(output_size, dtype=torch.float32),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer.forward(x)
        return out


class SquarePlus(torch.nn.Module):
    def forward(self, x):
        return torch.square(x + 1.0)


class Abs(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)
