import torch
import numpy


class EmbAndEnsemble(torch.nn.Module):
    def __init__(self, dataset_labels, embedding_size, ensemble):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    num_embeddings=len(label), embedding_dim=embedding_size
                )
                for label in dataset_labels[:-1]
            ]
        )
        self.ensemble = ensemble

    def forward(self, x, x_classes):
        x_enc = []
        for i, emb in enumerate(self.embeddings):
            x_enc.append(emb.forward(x_classes[:, i]))
        x_enc.append(x)
        x_cat = torch.cat((x_enc), dim=1)
        outs = self.ensemble.forward(x_cat)
        return outs


class Ensemble(torch.nn.Module):
    def __init__(self, input_size, output_size, n_members, depths, widths, device):
        super().__init__()
        self.members = torch.nn.ModuleList(
            [
                Member(input_size, output_size, depths[n], widths[n], device)
                for n in range(n_members)
            ]
        )

    def forward(self, x):
        outs = torch.stack([member.forward(x) for member in self.members])
        return outs


class Member(torch.nn.Module):
    def __init__(self, input_size, output_size, depth, width, device):
        super().__init__()
        self.input_layer = Layer(input_size, width)
        self.hidden_layers = torch.nn.Sequential(
            *[Layer(width, width) for n in range(depth)]
        )
        self.output_layer = torch.nn.Sequential(Layer(width, output_size),torch.nn.Softmax(1))

    def forward(self, x):
        out = self.input_layer.forward(x)
        out = self.hidden_layers.forward(out)
        out = self.output_layer.forward(out)
        return out


class Layer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.BatchNorm1d(output_size),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer.forward(x)
        return out
