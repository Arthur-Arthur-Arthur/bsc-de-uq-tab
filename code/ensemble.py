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
        x_enc = [embedding(input) for embedding, input in zip(self.embeddings, torch.rot90(x_classes,k=-1))]
        x_enc.append(x)
        x_cat = torch.cat((x_enc), dim=1)
        outs = self.ensemble.forward(x_cat)
        return outs


class Ensemble(torch.nn.Module):
    def __init__(self, input_size, output_size, n_members, depths, widths,modes, device):
        super().__init__()
        self.members = torch.nn.ModuleList(
            [
                Member(input_size, output_size, depths[n], widths[n],modes[n], device)
                for n in range(n_members)
            ]
        )

    def forward(self, x):
        outs = torch.stack([member.forward(x) for member in self.members])
        return outs


class Member(torch.nn.Module):
    def __init__(self, input_size, output_size, depth, width, mode, device):
        super().__init__()
        self.input_layer = Layer(input_size, width)
        if mode=="res":
            self.mode=1
        elif mode=="dense":
            self.mode=2
        else:
            self.mode=0
        if self.mode==0:
            self.hidden_layers = torch.nn.Sequential(
                *[Layer(width, width) for n in range(depth)]
            )
        elif self.mode==1:
            self.hidden_layers = torch.nn.ModuleList(
                [Layer(2*width, width) for n in range(depth)]
            )
        self.output_layer = torch.nn.Sequential(Layer(width, output_size),torch.nn.Softmax(-1))

    def forward(self, x):
        out = self.input_layer.forward(x)
        if self.mode==0:
            out = self.hidden_layers.forward(out)
        elif self.mode==1:
            out1=out
            for layer in self.hidden_layers:
                out = layer.forward(torch.cat((out,out1),dim=-1))
        out = self.output_layer.forward(out)
        return out


class Layer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size,dtype=torch.float32),
            torch.nn.BatchNorm1d(output_size,dtype=torch.float32),
            torch.nn.ReLU(),                
        )

    def forward(self, x):
        out = self.layer.forward(x)
        return out
