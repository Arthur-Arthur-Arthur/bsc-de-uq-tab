import torch

    
class LossSeperateNoLog(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prims, y):
        y = y.unsqueeze(dim=0)
        y = y.expand(y_prims.size())
        loss = -torch.mean(y * y_prims)
        return loss
    
class LossSeperate(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prims, y):
        y = y.unsqueeze(dim=0)
        y = y.expand(y_prims.size())
        losses = torch.mean(-(y * torch.log(y_prims + 1e-8)), 0)
        loss = torch.mean(losses)
        return loss


class LossShared(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prims: torch.Tensor, y):
        y_prim = y_prims.mean(dim=0)
        loss = -torch.mean(y * torch.log(y_prim + 1e-8))
        return loss

class LossRepulse(torch.nn.Module):
    def __init__(self,repulsion=1):
        super().__init__()
        self.repulsion=repulsion

    def forward(self, y_prims: torch.Tensor, y):
        y_prim = y_prims.mean(dim=0,keepdim=True)
        y = y.unsqueeze(dim=0)
        y = y.expand(y_prims.size())
        target_losses = torch.mean(-(y * torch.log(y_prims + 1e-8)), 0)
        repel_losses= torch.mean((y_prim* torch.log(y_prims + 1e-8)), 0)
        losses=target_losses+repel_losses*self.repulsion
        loss = torch.mean(losses)
        return loss