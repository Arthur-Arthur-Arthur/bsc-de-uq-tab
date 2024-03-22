import torch
import dataset

def random_offset(x:torch.Tensor,distance): #offsets 
    x+=torch.randint(-1,2,x.shape)*distance
    return x

def random_noise(x:torch.Tensor,noise_weight): #offsets 
    x*=(1-noise_weight)
    x+=torch.randn(x.shape)*noise_weight
    return x

def randomise_labels(x:torch.Tensor,noise_weight): #offsets 
    x*=(1-noise_weight)
    x+=torch.randn(x.shape)*noise_weight
    return x