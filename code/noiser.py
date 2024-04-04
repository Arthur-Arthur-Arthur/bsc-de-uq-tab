import torch
import dataset

def random_offset(x:torch.Tensor,distance): #offsets 
    x+=torch.randint(-1,2,x.shape)*distance
    return x

def random_noise(x:torch.Tensor,noise_weight): #offsets 
    x*=(1-noise_weight)
    x+=torch.randn(x.shape)*noise_weight
    return x

def randomise_labels(x_classes:torch.Tensor,randomised_perecentage,labels): #offsets 
    replace=torch.rand(x_classes.shape)>randomised_perecentage
    dont_replace= replace==False
    list_len = [len(i) for i in labels]
    random_labels=torch.randint(high=max(list_len),size=x_classes.shape)
    random_labels=[random_labels[:,i]%length for i,length in enumerate(list_len)]
    random_labels=torch.stack(random_labels,1)
    rand_x_classes=replace.long()*random_labels+dont_replace.long()*x_classes
    return rand_x_classes

