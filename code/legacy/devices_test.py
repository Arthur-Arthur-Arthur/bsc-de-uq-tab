import torch.cuda
print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])