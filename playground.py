import torch
import torch.nn.functional as F


x = torch.randn(80, 128, 7, 7) 
c = torch.ones((2000, 128, 1, 1))

filter = F.conv2d(input=x, weight=c)
print(filter.shape)
