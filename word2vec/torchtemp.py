import torch
import torch.nn.functional as F

a = torch.tensor([[0,1,0,2],[1,2,3,4],[2,14,100,0]]).float()
print(a.shape)
b = F.normalize(a, dim = 1, p = 2)
print(b)
