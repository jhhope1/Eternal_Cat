import torch

a = torch.tensor([[0,1,0,2],[1,2,3,4],[2,14,100,0]])
print(a.shape)
b = a[torch.tensor([2,0])].float()
print(b)

c = torch.tensor([1,2,7])
d = torch.zeros(10).scatter_(0, c, 1)
print(d)