import torch
weights = []


a = torch.tensor([2,3,4])
b = torch.tensor([2,3])
c = torch.tensor([2,3])
weights.extend(list(a))
weights.extend(list(b))
weights.extend(list(c))
print(weights)
print(torch.tensor(weights))