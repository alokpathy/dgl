# test.py
import torch

x = torch.randn(1, 2, 3, 4,device='cuda')
y = torch.randn(1, 2, 3, 4, device='cuda')

torch.cuda.synchronize()
torch.cuda.nvtx.range_push('calc')
z = x + y
torch.cuda.nvtx.range_pop()
print(z)
