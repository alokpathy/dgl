import torch
import time

a = torch.rand(10000, 10000).cuda()
b = torch.rand(10000, 10000).cuda()

start_time = time.time()
c = torch.mm(a, b)
torch.cuda.synchronize()
print(f"time: {time.time() - start_time}")
