import torch
import torch.nn as nn
import torchvision.models as models

# setup
device = 'cuda:0'
model = models.resnet18().to(device)
data = torch.randn(64, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (64,), device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_iters = 20
warmup_iters = 10
for i in range(nb_iters):
    optimizer.zero_grad()

    # start profiling after 10 warmup iterations
    if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))

    # push range for forward
    if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
    output = model(data)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    loss = criterion(output, target)

    if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
    optimizer.step()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    # pop iteration range
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()
