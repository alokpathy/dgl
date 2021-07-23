import argparse
import torch

# Microbenchmark for tensor cores on T4

parser = argparse.ArgumentParser(description='bmm microbenchmark')

parser.add_argument("--timing", default=False, action='store_true',
        help="enable cudaEvent timers")

args = parser.parse_args()
timing = args.timing

def start_time(timer):
    if timing:
        timer.record()

def stop_time(start_timer, stop_timer):
    if timing:
        stop_timer.record()
        torch.cuda.synchronize()
        return start_timer.elapsed_time(stop_timer)
    else:
        return 0.0

dims = [100, 200, 300, 400, 500, 600, 800]
# dims = [2 ** i for i in range(6, 15)]

mm_times = []
bmm_times = []

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

for dim in dims:
    torch.cuda.nvtx.range_push("nvtx-seed-dim{}".format(dim))
    torch.manual_seed(0)
    torch.cuda.nvtx.range_pop()

    print(f"dim: {dim}")
    torch.cuda.nvtx.range_push("nvtx-instantiation-dim{}".format(dim))
    a = torch.rand(dim, 1, dim).cuda()
    b = torch.rand(dim, dim, dim).cuda()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-emptycache-dim{}".format(dim))
    torch.cuda.empty_cache()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-warmup-mm-dim{}".format(dim))
    c = []
    for i in range(dim):
        c.append(torch.matmul(a[i], b[i])) # warmup iteration
    c = torch.cat(c)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-mm-dim{}".format(dim))
    start_time(start)
    c = []
    for i in range(dim):
        c.append(torch.matmul(a[i], b[i])) # warmup iteration
    c = torch.cat(c)
    torch.cuda.nvtx.range_pop()
    mm_times.append(stop_time(start, stop))

    torch.cuda.nvtx.range_push("nvtx-emptycache2-dim{}".format(dim))
    torch.cuda.empty_cache()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-warmup-tcmm-dim{}".format(dim))
    c = torch.bmm(a, b) # warmup iteration
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("nvtx-tcmm-dim{}".format(dim))
    start_time(start)
    c = torch.bmm(a, b)
    torch.cuda.nvtx.range_pop()
    bmm_times.append(stop_time(start, stop))

    torch.cuda.nvtx.range_push("nvtx-dealloc-dim{}".format(dim))
    del a
    del b
    del c
    torch.cuda.nvtx.range_pop()

speedups = []

if timing:
    for i in range(len(dims)):
        speedups.append(mm_times[i] / bmm_times[i])

if timing:
    print("dim mm_time bmm_time speedup")
    for i in range(len(dims)):
        print(dims[i], end=" ")
        print(mm_times[i], end=" ")
        print(bmm_times[i], end=" ")
        print(speedups[i], end=" ")
        print()
