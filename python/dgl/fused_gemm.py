import torch
from . import backend as F
from . import ndarray as nd
from ._ffi.function import _init_api

def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL['int32'] if x is None else F.zerocopy_to_dgl_ndarray(x)

def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return nd.NULL['int32'] if x is None else F.zerocopy_to_dgl_ndarray_for_write(x)

timing = False

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

preproc_time = 0.0
first_step = True
padding_time = 0.0
atohalf_time = 0.0
btohalf_time = 0.0
ctohalf_time = 0.0
blockedspmm_time = 0.0
abctofloat_time = 0.0

class FusedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, b1, a2, b2):
        arg_a1 = to_dgl_nd(a1)
        arg_b1 = to_dgl_nd(b1)
        arg_a2 = to_dgl_nd(a2)
        arg_b2 = to_dgl_nd(b2)

        # print(f"a1.size: {a1.size()} b1.size: {b1.size()}, a2.size: {a2.size()}, b2.size: {b2.size()}")
        ctx.save_for_backward(a1, b1, a2, b2)

        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1)
        c1 = F.zeros((a1.size(0), b1.size(1)), dtype, ctx_cuda).fill_(1).cuda()
        c2 = F.zeros((a2.size(0), b2.size(1)), dtype, ctx_cuda).fill_(1).cuda()

        arg_c1 = to_dgl_nd_for_write(c1)
        arg_c2 = to_dgl_nd_for_write(c2)

        _CAPI_DGLKernelFGEMM(arg_a1, arg_b1, arg_c1, \
                                a1.size(0), a1.size(1), b1.size(1), \
                                arg_a2, arg_b2, arg_c2, \
                                a2.size(0), a2.size(1), b2.size(1))
        return c1, c2

    @staticmethod
    def backward(ctx, dC1, dC2):
        a1, b1, a2, b2 = ctx.saved_tensors
        arg_dC1 = to_dgl_nd(dC1)
        arg_dC2 = to_dgl_nd(dC2)

        grad_a1 = grad_b1 = None
        grad_a2 = grad_b2 = None

        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1)

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            b1 = b1.t()
            b2 = b2.t()
            grad_a1 = F.zeros((dC1.size(0), b1.size(1)), dtype, ctx_cuda).cuda()
            grad_a2 = F.zeros((dC2.size(0), b2.size(1)), dtype, ctx_cuda).cuda()

            arg_b1 = to_dgl_nd(b1)
            arg_b2 = to_dgl_nd(b2)

            arg_grad_a1 = to_dgl_nd_for_write(grad_a1)
            arg_grad_a2 = to_dgl_nd_for_write(grad_a2)

            _CAPI_DGLKernelFGEMM(arg_dC1, arg_b1, arg_grad_a1, \
                                    dC1.size(0), dC1.size(1), b1.size(1), \
                                    arg_dC2, arg_b2, arg_grad_a2, \
                                    dC2.size(0), dC2.size(1), b2.size(1))
        elif ctx.needs_input_grad[0]:
            grad_a1 = torch.matmul(dC1, b1.t()).cuda()
        elif ctx.needs_input_grad[2]:
            grad_a2 = torch.matmul(dC2, b2.t()).cuda()

        if ctx.needs_input_grad[1] and ctx.needs_input_grad[3]:
            a1 = a1.t()
            a2 = a2.t()
            grad_b1 = F.zeros((a1.size(0), dC1.size(1)), dtype, ctx_cuda).cuda()
            grad_b2 = F.zeros((a2.size(0), dC2.size(1)), dtype, ctx_cuda).cuda()

            arg_a1 = to_dgl_nd(a1)
            arg_a2 = to_dgl_nd(a2)

            arg_grad_b1 = to_dgl_nd_for_write(grad_b1)
            arg_grad_b2 = to_dgl_nd_for_write(grad_b2)

            _CAPI_DGLKernelFGEMM(arg_a1, arg_dC1, arg_grad_b1, \
                                    a1.size(0), a1.size(1), dC1.size(1), \
                                    arg_a2, arg_dC2, arg_grad_b2, \
                                    a2.size(0), a2.size(1), dC2.size(1))
        elif ctx.needs_input_grad[1]:
            grad_b1 = torch.matmul(a1, dC1).cuda()
        elif ctx.needs_input_grad[3]:
            grad_b2 = torch.matmul(a2, dC2).cuda()

        return grad_a1, grad_b1, grad_a2, grad_b2

class FusedGEMMSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, b1, a2, b2):
        global preproc_time

        preproc_start = torch.cuda.Event(enable_timing=True)
        preproc_stop = torch.cuda.Event(enable_timing=True)

        start_time(preproc_start)
        arg_a1 = to_dgl_nd(a1)
        arg_a2 = to_dgl_nd(a2)

        b = torch.cat((b1, b2), dim=0)
        arg_b = to_dgl_nd(b)

        # print(f"a1.size: {a1.size()} b1.size: {b1.size()}, a2.size: {a2.size()}, b2.size: {b2.size()}")
        ctx.save_for_backward(a1, b1, a2, b2)

        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1)
        c = F.zeros((a1.size(0) + a2.size(0), b1.size(1)), dtype, ctx_cuda).cuda()

        arg_c = to_dgl_nd_for_write(c)
        preproc_time += stop_time(preproc_start, preproc_stop)

        _CAPI_DGLKernelFGEMMSpMM(arg_a1, arg_b, arg_c, \
                                    a1.size(0), a1.size(1), b1.size(1), \
                                    arg_a2, \
                                    a2.size(0), a2.size(1), b2.size(1))
        
        c1 = c[:a1.size(0)]
        c2 = c[a1.size(0):]

        if timing:
            print(f"preproc_time: {preproc_time}")

        return c1, c2

    @staticmethod
    def backward(ctx, dC1, dC2):
        a1, b1, a2, b2 = ctx.saved_tensors
        arg_dC1 = to_dgl_nd(dC1)
        arg_dC2 = to_dgl_nd(dC2)

        grad_a1 = grad_b1 = None
        grad_a2 = grad_b2 = None

        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1)

        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            b1_t = b1.t()
            b2_t = b2.t()
            b = torch.cat((b1_t, b2_t), dim=0)
            grad_a = F.zeros((dC1.size(0) + dC2.size(0), b.size(1)), dtype, ctx_cuda).cuda()

            arg_b = to_dgl_nd(b)

            arg_grad_a = to_dgl_nd_for_write(grad_a)

            # _CAPI_DGLKernelFGEMM(arg_dC1, arg_b1, arg_grad_a1, \
            #                         dC1.size(0), dC1.size(1), b1.size(1), \
            #                         arg_dC2, arg_b2, arg_grad_a2, \
            #                         dC2.size(0), dC2.size(1), b2.size(1))

            _CAPI_DGLKernelFGEMMSpMM(arg_dC1, arg_b, arg_grad_a, \
                                        dC1.size(0), dC1.size(1), b.size(1), \
                                        arg_dC2, \
                                        dC2.size(0), dC2.size(1), b.size(1))
            grad_a1 = grad_a[:a1.size(0)]
            grad_a2 = grad_a[a1.size(0):]
        elif ctx.needs_input_grad[0]:
            grad_a1 = torch.matmul(dC1, b1.t()).cuda()
        elif ctx.needs_input_grad[2]:
            grad_a2 = torch.matmul(dC2, b2.t()).cuda()

        if ctx.needs_input_grad[1] and ctx.needs_input_grad[3]:
            a1_t = a1.t()
            a2_t = a2.t()
            dC = torch.cat((dC1, dC2), dim=0)
            grad_b = F.zeros((a1_t.size(0) + a2_t.size(0), dC.size(1)), dtype, ctx_cuda).cuda()

            arg_a1_t = to_dgl_nd(a1_t)
            arg_a2_t = to_dgl_nd(a2_t)
            arg_dC = to_dgl_nd(dC)

            arg_grad_b = to_dgl_nd_for_write(grad_b)

            # _CAPI_DGLKernelFGEMM(arg_a1, arg_dC1, arg_grad_b1, \
            #                         a1.size(0), a1.size(1), dC1.size(1), \
            #                         arg_a2, arg_dC2, arg_grad_b2, \
            #                         a2.size(0), a2.size(1), dC2.size(1))

            _CAPI_DGLKernelFGEMMSpMM(arg_a1_t, arg_dC, arg_grad_b, \
                                        a1_t.size(0), a1_t.size(1), dC.size(1), \
                                        arg_a2_t, \
                                        a2_t.size(0), a2_t.size(1), dC.size(1))

            grad_b1 = grad_b[:b1.size(0)]
            grad_b2 = grad_b[b1.size(0):]
            
        elif ctx.needs_input_grad[1]:
            grad_b1 = torch.matmul(a1, dC1).cuda()
        elif ctx.needs_input_grad[3]:
            grad_b2 = torch.matmul(a2, dC2).cuda()

        return grad_a1, grad_b1, grad_a2, grad_b2

class FusedGEMMBlockSpMM(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, a1, b1, a2, b2):
    def forward(ctx, a_mats, b_mats):
        global preproc_time
        global padding_time
        global atohalf_time
        global btohalf_time
        global ctohalf_time
        global blockedspmm_time
        global abctofloat_time
        global first_step

        preproc_start = torch.cuda.Event(enable_timing=True)
        preproc_stop = torch.cuda.Event(enable_timing=True)

        a_mats_pad = []
        b_mats_pad = []

        block_dim = max([max(j.size(0), j.size(1)) for j in a_mats])

        torch.cuda.nvtx.range_push("nvtx-padding")
        start_time(preproc_start)

        for i in range(len(a_mats)):
            a_mats_pad.append(torch.nn.functional.pad(a_mats[i], (0, block_dim - a_mats[i].size(1), \
                                                                        0, block_dim - a_mats[i].size(0))))
            b_mats_pad.append(torch.nn.functional.pad(b_mats[i], (0, 0, 0, block_dim - a_mats[i].size(1))))

        padding_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-concat-mats")
        start_time(preproc_start)
        a_pad = torch.cat(a_mats_pad, dim=0).half()
        b_pad = torch.cat(b_mats_pad, dim=0).contiguous().half()
        ctx.save_for_backward(a_pad, b_pad)
        ctx.num_blocks = len(b_mats)
        arg_a_pad = to_dgl_nd(a_pad)
        arg_b_pad = to_dgl_nd(b_pad)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-construct-c")
        start_time(preproc_start)
        ctx_cuda = F.context(b_pad)
        c_pad = F.zeros((a_pad.size(0), b_pad.size(1)), torch.half, ctx_cuda).cuda()

        arg_c_pad = to_dgl_nd_for_write(c_pad)
        ctohalf_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-blocked-spmm")
        start_time(preproc_start)
        _CAPI_DGLKernelFGEMMBlockSpMM(arg_a_pad, arg_b_pad, arg_c_pad, \
                                        a_pad.size(0), a_pad.size(1), b_pad.size(1), block_dim)
        blockedspmm_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("nvtx-c-to-float")
        start_time(preproc_start)
        c_pad = c_pad.float()
        abctofloat_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-extractc1c2")
        c = []
        for i in range(len(a_mats_pad)):
            output_ci = c_pad[(i * block_dim):((i + 1) * block_dim)]
            c.append(output_ci[:a_mats[i].size(0), :b_mats[i].size(1)])
        c = torch.cat(c)
        torch.cuda.nvtx.range_pop()

        if timing and first_step:
            first_step = False
            preproc_time = 0.0
            padding_time = 0.0
            atohalf_time = 0.0
            btohalf_time = 0.0
            ctohalf_time = 0.0
            blockedspmm_time = 0.0
            abctofloat_time = 0.0
        if timing:
            print(f"preproc_time: {preproc_time}")
            print(f"padding_time: {padding_time}")
            print(f"atohalf_time: {atohalf_time}")
            print(f"btohalf_time: {btohalf_time}")
            print(f"ctohalf_time: {ctohalf_time}")
            print(f"blockedspmm_time: {blockedspmm_time}")
            print(f"abctofloat_time: {abctofloat_time}")

        return c

    @staticmethod
    def backward(ctx, dC):
        a_pad, b_pad = ctx.saved_tensors
        num_blocks = ctx.num_blocks

        return [None] * num_blocks, [None] * num_blocks

# class FusedGEMMBatchMM(torch.autograd.Function):
#     @staticmethod
#     # def forward(ctx, a1, b1, a2, b2, a3, b3, a4, b4):
#     def forward(ctx, a_mats, b_mats):
#         global preproc_time
# 
#         preproc_start = torch.cuda.Event(enable_timing=True)
#         preproc_stop = torch.cuda.Event(enable_timing=True)
# 
#         a_mats_pad = []
# 
#         block_dim = max([j.size(0) for j in a_mats])
# 
#         torch.cuda.nvtx.range_push("nvtx-padding")
#         start_time(preproc_start)
#         for i in range(len(a_mats)):
#             a_mats_pad.append(torch.nn.functional.pad(a_mats[i], (0, 0, 0, block_dim - a_mats[i].size(0))))
# 
#         preproc_time += stop_time(preproc_start, preproc_stop)
#         torch.cuda.nvtx.range_pop()
# 
#         torch.cuda.nvtx.range_push("nvtx-construct-ab3d")
#         a3d = torch.stack(a_mats_pad)
#         b3d = torch.stack(b_mats)
#         torch.cuda.nvtx.range_pop()
# 
#         torch.cuda.nvtx.range_push("nvtx-bmm")
#         # with torch.cuda.amp.autocast():
#         c = torch.bmm(a3d, b3d)
#         torch.cuda.nvtx.range_pop()
# 
#         torch.cuda.nvtx.range_push("nvtx-extract-c")
#         c_layers = [j.squeeze(0) for j in torch.tensor_split(c, c.size(0))]
#         for i in range(len(c_layers)):
#             c_layers[i] = c_layers[i][:a_mats[i].size(0), :b_mats[i].size(1)]
#         c = torch.cat(c_layers, dim=0)
#         torch.cuda.nvtx.range_pop()
# 
#         if timing:
#             print(f"preproc_time: {preproc_time}")
# 
#         return c
# 
#     @staticmethod
#     def backward(ctx, dC1, dC2, dC3, dC4):
#         return None, None, None, None, None, None, None, None

class FusedGEMMBatchMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_mats, b_mats, a_mats_rows):

        block_dim = max([j.size(0) for j in a_mats])

        torch.cuda.nvtx.range_push("nvtx-construct-a3d")
        a_mat_cat = torch.cat(a_mats)
        a3d = torch.cuda.FloatTensor(len(a_mats), block_dim, a_mats[0].size(1))

        arg_a3d = to_dgl_nd_for_write(a3d)
        arg_a_mat = F.to_dgl_nd(a_mat_cat)
        arg_a_mats_rows = F.to_dgl_nd(a_mats_rows)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-padding")
        _CAPI_DGLKernelPadA(arg_a3d, arg_a_mat, arg_a_mats_rows, len(a_mats), block_dim, a_mats[0].size(1))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-construct-b3d")
        b3d = b_mats
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-bmm")
        # with torch.cuda.amp.autocast():
        c3d = torch.bmm(a3d, b3d)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-unpadding")
        edge_count = a_mats_rows.sum().item()
        c = torch.cuda.FloatTensor(edge_count, c3d.size(2))
        arg_c3d = F.to_dgl_nd(c3d)
        arg_c = F.to_dgl_nd(c)
        _CAPI_DGLKernelUnpadC(arg_c3d, arg_c, arg_a_mats_rows, c3d.size(0), c3d.size(1), c3d.size(2))
        torch.cuda.nvtx.range_pop()

        return c

    @staticmethod
    def backward(ctx, dC1, dC2, dC3, dC4):
        return None, None, None, None, None, None, None, None

class CAPIGEMMs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_mats, b_mats, a_mats_rows):
        # arg_a_mats = [None] * len(a_mats)
        # arg_b_mats = [None] * len(a_mats)

        # c_mats = [None] * len(a_mats)
        # arg_c_mats = [None] * len(a_mats)

        # a_mats_rows = [0] * len(a_mats)
        torch.cuda.nvtx.range_push("nvtx-copy-rowcounts")
        # a_mats_rows = torch.IntTensor(len(a_mats))

        middim = b_mats[0].size(0)
        outcol = b_mats[0].size(1)
        
        total_edges = torch.sum(a_mats_rows).item()
        # for i in range(len(a_mats)):
        #     a_mats_rows[i] = a_mats[i].size(0)
        #     total_edges += a_mats[i].size(0)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-instantiate-c")
        ctx_cuda = F.context(b_mats[0])
        # c_mats = F.zeros((sum(a_mats_rows), b_mats[i].size(1)), torch.float32, ctx_cuda).cuda()
        c_mats = torch.cuda.FloatTensor(total_edges, b_mats[0].size(1))
        torch.cuda.nvtx.range_pop()

        # torch.cuda.nvtx.range_push("nvtx-instantiate-mats")
        # row_count = 0
        # for i in range(len(a_mats)):
        #     torch.cuda.nvtx.range_push("nvtx-append-argab{}".format(i))
        #     arg_a_mats[i] = F.to_dgl_nd(a_mats[i])
        #     arg_b_mats[i] = F.to_dgl_nd(b_mats[i])

        #     torch.cuda.nvtx.range_pop()
        #     c_mats[i] = c[row_count:(row_count + a_mats[i].size(0)),:]
        #     row_count += a_mats[i].size(0)
        #     torch.cuda.nvtx.range_push("nvtx-append-argc{}".format(i))
        #     arg_c_mats[i] = to_dgl_nd_for_write(c_mats[i])
        #     torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-cat-mats")
        # b_mats = torch.cat(b_mats)
        b_mats = b_mats.view(a_mats[0].size(1), -1)
        a_mats = torch.cat(a_mats)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-to-dgl-nd")
        arg_a_mats = F.to_dgl_nd(a_mats)
        arg_b_mats = F.to_dgl_nd(b_mats)
        arg_c_mats = to_dgl_nd_for_write(c_mats)
        arg_a_mats_rows = F.to_dgl_nd(a_mats_rows)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-capi-call")
        _CAPI_DGLKernelCAPIGEMMs(arg_a_mats, arg_b_mats, arg_c_mats, arg_a_mats_rows, \
                                    middim, outcol, len(a_mats_rows), total_edges)
        torch.cuda.nvtx.range_pop()

        return c_mats

    @staticmethod
    def backward(ctx, dC):
        return [None] * len(dC)

def fused_gemm(a1, b1, a2, b2):
    return FusedGEMM.apply(a1, b1, a2, b2)

def fused_gemm_spmm(a1, b1, a2, b2):
    return FusedGEMMSpMM.apply(a1, b1, a2, b2)

def fused_gemm_blockspmm(a_mats, b_mats):
    # return FusedGEMMBlockSpMM.apply(a1, b1, a2, b2)
    return FusedGEMMBlockSpMM.apply(a_mats, b_mats)

def fused_gemm_batchmm(a_mats, b_mats, a_mats_rows):
    return FusedGEMMBatchMM.apply(a_mats, b_mats, a_mats_rows)

def capi_gemms(a_mats, b_mats, a_mats_rows):
    return CAPIGEMMs.apply(a_mats, b_mats, a_mats_rows)

_init_api("dgl.fused_gemm")
