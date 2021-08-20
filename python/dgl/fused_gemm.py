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
    def forward(ctx, a_mats, b_mats, a_mats_rows):
        torch.cuda.nvtx.range_push("nvtx-cata")
        # a = torch.cat(a_mats, dim=0)
        a = a_mats
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-to-dgl-nd")
        b = b_mats
        arg_a = to_dgl_nd(a)
        arg_b = to_dgl_nd(b)
        arg_a_mats_rows = to_dgl_nd(a_mats_rows)
        # ctx.save_for_backward(a1, b1, a2, b2)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-construct-mats")
        ctx_cuda = F.context(b_mats[0])
        dtype = F.dtype(b_mats[0])
        c = torch.cuda.FloatTensor(a.size(0), b.size(1))
        dA_csrOffsets = torch.cuda.FloatTensor(a.size(0) + 1)
        dA_columns = torch.cuda.FloatTensor(a.numel())

        arg_c = to_dgl_nd_for_write(c)
        arg_dA_csroffsets = to_dgl_nd_for_write(dA_csrOffsets)
        arg_dA_columns = to_dgl_nd_for_write(dA_columns)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-capi-spmm")
        _CAPI_DGLKernelFGEMMSpMM(arg_a, arg_b, arg_c, arg_a_mats_rows, \
                                    arg_dA_csroffsets, arg_dA_columns, a.size(0), a.size(1), b.size(1))
        torch.cuda.nvtx.range_pop()
        
        if timing:
            print(f"preproc_time: {preproc_time}")

        return c

    @staticmethod
    def backward(ctx, dC):
        return None, None, None

class FusedGEMMBlockSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_mats, b_mats, a_mats_rows):
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

        # block_dim = a_mats[0].size(1)
        block_dim = a_mats.size(1)

        torch.cuda.nvtx.range_push("nvtx-padding")
        padding = 0
        for i in range(len(a_mats_rows)):
            if a_mats_rows[i] % block_dim != 0:
                padding += block_dim - (a_mats_rows[i].item() % block_dim)

        num_edges = torch.sum(a_mats_rows).item()
        a_pad = torch.cuda.HalfTensor(num_edges + padding, block_dim)
        # a_mat_cat = torch.cat(a_mats).half()
        a_mat_cat = a_mats.half()

        arg_a_pad = to_dgl_nd(a_pad)
        arg_a_mat = to_dgl_nd(a_mat_cat)
        arg_a_mats_rows = to_dgl_nd(a_mats_rows)

        start_time(preproc_start)
        _CAPI_DGLKernelPadA2D(arg_a_pad, arg_a_mat, arg_a_mats_rows, num_edges + padding, \
                                    block_dim, a_mats_rows.size(0))
        padding_time += stop_time(preproc_start, preproc_stop)

        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-concat-mats")
        # a_pad = torch.cat(a_mats_pad, dim=0).half()
        b_pad = b_mats.half()
        ctx.save_for_backward(a_pad, b_pad)
        ctx.num_blocks = len(a_mats_rows)
        # arg_a_pad = to_dgl_nd(a_pad)
        arg_b_pad = to_dgl_nd(b_pad)
        # arg_a_mats_rows = to_dgl_nd(a_mats_rows)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-construct-c")
        ctx_cuda = F.context(b_pad)
        c_pad = F.zeros((a_pad.size(0), b_pad.size(1)), torch.half, ctx_cuda).cuda()

        arg_c_pad = to_dgl_nd_for_write(c_pad)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-blocked-spmm")
        _CAPI_DGLKernelFGEMMBlockSpMM(arg_a_pad, arg_b_pad, arg_c_pad, arg_a_mats_rows, \
                                        a_pad.size(0), a_pad.size(1), b_pad.size(1), block_dim, len(a_mats_rows))
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("nvtx-abc-to-float")
        c_pad = c_pad.float()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-extractc")
        c = torch.cuda.FloatTensor(num_edges, b_mats.size(1))
        arg_c = to_dgl_nd_for_write(c)
        arg_c_pad = F.to_dgl_nd(c_pad)
        _CAPI_DGLKernelUnpadC2D(arg_c, arg_c_pad, arg_a_mats_rows, c.size(0), c.size(1), len(a_mats_rows))
        # c = []
        # row_idx = 0
        # for i in range(len(a_mats_rows)):
        #     if a_mats_rows[i] % block_dim == 0:
        #         padding = 0
        #     else:
        #         padding = block_dim - (a_mats_rows[i] % block_dim)
        #     output_ci = c_pad[row_idx:(row_idx + a_mats_rows[i] + padding)]
        #     c.append(output_ci[:a_mats_rows[i], :b_mats.size(1)])
        #     row_idx += a_mats_rows[i] + padding
        # c = torch.cat(c)
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

        return None, None, None

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

        # torch.cuda.nvtx.range_push("nvtx-mats-to-half")
        # a_mats = a_mats.half()
        # b_mats = b_mats.half()
        # c_mats = c_mats.half()
        # torch.cuda.nvtx.range_pop()

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

        # torch.cuda.nvtx.range_push("nvtx-mats-to-float")
        # c_mats = c_mats.float()
        # torch.cuda.nvtx.range_pop()

        return c_mats

    @staticmethod
    def backward(ctx, dC):
        return [None] * len(dC)

def fused_gemm(a1, b1, a2, b2):
    return FusedGEMM.apply(a1, b1, a2, b2)

def fused_gemm_spmm(a_mats, b_mats, a_mats_rows):
    return FusedGEMMSpMM.apply(a_mats, b_mats, a_mats_rows)

def fused_gemm_blockspmm(a_mats, b_mats, a_mats_rows):
    return FusedGEMMBlockSpMM.apply(a_mats, b_mats, a_mats_rows)

def fused_gemm_batchmm(a_mats, b_mats, a_mats_rows):
    return FusedGEMMBatchMM.apply(a_mats, b_mats, a_mats_rows)

def capi_gemms(a_mats, b_mats, a_mats_rows):
    return CAPIGEMMs.apply(a_mats, b_mats, a_mats_rows)

_init_api("dgl.fused_gemm")
