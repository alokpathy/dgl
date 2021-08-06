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

timing = True

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
    def forward(ctx, a1, b1, a2, b2):
        global preproc_time
        global padding_time
        global atohalf_time
        global btohalf_time
        global ctohalf_time
        global blockedspmm_time
        global abctofloat_time

        preproc_start = torch.cuda.Event(enable_timing=True)
        preproc_stop = torch.cuda.Event(enable_timing=True)

        a1_pad = a1
        a2_pad = a2

        b1_pad = b1
        b2_pad = b2


        block_dim = max(a1.size(0), a1.size(1), a2.size(0), a2.size(1))

        torch.cuda.nvtx.range_push("nvtx-padding")
        start_time(preproc_start)
        # zero-pad A matrix height
        if a1.size(0) < block_dim:
            padding = block_dim - a1.size(0)
            # a1_pad = torch.cat((a1_pad, torch.Tensor(padding, a1_pad.size(1)).cuda().fill_(0)))
            a1_pad = torch.nn.functional.pad(a1_pad, (0, 0, 0, padding))

        if a2.size(0) < block_dim:
            padding = block_dim - a2.size(0)
            # a2_pad = torch.cat((a2_pad, torch.Tensor(padding, a2_pad.size(1)).cuda().fill_(0)))
            a2_pad = torch.nn.functional.pad(a2_pad, (0, 0, 0, padding))

        # zero-pad A matrix width / B matrix height
        if a1.size(1) < block_dim:
            padding = block_dim - a1.size(1)
            # a1_pad = torch.cat((a1_pad, torch.Tensor(a1_pad.size(0), padding).cuda().fill_(0)), dim=1)
            # b1_pad = torch.cat((b1_pad, torch.Tensor(padding, b1_pad.size(1)).cuda().fill_(0)), dim=0)
            a1_pad = torch.nn.functional.pad(a1_pad, (0, padding, 0, 0))
            b1_pad = torch.nn.functional.pad(b1_pad, (0, 0, 0, padding))

        if a2.size(1) < block_dim:
            padding = block_dim - a2.size(1)
            # a2_pad = torch.cat((a2_pad, torch.Tensor(a2_pad.size(0), padding).cuda().fill_(0)), dim=1)
            # b2_pad = torch.cat((b2_pad, torch.Tensor(padding, b2_pad.size(1)).cuda().fill_(0)), dim=0)
            a2_pad = torch.nn.functional.pad(a2_pad, (0, padding, 0, 0))
            b2_pad = torch.nn.functional.pad(b2_pad, (0, 0, 0, padding))

        padding_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        ctx.save_for_backward(a1_pad, b1_pad, a2_pad, b2_pad)
        ctx.a1_size = a1.size()
        ctx.a2_size = a2.size()
        ctx.b1_size = b1.size()
        ctx.b2_size = b2.size()

        torch.cuda.nvtx.range_push("nvtx-a-to-half")
        start_time(preproc_start)
        a1_pad = a1_pad.half()
        a2_pad = a2_pad.half()
        atohalf_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-b-to-half")
        start_time(preproc_start)
        arg_a1_pad = to_dgl_nd(a1_pad)
        arg_a2_pad = to_dgl_nd(a2_pad)

        b_pad = torch.cat((b1_pad, b2_pad), dim=0)
        b_pad = b_pad.half()
        arg_b_pad = to_dgl_nd(b_pad)
        btohalf_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-c-to-half")
        start_time(preproc_start)
        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1_pad)
        c = F.zeros((a1_pad.size(0) + a2_pad.size(0), b1_pad.size(1)), dtype, ctx_cuda).cuda()
        c = c.half()

        arg_c = to_dgl_nd_for_write(c)
        ctohalf_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-blocked-spmm")
        start_time(preproc_start)
        _CAPI_DGLKernelFGEMMBlockSpMM(arg_a1_pad, arg_b_pad, arg_c, \
                                        a1_pad.size(0), a1_pad.size(1), b1_pad.size(1), \
                                        arg_a2_pad, \
                                        a2_pad.size(0), a2_pad.size(1), b2_pad.size(1))
        blockedspmm_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("nvtx-abc-to-float")
        start_time(preproc_start)
        a1_pad = a1_pad.float()
        a2_pad = a2_pad.float()
        b_pad = b_pad.float()
        c = c.float()
        abctofloat_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-extractc1c2")
        c1 = c[:a1_pad.size(0)]
        c2 = c[a1_pad.size(0):]

        c1 = c1[:a1.size(0), :a1.size(1)]
        c2 = c2[:a2.size(0), :a2.size(1)]
        torch.cuda.nvtx.range_pop()

        if timing:
            print(f"preproc_time: {preproc_time}")
            print(f"padding_time: {padding_time}")
            print(f"atohalf_time: {atohalf_time}")
            print(f"btohalf_time: {btohalf_time}")
            print(f"ctohalf_time: {ctohalf_time}")
            print(f"blockedspmm_time: {blockedspmm_time}")
            print(f"abctofloat_time: {abctofloat_time}")

        return c1, c2

    @staticmethod
    def backward(ctx, dC1, dC2):
        a1_pad, b1_pad, a2_pad, b2_pad = ctx.saved_tensors
        a1_size = ctx.a1_size
        a2_size = ctx.a2_size
        b1_size = ctx.b1_size
        b2_size = ctx.b2_size

        block_dim = a1_pad.size(0)

        dC1_pad = dC1
        dC2_pad = dC2

        # zero-pad dC matrix height
        if dC1.size(0) < block_dim:
            padding = block_dim - dC1.size(0)
            dC1_pad = torch.cat((dC1_pad, torch.Tensor(padding, dC1_pad.size(1)).cuda().fill_(0)))

        if dC2.size(0) < block_dim:
            padding = block_dim - dC2.size(0)
            dC2_pad = torch.cat((dC2_pad, torch.Tensor(padding, dC2_pad.size(1)).cuda().fill_(0)))

        # zero-pad A matrix width / B matrix height
        if dC1.size(1) < block_dim:
            padding = block_dim - dC1.size(1)
            dC1_pad = torch.cat((dC1_pad, torch.Tensor(dC1_pad.size(0), padding).cuda().fill_(0)), dim=1)

        if dC2.size(1) < block_dim:
            padding = block_dim - dC2.size(1)
            dC2_pad = torch.cat((dC2_pad, torch.Tensor(dC2_pad.size(0), padding).cuda().fill_(0)), dim=1)

        grad_a1 = grad_b1 = None
        grad_a2 = grad_b2 = None

        ctx_cuda = F.context(b1_pad)
        dtype = F.dtype(b1_pad)

        # if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
        #     b1_pad_t = b1_pad.t()
        #     b2_pad_t = b2_pad.t()
        #     b_pad = torch.cat((b1_pad_t, b2_pad_t), dim=0)
        #     grad_a_pad = F.zeros((dC1_pad.size(0) + dC2_pad.size(0), b_pad.size(1)), dtype, ctx_cuda).cuda()


        #     # _CAPI_DGLKernelFGEMM(arg_dC1, arg_b1, arg_grad_a1, \
        #     #                         dC1.size(0), dC1.size(1), b1.size(1), \
        #     #                         arg_dC2, arg_b2, arg_grad_a2, \
        #     #                         dC2.size(0), dC2.size(1), b2.size(1))

        #     dC1_pad = dC1_pad.half().contiguous()
        #     dC2_pad = dC2_pad.half().contiguous()
        #     b_pad = b_pad.half().contiguous()
        #     grad_a_pad = grad_a_pad.half().contiguous()

        #     arg_dC1_pad = to_dgl_nd(dC1_pad)
        #     arg_dC2_pad = to_dgl_nd(dC2_pad)
        #     arg_b_pad = to_dgl_nd(b_pad)
        #     arg_grad_a_pad = to_dgl_nd_for_write(grad_a_pad)

        #     _CAPI_DGLKernelFGEMMBlockSpMM(arg_dC1_pad, arg_b_pad, arg_grad_a_pad, \
        #                                     dC1_pad.size(0), dC1_pad.size(1), b_pad.size(1), \
        #                                     arg_dC2_pad, \
        #                                     dC2_pad.size(0), dC2_pad.size(1), b_pad.size(1))

        #     dC1_pad = dC1_pad.float()
        #     dC2_pad = dC2_pad.float()
        #     b_pad = b_pad.float()
        #     grad_a_pad = grad_a_pad.float()

        #     grad_a1_pad = grad_a_pad[:a1_pad.size(0)]
        #     grad_a2_pad = grad_a_pad[a1_pad.size(0):]

        if ctx.needs_input_grad[0]:
            grad_a1_pad = torch.matmul(dC1_pad, b1_pad.t()).cuda()
        if ctx.needs_input_grad[2]:
            grad_a2_pad = torch.matmul(dC2_pad, b2_pad.t()).cuda()

        # if ctx.needs_input_grad[1] and ctx.needs_input_grad[3]:
        #     a1_pad_t = a1_pad.t().contiguous()
        #     a2_pad_t = a2_pad.t().contiguous()
        #     dC_pad = torch.cat((dC1_pad, dC2_pad), dim=0)
        #     grad_b_pad = F.zeros((a1_pad_t.size(0) + a2_pad_t.size(0), dC1_pad.size(1)), dtype, ctx_cuda).cuda()

        #     # _CAPI_DGLKernelFGEMM(arg_a1, arg_dC1, arg_grad_b1, \
        #     #                         a1.size(0), a1.size(1), dC1.size(1), \
        #     #                         arg_a2, arg_dC2, arg_grad_b2, \
        #     #                         a2.size(0), a2.size(1), dC2.size(1))

        #     a1_pad_t = a1_pad_t.half()
        #     a2_pad_t = a2_pad_t.half()
        #     dC_pad = dC_pad.half()
        #     grad_a_pad = grad_a_pad.half()

        #     arg_a1_pad_t = to_dgl_nd(a1_pad_t)
        #     arg_a2_pad_t = to_dgl_nd(a2_pad_t)
        #     arg_dC_pad = to_dgl_nd(dC_pad)
        #     arg_grad_b_pad = to_dgl_nd_for_write(grad_b_pad)

        #     _CAPI_DGLKernelFGEMMBlockSpMM(arg_a1_pad_t, arg_dC_pad, arg_grad_b_pad, \
        #                                     a1_pad_t.size(0), a1_pad_t.size(1), dC_pad.size(1), \
        #                                     arg_a2_pad_t, \
        #                                     a2_pad_t.size(0), a2_pad_t.size(1), dC_pad.size(1))
        #     a1_pad_t = a1_pad_t.float()
        #     a2_pad_t = a2_pad_t.float()
        #     dC_pad = dC_pad.float()
        #     grad_a_pad = grad_a_pad.float()

        #     grad_b1_pad = grad_b_pad[:b1_pad.size(0),:]
        #     grad_b2_pad = grad_b_pad[b1_pad.size(0):,:]
            
        if ctx.needs_input_grad[1]:
            grad_b1_pad = torch.matmul(a1_pad.t(), dC1_pad).cuda()
        if ctx.needs_input_grad[3]:
            grad_b2_pad = torch.matmul(a2_pad.t(), dC2_pad).cuda()

        grad_a1 = grad_a1_pad[:a1_size[0], :a1_size[1]]
        grad_a2 = grad_a2_pad[:a2_size[0], :a2_size[1]]
        grad_b1 = grad_b1_pad[:b1_size[0], :b1_size[1]]
        grad_b2 = grad_b2_pad[:b2_size[0], :b2_size[1]]

        return grad_a1, grad_b1, grad_a2, grad_b2

class FusedGEMMBatchMM(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, a1, b1, a2, b2, a3, b3, a4, b4):
    def forward(ctx, a_mats, b_mats):
        global preproc_time

        preproc_start = torch.cuda.Event(enable_timing=True)
        preproc_stop = torch.cuda.Event(enable_timing=True)

        a_mats_pad = []

        block_dim = max([j.size(0) for j in a_mats])

        torch.cuda.nvtx.range_push("nvtx-padding")
        start_time(preproc_start)
        for i in range(len(a_mats)):
            a_mats_pad.append(torch.nn.functional.pad(a_mats[i], (0, 0, 0, block_dim - a_mats[i].size(0))))

        preproc_time += stop_time(preproc_start, preproc_stop)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-construct-ab3d")
        a3d = torch.stack(a_mats_pad)
        b3d = torch.stack(b_mats)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-bmm")
        # with torch.cuda.amp.autocast():
        c = torch.bmm(a3d, b3d)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("nvtx-extract-c")
        c_layers = [j.squeeze(0) for j in torch.tensor_split(c, c.size(0))]
        for i in range(len(c_layers)):
            c_layers[i] = c_layers[i][:a_mats[i].size(0), :a_mats[i].size(1)]
        c = torch.cat(c_layers, dim=0)
        torch.cuda.nvtx.range_pop()

        if timing:
            print(f"preproc_time: {preproc_time}")

        return c

    @staticmethod
    def backward(ctx, dC1, dC2, dC3, dC4):
        return None, None, None, None, None, None, None, None

def fused_gemm(a1, b1, a2, b2):
    return FusedGEMM.apply(a1, b1, a2, b2)

def fused_gemm_spmm(a1, b1, a2, b2):
    return FusedGEMMSpMM.apply(a1, b1, a2, b2)

def fused_gemm_blockspmm(a1, b1, a2, b2):
    return FusedGEMMBlockSpMM.apply(a1, b1, a2, b2)

def fused_gemm_batchmm(a_mats, b_mats):
    # return FusedGEMMBatchMM.apply(a1, b1, a2, b2, a3, b3, a4, b4)
    return FusedGEMMBatchMM.apply(a_mats, b_mats)

_init_api("dgl.fused_gemm")
