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

class FusedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a1, b1, a2, b2):
        arg_a1 = to_dgl_nd(a1)
        arg_b1 = to_dgl_nd(b1)
        arg_a2 = to_dgl_nd(a2)
        arg_b2 = to_dgl_nd(b2)

        ctx.save_for_backward(a1, b1, a2, b2)

        ctx_cuda = F.context(b1)
        dtype = F.dtype(b1)
        c1 = F.zeros((a1.size(0), b1.size(1)), dtype, ctx_cuda).fill_(1).cuda()
        c2 = F.zeros((a2.size(0), b2.size(1)), dtype, ctx_cuda).fill_(1).cuda()

        arg_c1 = to_dgl_nd_for_write(c1)
        arg_c2 = to_dgl_nd_for_write(c2)
        print(f"a1.size: {a1.size()}")
        print(f"a2.size: {a2.size()}")
        print(f"b1.size: {b1.size()}")
        print(f"b2.size: {b2.size()}")
        _CAPI_DGLKernelFGEMM(arg_a1, arg_b1, arg_c1, \
                                a1.size(0), a1.size(1), b1.size(1), \
                                arg_a2, arg_b2, arg_c2, \
                                a2.size(0), a2.size(1), b2.size(1))
        return c1, c2

    @staticmethod
    def backward(ctx, dC):
        print("in custom backward")
        # a, b = ctx.saved_tensors
        # arg_dC = to_dgl_nd(dC)

        # grad_a = grad_b = None

        # ctx_cuda = F.context(b)
        # dtype = F.dtype(b)

        # if ctx.needs_input_grad[0]:
        #     b = b.t()
        #     grad_a = F.zeros((dC.size(0), b.size(1)), dtype, ctx_cuda).cuda()

        #     arg_b = to_dgl_nd(b)
        #     arg_grad_a = to_dgl_nd_for_write(grad_a)

        #     _CAPI_DGLKernelFGEMM(arg_dC, arg_b, arg_grad_a, \
        #                             dC.size(0), dC.size(1), b.size(1))

        # if ctx.needs_input_grad[1]:
        #     a = a.t()
        #     grad_b = F.zeros((a.size(0), dC.size(1)), dtype, ctx_cuda).cuda()

        #     arg_a = to_dgl_nd(a)
        #     arg_grad_b = to_dgl_nd_for_write(grad_b)

        #     _CAPI_DGLKernelFGEMM(arg_a, arg_dC, arg_grad_b, \
        #                             a.size(0), a.size(1), dC.size(1))

        # return grad_a, grad_b
        return None, None, None, None

def fused_gemm(a1, b1, a2, b2):
    return FusedGEMM.apply(a1, b1, a2, b2)

_init_api("dgl.fused_gemm")
