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
    def forward(ctx, a, b):
        arg_a = to_dgl_nd(a)
        arg_b = to_dgl_nd(b)

        ctx.save_for_backward(a, b)

        ctx_cuda = F.context(b)
        dtype = F.dtype(b)
        c = F.zeros((a.size(0), b.size(1)), dtype, ctx_cuda).cuda()
        arg_c = to_dgl_nd_for_write(c)
        _CAPI_DGLKernelFGEMM(arg_a, arg_b, arg_c, \
                                a.size(0), a.size(1), b.size(1), \
                                a.size(1), b.size(1), c.size(1))
        return c

    @staticmethod
    def backward(ctx, dC):
        a, b = ctx.saved_tensors
        arg_dC = to_dgl_nd(dC)

        grad_a = grad_b = None

        ctx_cuda = F.context(b)
        dtype = F.dtype(b)

        if ctx.needs_input_grad[0]:
            b = b.t()
            grad_a = F.zeros((dC.size(0), b.size(1)), dtype, ctx_cuda).cuda()

            arg_b = to_dgl_nd(b)
            arg_grad_a = to_dgl_nd_for_write(grad_a)

            _CAPI_DGLKernelFGEMM(arg_dC, arg_b, arg_grad_a, \
                                    dC.size(0), dC.size(1), b.size(1), \
                                    dC.size(1), b.size(1), grad_a.size(1))

        if ctx.needs_input_grad[1]:
            a = a.t()
            grad_b = F.zeros((a.size(0), dC.size(1)), dtype, ctx_cuda).cuda()

            arg_a = to_dgl_nd(a)
            arg_grad_b = to_dgl_nd_for_write(grad_b)

            _CAPI_DGLKernelFGEMM(arg_a, arg_dC, arg_grad_b, \
                                    a.size(0), a.size(1), dC.size(1), \
                                    a.size(1), dC.size(1), grad_b.size(1))

        return grad_a, grad_b

def fused_gemm(a, b):
    return FusedGEMM.apply(a, b)

_init_api("dgl.fused_gemm")
