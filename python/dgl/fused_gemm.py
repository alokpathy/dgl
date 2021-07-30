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

        _CAPI_DGLKernelFGEMMSpMM(arg_a1, arg_b, arg_c, \
                                    a1.size(0), a1.size(1), b1.size(1), \
                                    arg_a2, \
                                    a2.size(0), a2.size(1), b2.size(1))
        
        c1 = c[:a1.size(0)]
        c2 = c[a1.size(0):]
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
            b = torch.cat((b1, b2), dim=0)
            grad_a = F.zeros((dC1.size(0) + dC2.size(0), b1.size(1)), dtype, ctx_cuda).cuda()

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
        elif ctx.needs_input_grad[0]:
            grad_a1 = torch.matmul(dC1, b1.t()).cuda()
        elif ctx.needs_input_grad[2]:
            grad_a2 = torch.matmul(dC2, b2.t()).cuda()

        if ctx.needs_input_grad[1] and ctx.needs_input_grad[3]:
            a1 = a1.t()
            a2 = a2.t()
            dC = torch.cat((dC1, dC2), dim=0)
            grad_b = F.zeros((a1.size(0) + a2.size(0), dC1.size(1)), dtype, ctx_cuda).cuda()

            arg_a1 = to_dgl_nd(a1)
            arg_a2 = to_dgl_nd(a2)
            arg_dC = to_dgl_nd(dC)

            arg_grad_b = to_dgl_nd_for_write(grad_b)

            # _CAPI_DGLKernelFGEMM(arg_a1, arg_dC1, arg_grad_b1, \
            #                         a1.size(0), a1.size(1), dC1.size(1), \
            #                         arg_a2, arg_dC2, arg_grad_b2, \
            #                         a2.size(0), a2.size(1), dC2.size(1))

            _CAPI_DGLKernelFGEMMSpMM(arg_a1, arg_dC, arg_grad_b, \
                                        a1.size(0), a1.size(1), dC.size(1), \
                                        arg_a2, \
                                        a2.size(0), a2.size(1), dC.size(1))
        elif ctx.needs_input_grad[1]:
            grad_b1 = torch.matmul(a1, dC1).cuda()
        elif ctx.needs_input_grad[3]:
            grad_b2 = torch.matmul(a2, dC2).cuda()

        return grad_a1, grad_b1, grad_a2, grad_b2

def fused_gemm(a1, b1, a2, b2):
    return FusedGEMM.apply(a1, b1, a2, b2)

def fused_gemm_spmm(a1, b1, a2, b2):
    return FusedGEMMSpMM.apply(a1, b1, a2, b2)

_init_api("dgl.fused_gemm")
