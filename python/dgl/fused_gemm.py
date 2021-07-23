from . import backend as F
from . import ndarray as nd
from ._ffi.function import _init_api

def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray(x)

def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray_for_write(x)

def fused_gemm(a, b):
    arg_a = to_dgl_nd(a)
    arg_b = to_dgl_nd(b)
    ctx = F.context(b)
    dtype = F.dtype(b)
    c = F.zeros((a.size(0), b.size(1)), dtype, ctx)
    arg_c = to_dgl_nd_for_write(c)
    _CAPI_DGLKernelFGEMM(arg_a, arg_b, arg_c, \
                            a.size(0), a.size(1), b.size(1), \
                            a.size(1), b.size(1), c.size(1))
    return c

_init_api("dgl.fused_gemm")
