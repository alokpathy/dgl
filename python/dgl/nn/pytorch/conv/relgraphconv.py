"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import functools
import numpy as np
import torch as th
from torch import nn

from .... import function as fn
from .. import utils
from ....base import DGLError
from .... import edge_subgraph
from ....fused_gemm import capi_gemms, fused_gemm, fused_gemm_spmm, fused_gemm_blockspmm, fused_gemm_batchmm
from ....ops.spmm import gspmm

import torch.autograd.profiler as profiler

timing = False

def start_time(timer):
    if timing:
        timer.record()

def stop_time(start_timer, stop_timer):
    if timing:
        stop_timer.record()
        th.cuda.synchronize()
        return start_timer.elapsed_time(stop_timer)
    else:
        return 0.0

def bdd_lowmem_einsum(h_t, weight, num_rels, num_bases, submat_in, submat_out, out_feat):
    msg = []
    edge_count = 0
    elem_count = 0

    for etype in range(num_rels):
        if h_t[etype].shape[0] == 0:
            continue
        edge_count += h_t[etype].shape[0]
        tmp_w = weight[etype].view(num_bases, submat_in, submat_out)
        tmp_h = h_t[etype].view(-1, num_bases, submat_in)
        th.cuda.nvtx.range_push("nvtx-lowmem-einsums-type{}".format(etype))
        elem_count += tmp_w.numel() + tmp_h.numel()
        result = th.einsum('abc,bcd->abd', tmp_h, tmp_w)
        th.cuda.nvtx.range_pop()
        msg.append(result.reshape(-1, out_feat))

    msg = th.cat(msg)
    return msg, edge_count, elem_count

def bdd_lowmem_loop(h_t, weight, num_rels, num_bases, submat_in, submat_out, out_feat):
    msg = []
    edge_count = 0
    elem_count = 0

    for etype in range(num_rels):
        if h_t[etype].shape[0] == 0:
            continue
        edge_count += h_t[etype].shape[0]
        tmp_w = weight[etype].view(num_bases, submat_in, submat_out)
        tmp_h = h_t[etype].view(-1, num_bases, submat_in)
        th.cuda.nvtx.range_push("nvtx-lowmem-loop-type{}".format(etype))
        elem_count += tmp_w.numel() + tmp_h.numel()

        result = th.cuda.FloatTensor(tmp_h.size(0), tmp_h.size(1), tmp_w.size(2))
        for i in range(tmp_h.size(0)): # iterate over each edge
            for j in range(tmp_w.size(2)): 
                result[i,:,j] = (tmp_h[i,:,:] * tmp_w[:,:,j]).sum(dim=1)
        msg.append(result.reshape(-1, out_feat))

    msg = th.cat(msg)
    return msg, edge_count, elem_count

def bdd_lowmem_matmuls(h_t, weight, num_rels, num_bases, submat_in, submat_out, out_feat):
    msg = []
    edge_count = 0
    elem_count = 0

    for etype in range(num_rels):
        if h_t[etype].shape[0] == 0:
            continue
        edge_count += h_t[etype].shape[0]
            
        # # no num_bases assumption
        # result = th.cuda.FloatTensor(h_t[etype].size(0), out_feat)
        # tmp_w = weight[etype].view(num_bases, submat_in, submat_out)
        # msg_type = []

        # for i in range(num_bases):
        #     stride = h_t[etype].size(1) // num_bases
        #     col_start = i * stride
        #     col_end   = (i + 1) * stride
        #     result[:,col_start:col_end] = th.matmul(h_t[etype][:,col_start:col_end], tmp_w[i])
        #     elem_count += h_t[etype][:,col_start:col_end].numel() + tmp_w[i].numel()
        #     # result = th.matmul(h_t[etype][:,col_start:col_end], tmp_w[i])
        #     # msg_type.append(result)
        # # result = th.cat(msg_type, dim=1)
        # msg.append(result)

        # assumes num_bases = 1
        elem_count += h_t[etype].numel() + weight[etype].numel()
        result = th.matmul(h_t[etype], weight[etype].view(submat_in, submat_out))
        msg.append(result)

    msg = th.cat(msg)
    return msg, edge_count, elem_count

def bdd_lowmem_capi_matmuls(h_t, weight, num_rels, num_bases, submat_in, submat_out, out_feat, \
                                nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    th.cuda.nvtx.range_push("nvtx-capigemms-preproc")
    # th.cuda.nvtx.range_push("nvtx-preproc-nonemptyrels")
    # nonempty_rels = []
    # for etype in range(num_rels):
    #     if h_t[etype].shape[0] > 0:
    #         edge_count += h_t[etype].shape[0]
    #         nonempty_rels.append(etype)
    # th.cuda.nvtx.range_pop()

    th.cuda.nvtx.range_push("nvtx-preproc-mergew")
    # weight_merged = [(weight.data)[j] for j in nonempty_rels]
    # weight_merged = th.tensor_split(weight.data, weight.data.size(0), dim=0)
    # weight_merged = [weight_merged[j].squeeze(0) for j in nonempty_rels]
    weight_merged = th.index_select(weight.data, 0, nonempty_rels)
    th.cuda.nvtx.range_pop()

    th.cuda.nvtx.range_push("nvtx-preproc-mergeh")
    nonempty_rels = nonempty_rels.cpu()
    h_t_merged = [h_t[j] for j in nonempty_rels]
    th.cuda.nvtx.range_pop()
    th.cuda.nvtx.range_pop()
    results = capi_gemms(h_t_merged, weight_merged, etypes)

    return results, edge_count, elem_count

def bdd_lowmem_fgemm_batchmm(h_t, weight, num_rels, num_bases, submat_in, submat_out, out_feat, \
                                nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    msg = []

    merge_count = len(nonempty_rels)
    for i in range(0, len(nonempty_rels), merge_count):
        if i + (merge_count - 1) < len(nonempty_rels):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types{}".format(str(etypes)))

            th.cuda.nvtx.range_push("nvtx-mergew")
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-nonemptyrels-cpu")
            nonempty_rels = nonempty_rels.cpu()
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-mergeh")
            h_t_merged = [h_t[j] for j in nonempty_rels]
            th.cuda.nvtx.range_pop()

            results = fused_gemm_batchmm(h_t_merged, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count

def bdd_lowmem_fgemm_spmm(h, weight, num_rels, num_bases, submat_in, submat_out, out_feat, \
                                    nonempty_rels, etypes):

    edge_count = 0
    elem_count = 0

    msg = []
    merge_count = nonempty_rels.size(0)
    for i in range(0, len(nonempty_rels), merge_count):
        if i + (merge_count - 1) < nonempty_rels.size(0):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types{}".format(str(nonempty_rels)))

            # for j in etypes:
            #     elem_count += h_t[j].numel() + weight[j].numel()

            # with th.cuda.amp.autocast():
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            # weight_merged = [weight[j] for j in nonempty_rels]

            th.cuda.nvtx.range_push("nvtx-mergew")
            # weight_merged = [weight[j] for j in etypes]
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            th.cuda.nvtx.range_pop()

            # th.cuda.nvtx.range_push("nvtx-nonemptyrels-cpu")
            # nonempty_rels = nonempty_rels.cpu()
            # th.cuda.nvtx.range_pop()

            # th.cuda.nvtx.range_push("nvtx-mergeh")
            # print(f"type(h_t): {type(h_t)}")
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            # th.cuda.nvtx.range_pop()

            # results = fused_gemm_spmm(h_t_merged, weight_merged, etypes)
            results = fused_gemm_spmm(h, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count

def bdd_lowmem_fgemm_blockspmm(h, weight, num_rels, num_bases, submat_in, submat_out, out_feat, \
                                    nonempty_rels, etypes):

    edge_count = 0
    elem_count = 0

    msg = []
    merge_count = nonempty_rels.size(0)
    for i in range(0, len(nonempty_rels), merge_count):
        if i + (merge_count - 1) < nonempty_rels.size(0):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types")

            # for j in etypes:
            #     elem_count += h_t[j].numel() + weight[j].numel()

            # with th.cuda.amp.autocast():
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            # weight_merged = [weight[j] for j in nonempty_rels]

            th.cuda.nvtx.range_push("nvtx-mergew")
            # weight_merged = [weight[j] for j in etypes]
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_push("nvtx-mergeh")
            nonempty_rels = nonempty_rels.cpu()
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            th.cuda.nvtx.range_pop()

            # results = fused_gemm_blockspmm(h_t_merged, weight_merged, etypes)
            results = fused_gemm_blockspmm(h, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count

def lowmem_matmul(h_t, weight, num_rels):
    bmm_start = th.cuda.Event(enable_timing=True)
    bmm_stop = th.cuda.Event(enable_timing=True)

    msg = []
    bmm_time = 0.0
    dim_count = 0

    print(f"num_rels: {num_rels}")
    for etype in range(num_rels):
        if h_t[etype].shape[0] == 0:
            continue
        th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
        start_time(bmm_start)
        # with th.cuda.amp.autocast():
        dim_count += h_t[etype].numel() + weight[etype].numel()
        result = th.matmul(h_t[etype], weight[etype])
        ##  print(f"etype: {etype} result: {result}")
        msg.append(result)
        bmm_time += stop_time(bmm_start, bmm_stop)
        th.cuda.nvtx.range_pop()

    return msg, bmm_time, dim_count

def lowmem_capi_matmul(h, weight, num_rels, nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    th.cuda.nvtx.range_push("nvtx-capigemms-preproc")
    # th.cuda.nvtx.range_push("nvtx-preproc-nonemptyrels")
    # nonempty_rels = []
    # for etype in range(num_rels):
    #     if h_t[etype].shape[0] > 0:
    #         edge_count += h_t[etype].shape[0]
    #         nonempty_rels.append(etype)
    # th.cuda.nvtx.range_pop()

    th.cuda.nvtx.range_push("nvtx-preproc-mergew")
    # weight_merged = [(weight.data)[j] for j in nonempty_rels]
    # weight_merged = th.tensor_split(weight.data, weight.data.size(0), dim=0)
    # weight_merged = [weight_merged[j].squeeze(0) for j in nonempty_rels]
    weight_merged = th.index_select(weight.data, 0, nonempty_rels)
    th.cuda.nvtx.range_pop()

    th.cuda.nvtx.range_push("nvtx-preproc-mergeh")
    # nonempty_rels = nonempty_rels.cpu()
    # h_t_merged = [h_t[j] for j in nonempty_rels]
    h_t_merged = h
    th.cuda.nvtx.range_pop()
    th.cuda.nvtx.range_pop()
    results = capi_gemms(h_t_merged, weight_merged, etypes)

    return results, edge_count, elem_count

def lowmem_fgemm_spgemm(h_t, weight, num_rels):
    bmm_start = th.cuda.Event(enable_timing=True)
    bmm_stop = th.cuda.Event(enable_timing=True)

    msg = []
    bmm_time = 0.0
    dim_count = 0

    nonempty_rels = []
    for etype in range(num_rels):
        if h_t[etype].shape[0] > 0:
            nonempty_rels.append(etype)

    for i in range(0, len(nonempty_rels), 2):
        if i + 1 < len(nonempty_rels):
            etype1 = nonempty_rels[i]
            etype2 = nonempty_rels[i + 1]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-type{}-type{}".format(etype1, etype2))
            start_time(bmm_start)
            dim_count += h_t[etype1].numel() + weight[etype1].numel() + \
                            h_t[etype2].numel() + weight[etype2].numel()
            # with th.cuda.amp.autocast():
            # result = th.matmul(h_t[etype], weight[etype])
            result1, result2 = fused_gemm(h_t[etype1], weight[etype1], h_t[etype2], weight[etype2])
            msg.append(result1)
            msg.append(result2)
            bmm_time += stop_time(bmm_start, bmm_stop)
            th.cuda.nvtx.range_pop()
        else:
            etype = nonempty_rels[i]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
            start_time(bmm_start)
            dim_count += h_t[etype].numel() + weight[etype].numel()
            # with th.cuda.amp.autocast():
            result = th.matmul(h_t[etype], weight[etype])
            msg.append(result)
            bmm_time += stop_time(bmm_start, bmm_stop)
            th.cuda.nvtx.range_pop()

    return msg, bmm_time, dim_count

def lowmem_fgemm_gemm(h_t, weight, num_rels):
    bmm_start = th.cuda.Event(enable_timing=True)
    bmm_stop = th.cuda.Event(enable_timing=True)

    msg = []
    bmm_time = 0.0
    dim_count = 0

    nonempty_rels = []
    for etype in range(num_rels):
        if h_t[etype].shape[0] > 0:
            nonempty_rels.append(etype)

    for i in range(0, len(nonempty_rels), 2):
        if i + 1 < len(nonempty_rels):
            etype1 = nonempty_rels[i]
            etype2 = nonempty_rels[i + 1]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-type{}-type{}".format(etype1, etype2))
            start_time(bmm_start)
            dim_count += h_t[etype1].numel() + weight[etype1].numel() + \
                            h_t[etype2].numel() + weight[etype2].numel()
            # with th.cuda.amp.autocast():
            result1, result2 = fused_gemm(h_t[etype1], weight[etype1], h_t[etype2], weight[etype2])
            # stacked_h = th.cat((h_t[etype1], h_t[etype2]), dim=0)
            # stacked_w = th.cat((weight[etype1], weight[etype2]), dim=1)
            
            # result = th.matmul(stacked_h, stacked_w)

            result1, result2 = th.split(result, [h_t[etype1].size(0), h_t[etype2].size(0)])
            result1 = result1[:,:weight[etype1].size(1)]
            result2 = result2[:,weight[etype2].size(1):]

            msg.append(result1)
            msg.append(result2)
            bmm_time += stop_time(bmm_start, bmm_stop)
            th.cuda.nvtx.range_pop()
        else:
            etype = nonempty_rels[i]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-type{}-type{}".format(etype1, etype2))
            start_time(bmm_start)
            dim_count += h_t[etype].numel() + weight[etype].numel()
            # with th.cuda.amp.autocast():
            result = th.matmul(h_t[etype], weight[etype])
            msg.append(result)
            bmm_time += stop_time(bmm_start, bmm_stop)
            th.cuda.nvtx.range_pop()

    return msg, bmm_time, dim_count

def lowmem_fgemm_spmm(h, weight, num_rels, nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    msg = []
    merge_count = nonempty_rels.size(0)
    for i in range(0, len(nonempty_rels), merge_count):
        if i + (merge_count - 1) < nonempty_rels.size(0):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types{}".format(str(nonempty_rels)))

            # for j in etypes:
            #     elem_count += h_t[j].numel() + weight[j].numel()

            # with th.cuda.amp.autocast():
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            # weight_merged = [weight[j] for j in nonempty_rels]

            th.cuda.nvtx.range_push("nvtx-mergew")
            # weight_merged = [weight[j] for j in etypes]
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            weight_merged = weight_merged.view(-1, weight.data.size(2))
            th.cuda.nvtx.range_pop()

            # th.cuda.nvtx.range_push("nvtx-nonemptyrels-cpu")
            # nonempty_rels = nonempty_rels.cpu()
            # th.cuda.nvtx.range_pop()

            # th.cuda.nvtx.range_push("nvtx-mergeh")
            # print(f"type(h_t): {type(h_t)}")
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            # th.cuda.nvtx.range_pop()

            # results = fused_gemm_spmm(h_t_merged, weight_merged, etypes)
            results = fused_gemm_spmm(h, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count

def lowmem_fgemm_blockspmm(h, weight, num_rels, nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    msg = []
    th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-preproc")
    merge_count = nonempty_rels.size(0)
    len_nonempty_rels = len(nonempty_rels)
    th.cuda.nvtx.range_pop()
    for i in range(0, len_nonempty_rels, merge_count):
        if i + (merge_count - 1) < nonempty_rels.size(0):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types")

            th.cuda.nvtx.range_push("nvtx-mergew")
            # weight_merged = [weight[j] for j in etypes]
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            weight_merged = weight_merged.view(-1, weight.data.size(2))
            th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_push("nvtx-mergeh")
            # nonempty_rels = nonempty_rels.cpu()
            # h_t_merged = [h_t[j] for j in nonempty_rels]
            th.cuda.nvtx.range_pop()

            # results = fused_gemm_blockspmm(h_t_merged, weight_merged, etypes)
            results = fused_gemm_blockspmm(h, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count

def lowmem_fgemm_batchmm(h_t, weight, num_rels, nonempty_rels, etypes):
    edge_count = 0
    elem_count = 0

    msg = []

    merge_count = len(nonempty_rels)
    for i in range(0, len(nonempty_rels), merge_count):
        if i + (merge_count - 1) < len(nonempty_rels):
            # etypes = nonempty_rels[i:(i + merge_count)]
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-fused-types{}".format(str(etypes)))

            th.cuda.nvtx.range_push("nvtx-mergew")
            weight_merged = th.index_select(weight.data, 0, nonempty_rels)
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-nonemptyrels-cpu")
            nonempty_rels = nonempty_rels.cpu()
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-mergeh")
            h_t_merged = [h_t[j] for j in nonempty_rels]
            th.cuda.nvtx.range_pop()

            results = fused_gemm_batchmm(h_t_merged, weight_merged, etypes)
            msg.append(results)
            th.cuda.nvtx.range_pop()
        else:
            for j in range(i, len(nonempty_rels)):
                etype = nonempty_rels[j]
                th.cuda.nvtx.range_push("nvtx-lowmem-matmuls-type{}".format(etype))
                start_time(bmm_start)
                dim_count += h_t[etype].numel() + weight[etype].numel()
                # with th.cuda.amp.autocast():
                result = th.matmul(h_t[etype], weight[etype])
                # print(f"etype: {etype} result: {result}")
                msg.append(result)
                th.cuda.nvtx.range_pop()

    return results, edge_count, elem_count
class RelGraphConv(nn.Module):
    r"""Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str
        Which weight regularizer to use "basis" or "bdd".
        "basis" is short for basis-diagonal-decomposition.
        "bdd" is short for block-diagonal-decomposition.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function. Default: False.
        This option trades speed with memory consumption, and will slowdown the forward/backward.
        Turn it on when you encounter OOM problem during training or evaluation. Default: ``False``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> conv.weight.shape
    torch.Size([2, 10, 2])
    >>> etype = th.tensor(np.array([0,1,2,0,1,2]).astype(np.int64))
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)

    >>> # One-hot input
    >>> one_hot_feat = th.tensor(np.array([0,1,2,3,4,5]).astype(np.int64))
    >>> res = conv(g, one_hot_feat, etype)
    >>> res
    tensor([[ 0.5925,  0.0985],
            [-0.3953,  0.8408],
            [-0.9819,  0.5284],
            [-1.0085, -0.1721],
            [ 0.5962,  1.2002],
            [ 0.0365, -0.3532]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 low_mem=False,
                 dropout=0.0,
                 layer_norm=False):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % self.num_bases != 0 or out_feat % self.num_bases != 0:
                raise ValueError(
                    'Feature size must be a multiplier of num_bases (%d).'
                    % self.num_bases
                )
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(th.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

            if self.num_bases == 1 and self.low_mem:
                self.weight = nn.Parameter(self.weight.view(self.num_rels, self.submat_in, self.submat_out))

            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges, etypes, nonempty_rels=None, nonempty_etypes=None, norm=None):
        """Message function for basis regularizer.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Input to DGL message UDF.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either:

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        """

        th.cuda.nvtx.range_push("nvtx-basis-mult")
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        th.cuda.nvtx.range_pop()

        if isinstance(edges, th.Tensor):
            h = edges
        else:
            h = edges.src['h']
        device = h.device

        if h.dtype == th.int64 and h.ndim == 1:
            # Each element is the node's ID. Use index select: weight[etypes, h, :]
            # The following is a faster version of it.
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device),
                                              th.tensor(etypes, device=device))
            weight = weight.view(-1, weight.shape[2])
            flatidx = etypes * weight.shape[1] + h
            msg = weight.index_select(0, flatidx)
        elif self.low_mem:
            # A more memory-friendly implementation.
            # Calculate msg @ W_r before put msg into edge.
            assert isinstance(etypes, list)
            th.cuda.nvtx.range_push("nvtx-lowmem-matmuls")
            th.cuda.nvtx.range_push("nvtx-lowmem-instantiation")

            # # Comment if using block spmm
            # th.cuda.nvtx.range_push("nvtx-lowmem-split")
            # h_t = th.split(h, etypes)
            # th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-lowmem-nonemptyrels")
            nonempty_rels = th.LongTensor([i for i in range(len(etypes)) if etypes[i] != 0]).cuda()
            th.cuda.nvtx.range_pop()

            # th.cuda.nvtx.range_push("nvtx-lowmem-etypes")
            # etypes = th.IntTensor([i for i in etypes if i != 0])
            # th.cuda.nvtx.range_pop()
            etypes = nonempty_etypes
            th.cuda.nvtx.range_pop()
            
            # matmul
            # msg, bmm_time, dim_count = lowmem_matmul(h_t, weight, self.num_rels)

            # capi matmul
            # msg, bmm_time, dim_count = lowmem_capi_matmul(h_t, weight, self.num_rels, nonempty_rels, etypes)
            # msg, bmm_time, dim_count = lowmem_capi_matmul(h, weight, self.num_rels, nonempty_rels, etypes)

            # fused gemm with spgemm
            # msg, bmm_time, dim_count = lowmem_fgemm_spgemm(h_t, weight, self.num_rels)

            # fused gemm with larger gemm
            # msg, bmm_time, dim_count = lowmem_fgemm_gemm(h_t, weight, self.num_rels)

            # fused gemm with spmm
            # msg, bmm_time, dim_count = lowmem_fgemm_spmm(h, weight, self.num_rels, nonempty_rels, etypes)

            # # fused gemm with block spmm
            msg, bmm_time, dim_count = lowmem_fgemm_blockspmm(h, weight, self.num_rels, \
                                                                    nonempty_rels, etypes)

            # fused gemm with bmm
            # msg, bmm_time, dim_count = lowmem_fgemm_batchmm(h_t, weight, self.num_rels, nonempty_rels, etypes)

            if timing:
                print(f"bmm_time: {bmm_time} dim_count: {dim_count}", flush=True)
            if not th.is_tensor(msg):
                msg = th.cat(msg)
            th.cuda.nvtx.range_pop()
        else:
            bmm_start = th.cuda.Event(enable_timing=True)
            bmm_stop = th.cuda.Event(enable_timing=True)

            # Use batched matmult
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device),
                                              th.tensor(etypes, device=device))
            th.cuda.nvtx.range_push("nvtx-index-select")
            weight = weight.index_select(0, etypes)
            th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_push("nvtx-highmem-batchmm")
            # print(f"h.size: {h.unsqueeze(1).size()} weight.size: {weight.size()}")
            start_time(bmm_start)
            # with th.cuda.amp.autocast():
            dim_count = h.numel() + weight.numel()
            msg = th.bmm(h.unsqueeze(1), weight).squeeze(1)
            if timing:
                print(f"bmm_time: {stop_time(bmm_start, bmm_stop)} dim_count: {dim_count}")
            th.cuda.nvtx.range_pop()
            # print(f"etypes.size: {etypes.size()} h.size: {h.unsqueeze(1).size()} weight.size: {weight.size()} msg.size: {msg.size()}")

        if norm is not None and isinstance(norm, th.Tensor):
            msg = msg * norm
        elif 'norm' in edges.data:
            msg = msg * edges.data['norm']

        print(f"msg.size: {msg.size()}")
        print(f"msg: {msg}")
        print(f"msg.sum: {msg.sum()}")
        return {'msg': msg}

    def bdd_message_func(self, edges, etypes, nonempty_rels=None, nonempty_etypes=None):
        global epoch
        """Message function for block-diagonal-decomposition regularizer.

        Parameters
        ----------
        edges : dgl.EdgeBatch
            Input to DGL message UDF.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either:

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        """
        h = edges.src['h']
        device = h.device

        if h.dtype == th.int64 and h.ndim == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')

        if self.low_mem:
            # A more memory-friendly implementation.
            # Calculate msg @ W_r before put msg into edge.
            assert isinstance(etypes, list)
            th.cuda.nvtx.range_push("nvtx-lowmem-instantiation")

            # # comment if using block spmm
            # th.cuda.nvtx.range_push("nvtx-lowmem-split")
            # h_t = th.split(h, etypes)
            # th.cuda.nvtx.range_pop()

            etypes = nonempty_etypes
            msg = []
            th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_push("nvtx-lowmem-bdd-matmuls")
            # elem_count = 0
            # edge_count = 0

            # # einsum 
            # msg, edge_count, elem_count = bdd_lowmem_einsum(h_t, self.weight, \
            #                                             self.num_rels, self.num_bases, \
            #                                             self.submat_in, self.submat_out, self.out_feat)

            # manual loop over tensors
            # msg, edge_count, elem_count = bdd_lowmem_loop(h_t, self.weight, \
            #                                                     self.num_rels, self.num_bases, \
            #                                                     self.submat_in, self.submat_out, self.out_feat)

            # series of GEMMs
            # msg, edge_count, elem_count = bdd_lowmem_matmuls(h_t, self.weight, \
            #                                                     self.num_rels, self.num_bases, \
            #                                                     self.submat_in, self.submat_out, self.out_feat)

            # # CAPI series of GEMMs
            # msg, edge_count, elem_count = bdd_lowmem_capi_matmuls(h_t, self.weight, \
            #                                         self.num_rels, self.num_bases, \
            #                                         self.submat_in, self.submat_out, self.out_feat,
            #                                         nonempty_rels, etypes)

            # # batch mm
            # msg, edge_count, elem_count = bdd_lowmem_fgemm_batchmm(h_t, self.weight, \
            #                                         self.num_rels, self.num_bases, \
            #                                         self.submat_in, self.submat_out, self.out_feat, \
            #                                         nonempty_rels, etypes)

            # msg, edge_count, elem_count = bdd_lowmem_fgemm_spmm(h_t, self.weight, \
            # # spmm
            # msg, edge_count, elem_count = bdd_lowmem_fgemm_spmm(h, self.weight, \
            #                                             self.num_rels, self.num_bases, \
            #                                             self.submat_in, self.submat_out, self.out_feat, \
            #                                             nonempty_rels, etypes)

            # msg, edge_count, elem_count = bdd_lowmem_fgemm_blockspmm(h_t, self.weight, \
            # block spmm
            msg, edge_count, elem_count = bdd_lowmem_fgemm_blockspmm(h, self.weight, \
                                                        self.num_rels, self.num_bases, \
                                                        self.submat_in, self.submat_out, self.out_feat, \
                                                        nonempty_rels, etypes)

            th.cuda.nvtx.range_pop()
            print(f"msg: {msg}")
            print(f"msg.sum: {msg.sum()}")
            # print(f"elem_count: {elem_count} edge_count: {edge_count}")
        else:
            # Use batched matmult
            if isinstance(etypes, list):
                etypes = th.repeat_interleave(th.arange(len(etypes), device=device),
                                              th.tensor(etypes, device=device))
            weight = self.weight.index_select(0, etypes).view(
                -1, self.submat_in, self.submat_out)
            node = h.view(-1, 1, self.submat_in)
            elem_count = node.numel() + weight.numel()
            msg = th.bmm(node, weight).view(-1, self.out_feat)
            # print(f"msg: {msg} elem_count: {elem_count}")
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, feat, etypes, norm=None, epoch_fwd=0, nonempty_rels=None, 
                    nonempty_etypes=None, index=None):

        global epoch
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input node features. Could be either

                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor or list[int]
            Edge type data. Could be either

                * An :math:`(|E|,)` dense tensor. Each element corresponds to the edge's type ID.
                  Preferred format if ``lowmem == False``.
                * An integer list. The i^th element is the number of edges of the i^th type.
                  This requires the input graph to store edges sorted by their type IDs.
                  Preferred format if ``lowmem == True``.
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`.

        Returns
        -------
        torch.Tensor
            New node features.

        Notes
        -----
        Under the ``low_mem`` mode, DGL will sort the graph based on the edge types
        and compute message passing one type at a time. DGL recommends sorts the
        graph beforehand (and cache it if possible) and provides the integer list
        format to the ``etypes`` argument. Use DGL's :func:`~dgl.to_homogeneous` API
        to get a sorted homogeneous graph from a heterogeneous graph. Pass ``return_count=True``
        to it to get the ``etypes`` in integer list.
        """
        th.cuda.nvtx.range_push("nvtx-layer")
        epoch = epoch_fwd
        if isinstance(etypes, th.Tensor):
            if len(etypes) != g.num_edges():
                raise DGLError('"etypes" tensor must have length equal to the number of edges'
                               ' in the graph. But got {} and {}.'.format(
                                   len(etypes), g.num_edges()))
            if self.low_mem and not (feat.dtype == th.int64 and feat.ndim == 1):
                th.cuda.nvtx.range_push("nvtx-etypes-list")
                # Create a new etypes to be an integer list of number of edges.
                if norm is not None:
                    norm = norm[index]
                th.cuda.nvtx.range_pop()

        with g.local_scope():
            th.cuda.nvtx.range_push("nvtx-store-feat")
            g.srcdata['h'] = feat
            th.cuda.nvtx.range_pop()

            if norm is not None:
                th.cuda.nvtx.range_push("nvtx-store-norm")
                g.edata['norm'] = norm
                th.cuda.nvtx.range_pop()
            if self.self_loop:
                th.cuda.nvtx.range_push("nvtx-select-loop")
                loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                                                         self.loop_weight)
                th.cuda.nvtx.range_pop()

            # with profiler.record_function("rf-spmm"):
            th.cuda.nvtx.range_push("nvtx-message-passing")
            th.cuda.nvtx.range_push("nvtx-get-edge-data")
            src, dst = g.edges()
            edge_data = g.ndata['h']["_N"][src]
            th.cuda.nvtx.range_pop()

            # # message passing
            # g.update_all(functools.partial(self.message_func, etypes=etypes, \
            #         nonempty_rels=nonempty_rels, nonempty_etypes=nonempty_etypes), \
            #         fn.sum(msg='msg', out='h'))

            updated_edge_data = self.message_func(edge_data, etypes=etypes, \
                        nonempty_rels=nonempty_rels, nonempty_etypes=nonempty_etypes, \
                        norm=norm)["msg"]
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-spmm")
            node_repr = gspmm(g, "copy_rhs", "sum", None, updated_edge_data)
            g.dstdata['h'] = node_repr

            th.cuda.nvtx.range_pop()

            # apply bias and activation
            node_repr = g.dstdata['h']

            if self.layer_norm:
                # with profiler.record_function("rf-norm"):
                th.cuda.nvtx.range_push("nvtx-norm")
                node_repr = self.layer_norm_weight(node_repr)
                th.cuda.nvtx.range_pop()
            if self.bias:
                # with profiler.record_function("rf-bias"):
                th.cuda.nvtx.range_push("nvtx-bias")
                node_repr = node_repr + self.h_bias
                th.cuda.nvtx.range_pop()
            if self.self_loop:
                # with profiler.record_function("rf-selfloop"):
                th.cuda.nvtx.range_push("nvtx-selfloop")
                node_repr = node_repr + loop_message
                th.cuda.nvtx.range_pop()
            if self.activation:
                # with profiler.record_function("rf-activation"):
                th.cuda.nvtx.range_push("nvtx-activation")
                node_repr = self.activation(node_repr)
                th.cuda.nvtx.range_pop()
            # with profiler.record_function("rf-dropout"):
            th.cuda.nvtx.range_push("nvtx-dropout")
            node_repr = self.dropout(node_repr)
            th.cuda.nvtx.range_pop()
            th.cuda.nvtx.range_pop() # nvtx-layer
            return node_repr

_TORCH_HAS_SEARCHSORTED = getattr(th, 'searchsorted', None)

def _searchsorted(sorted_sequence, values):
    # searchsorted is introduced to PyTorch in 1.6.0
    if _TORCH_HAS_SEARCHSORTED:
        return th.searchsorted(sorted_sequence, values)
    else:
        device = values.device
        return th.from_numpy(np.searchsorted(sorted_sequence.cpu().numpy(),
                                             values.cpu().numpy())).to(device)
