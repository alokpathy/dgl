"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse, gc
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.multiprocessing as mp
from dgl.multiprocessing import Queue
from dgl.nn.functional import edge_softmax
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import RelGraphEmbedLayer
from dgl.nn import RelGraphConv
import tqdm

import math

from ogb.nodeproppred import DglNodePropPredDataset
import torch.autograd.profiler as profiler

th.manual_seed(0)

class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(th.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(th.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h, step=0):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            th.cuda.nvtx.range_push("nvtx-attn-iters")
            iter_count = 0
            outer_start = th.cuda.Event(enable_timing=True)
            outer_end = th.cuda.Event(enable_timing=True)
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            subgraph_time = 0.0
            getlinear_params_time = 0.0
            multiply_linears_time = 0.0
            getrelation_params_time = 0.0
            einsum_time = 0.0
            storedata_time = 0.0
            apply_edges_time = 0.0
            compute_attnscore_time = 0.0
            edge_softmax_time = 0.0
            store_attnscore_time = 0.0

            # outer_start.record()
            for srctype, etype, dsttype in G.canonical_etypes:
                if step == 3 and iter_count == 20:
                    th.cuda.nvtx.range_pop()
                    th.cuda.profiler.cudart().cudaProfilerStop()
                    exit()
                th.cuda.nvtx.range_push("nvtx-attn-iters{}".format(iter_count))

                th.cuda.nvtx.range_push("nvtx-subgraph")
                # start.record()
                sub_graph = G[srctype, etype, dsttype]
                # end.record()
                # th.cuda.synchronize()
                # subgraph_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-getlinear-params")
                # start.record()
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                # end.record()
                # th.cuda.synchronize()
                # getlinear_params_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-multiply-linears")
                # start.record()
                # k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                # v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                k = k_linear(h["src"][srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h["src"][srctype]).view(-1, self.n_heads, self.d_k)
                # q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h["dst"][dsttype]).view(-1, self.n_heads, self.d_k)
                # end.record()
                # th.cuda.synchronize()
                # multiply_linears_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-getrelation-params")
                # start.record()
                # e_id = self.edge_dict[etype]
                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]
                # end.record()
                # th.cuda.synchronize()
                # getrelation_params_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-einsum")
                # start.record()
                k = th.einsum("bij,ijk->bik", k, relation_att)
                v = th.einsum("bij,ijk->bik", v, relation_msg)
                # end.record()
                # th.cuda.synchronize()
                # einsum_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-storedata")
                # start.record()
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                # end.record()
                # th.cuda.synchronize()
                # storedata_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-apply-edges")
                # start.record()
                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                # end.record()
                # th.cuda.synchronize()
                # apply_edges_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-compute-attnscore")
                # start.record()
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                # end.record()
                # th.cuda.synchronize()
                # compute_attnscore_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-edge-softmax")
                # start.record()
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
                # end.record()
                # th.cuda.synchronize()
                # edge_softmax_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-store-attnscore")
                # start.record()
                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
                # end.record()
                # th.cuda.synchronize()
                # store_attnscore_time += start.elapsed_time(end)
                th.cuda.nvtx.range_pop()

                iter_count += 1
                th.cuda.nvtx.range_pop()
            # outer_end.record()
            # th.cuda.synchronize()
            # total_time = outer_start.elapsed_time(outer_end)
            th.cuda.nvtx.range_pop()

            # if step == 3:
            #     print(f"total_time = {total_time}")
            #     print(f"subgraph_time = {subgraph_time}")
            #     print(f"getlinear_params_time = {getlinear_params_time}")
            #     print(f"multiply_linears_time = {multiply_linears_time}")
            #     print(f"getrelation_params_time = {getrelation_params_time}")
            #     print(f"einsum_time = {einsum_time}")
            #     print(f"storedata_time = {storedata_time}")
            #     print(f"apply_edges_time = {apply_edges_time}")
            #     print(f"compute_attnscore_time = {compute_attnscore_time}")
            #     print(f"edge_softmax_time = {edge_softmax_time}")
            #     print(f"store_attnscore_time = {store_attnscore_time}")

            th.cuda.nvtx.range_push("nvtx-multi-update-all")
            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')
            th.cuda.nvtx.range_pop()

            th.cuda.nvtx.range_push("nvtx-ts-aggr")
            iter_count = 0
            new_h = {}
            # for ntype in G.ntypes:
            for ntype in G.dsttypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                th.cuda.nvtx.range_push("nvtx-ts-aggr{}".format(iter_count))
                th.cuda.nvtx.range_push("nvtx-ts-aggr-drop-linear")
                n_id = node_dict[ntype]
                alpha = th.sigmoid(self.skip[n_id])
                # t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                t = G.dstnodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                # trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                trans_out = trans_out * alpha + h["dst"][ntype] * (1-alpha)
                th.cuda.nvtx.range_pop()
                th.cuda.nvtx.range_push("nvtx-ts-aggr-norm")
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
                th.cuda.nvtx.range_pop()
                th.cuda.nvtx.range_pop()
                iter_count += 1
            th.cuda.nvtx.range_pop()
            return new_h

class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, blocks, out_key, epoch=0, step=0):
        device = th.device("cuda:0")
        for i in range(self.n_layers):
            h = {}
            h["src"] = {}
            h["dst"] = {}
            G = blocks[i]
            G = G.to(device)
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                # nsrcnode_dict = {ntype : G.number_of_src_nodes(ntype) for ntype in G.srctypes}
                # ndstnode_dict = {ntype : G.number_of_dst_nodes(ntype) for ntype in G.dsttypes}
                # h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
                h["src"][ntype] = F.gelu(self.adapt_ws[n_id](G.srcnodes[ntype].data['inp']))
                h["dst"][ntype] = F.gelu(self.adapt_ws[n_id](G.dstnodes[ntype].data['inp']))
            if epoch == 0 and step == 3:
                print(f"block: {G} len(G.ntypes): {len(G.ntypes)}")
                th.cuda.profiler.cudart().cudaProfilerStart()
                th.cuda.nvtx.range_push("nvtx-layer")
                h = self.gcs[i](G, h, step=step)
                th.cuda.nvtx.range_pop()
                th.cuda.profiler.cudart().cudaProfilerStop()
                exit()
            else:
                h = self.gcs[i](G, h)
        return self.out(h[out_key])

def gen_norm(g):
    _, v, eid = g.all_edges(form='all')
    _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    norm = th.ones(eid.shape[0], device=eid.device) / degrees
    norm = norm.unsqueeze(1)
    g.edata['norm'] = norm

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """
    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            gen_norm(block)
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def evaluate(model, embed_layer, eval_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with th.no_grad():
        th.cuda.empty_cache()
        for sample_data in tqdm.tqdm(eval_loader):
            seeds, blocks = sample_data

            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata['ntype'],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())

    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)

    return eval_logits, eval_seeds

def run(proc_id, n_gpus, n_cpus, args, devices, dataset, split, queue=None):
    #     run(0, n_gpus, n_cpus, args, devices,
    #         (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
    #         train_idx, val_idx, test_idx, labels), None, None)
    dev_id = devices[proc_id] if devices[proc_id] != 'cpu' else -1
    g, node_feats, num_of_ntype, num_classes, num_rels, category, \
        train_idx, val_idx, test_idx, labels = dataset

    if split is not None:
        train_seed, val_seed, test_seed = split
        train_idx = train_idx[train_seed]
        val_idx = val_idx[val_seed]
        test_idx = test_idx[test_seed]

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    node_tids = g.ndata[dgl.NTYPE]
    sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    loader = dgl.dataloading.NodeDataLoader(
        g, {category: train_idx}, sampler,
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    # validation sampler
    # we do not use full neighbor to save computation resources
    val_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    val_loader = dgl.dataloading.NodeDataLoader(
        g, {category: val_idx}, val_sampler,
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    # # test sampler
    # test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    # test_loader = DataLoader(dataset=test_idx.numpy(),
    #                          batch_size=args.eval_batch_size,
    #                          collate_fn=test_sampler.sample_blocks,
    #                          shuffle=False,
    #                          num_workers=args.num_workers)

    world_size = n_gpus
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        backend = 'nccl'

        # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
        if args.dgl_sparse is False:
            backend = 'gloo'
        print("backend using {}".format(backend))
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    # create model
    # all model params are in device.
    device = th.device("cuda:0")
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    # for etype in g.etypes:
    for etype in g.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
        g.edges[etype].data['id'] = th.ones(g.number_of_edges(etype), dtype=th.long) * edge_dict[etype] 

    #     Random initialize input feature
    for ntype in g.ntypes:
        emb = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb

    g = g.to(device)

    model = HGT(g,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hidden,
                n_out=labels.max().item()+1,
                n_layers=2,
                n_heads=4,
                use_norm = True).to(device)
    optimizer = th.optim.AdamW(model.parameters())
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epochs, max_lr = args.max_lr)

    if dev_id >= 0 and n_gpus == 1:
        th.cuda.set_device(dev_id)
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        # with dgl_sparse emb, only node embedding is not in GPU
        if args.dgl_sparse:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        if args.dgl_sparse:
            embed_layer.cuda(dev_id)
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        else:
            if len(list(embed_layer.parameters())) > 0:
                embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)

    # optimizer
    dense_params = list(model.parameters())
    if args.node_feats:
        if  n_gpus > 1:
            dense_params += list(embed_layer.module.embeds.parameters())
        else:
            dense_params += list(embed_layer.embeds.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)

    if args.dgl_sparse:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)
        if n_gpus > 1 and isinstance(embed_layer, DistributedDataParallel):
            dgl_emb = embed_layer.module.dgl_emb
        else:
            dgl_emb = embed_layer.dgl_emb
        emb_optimizer = dgl.optim.SparseAdam(params=dgl_emb, lr=args.sparse_lr, eps=1e-8) if len(dgl_emb) > 0 else None

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    train_time = 0
    validation_time = 0
    test_time = 0
    last_val_acc = 0.0
    do_test = False
    if n_gpus > 1 and n_cpus - args.num_workers > 0:
        th.set_num_threads(n_cpus-args.num_workers)
    train_step = th.tensor(0)
    for epoch in range(args.n_epochs):
        tstart = time.time()
        model.train()

        for i, sample_data in enumerate(loader):
            input_nodes, seeds, blocks = sample_data
            t0 = time.time()

            seeds = seeds[category]     # we only predict the nodes with type "category"
            lbl = labels[seeds]

            t0 = time.time()
            logits = model(blocks, category, epoch=epoch, step=i)
            # logits = model(blocks, 'paper')
            t1 = time.time()
            # The loss is computed only for labeled nodes.
            # loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
            loss = F.cross_entropy(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            t2 = time.time()
            th.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            train_acc = th.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)
            if i % 100 and proc_id == 0:
                print("Train Accuracy: {:.4f} | Train Loss: {:.4f}".
                    format(train_acc, loss.item()))
        train_step += 1
        scheduler.step(train_step)
        gc.collect()
        print("Epoch {:05d}:{:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            format(epoch, args.n_epochs, forward_time[-1], backward_time[-1]))
        tend = time.time()
        train_time += (tend - tstart)

        def collect_eval():
            eval_logits = []
            eval_seeds = []
            for i in range(n_gpus):
                log = queue.get()
                eval_l, eval_s = log
                eval_logits.append(eval_l)
                eval_seeds.append(eval_s)
            eval_logits = th.cat(eval_logits)
            eval_seeds = th.cat(eval_seeds)
            eval_loss = F.cross_entropy(eval_logits, labels[eval_seeds].cpu()).item()
            eval_acc = th.sum(eval_logits.argmax(dim=1) == labels[eval_seeds].cpu()).item() / len(eval_seeds)

            return eval_loss, eval_acc

        vstart = time.time()
        # if (queue is not None) or (proc_id == 0):
        #     val_logits, val_seeds = evaluate(model, embed_layer, val_loader, node_feats)
        #     if queue is not None:
        #         queue.put((val_logits, val_seeds))

        #     # gather evaluation result from multiple processes
        #     if proc_id == 0:
        #         val_loss, val_acc = collect_eval() if queue is not None else \
        #             (F.cross_entropy(val_logits, labels[val_seeds].cpu()).item(), \
        #             th.sum(val_logits.argmax(dim=1) == labels[val_seeds].cpu()).item() / len(val_seeds))

        #         do_test = val_acc > last_val_acc
        #         last_val_acc = val_acc
        #         print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
        #                 format(val_acc, val_loss))
        # if n_gpus > 1:
        #     th.distributed.barrier()
        #     if proc_id == 0:
        #         for i in range(1, n_gpus):
        #             queue.put(do_test)
        #     else:
        #         do_test = queue.get()

        vend = time.time()
        validation_time += (vend - vstart)

        if epoch > 0 and do_test:
            tstart = time.time()
            if (queue is not None) or (proc_id == 0):
                test_logits, test_seeds = evaluate(model, embed_layer, test_loader, node_feats)
                if queue is not None:
                    queue.put((test_logits, test_seeds))

                # gather evaluation result from multiple processes
                if proc_id == 0:
                    test_loss, test_acc = collect_eval() if queue is not None else \
                        (F.cross_entropy(test_logits, labels[test_seeds].cpu()).item(), \
                        th.sum(test_logits.argmax(dim=1) == labels[test_seeds].cpu()).item() / len(test_seeds))
                    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
                    print()
            tend = time.time()
            test_time += (tend-tstart)

            # sync for test
            if n_gpus > 1:
                th.distributed.barrier()

    print("{}/{} Mean forward time: {:4f}".format(proc_id, n_gpus,
                                                  np.mean(forward_time[len(forward_time) // 4:])))
    print("{}/{} Mean backward time: {:4f}".format(proc_id, n_gpus,
                                                   np.mean(backward_time[len(backward_time) // 4:])))
    if proc_id == 0:
        print("Final Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
        print("Train {}s, valid {}s, test {}s".format(train_time, validation_time, test_time))

def main(args, devices):
    # load graph data
    ogb_dataset = False
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
        category = "rev-productionPeriod"
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset(name=args.dataset)
        ogb_dataset = True
    else:
        raise ValueError()

    if ogb_dataset is True:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        if args.dataset == 'ogbn-mag':
            category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

    else:
        # Load from hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        train_mask = hg.nodes[category].data.pop('train_mask')
        test_mask = hg.nodes[category].data.pop('test_mask')
        labels = hg.nodes[category].data.pop('labels')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

        # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
        # Split train set into train and validation if args.validation is set
        # otherwise use train set as the validation set.
        if args.validation:
            val_idx = train_idx[:len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5:]
        else:
            val_idx = train_idx

    node_feats = []
    for ntype in hg.ntypes:
        if len(hg.nodes[ntype].data) == 0 or args.node_feats is False:
            node_feats.append(hg.number_of_nodes(ntype))
        else:
            assert len(hg.nodes[ntype].data) == 1
            feat = hg.nodes[ntype].data.pop('feat')
            node_feats.append(feat.share_memory_())

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
        print('{}:{}'.format(i, ntype))

    g = hg
    print(f"g: {g}")
    print(f"g.ntypes: {g.ntypes}")
    print(f"g.etypes: {g.etypes}")
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    # for i, etype in enumerate(g.etypes):
    for i, etype in enumerate(g.canonical_etypes):
        edge_dict[etype] = len(edge_dict)
        g.edges[etype].data['id'] = th.ones(g.number_of_edges(etype), dtype=th.long) * edge_dict[etype] 
        # g.edges[etype].data['id'] = th.ones(g[etype].number_of_edges(), dtype=th.long) * edge_dict[etype] 

    #     Random initialize input feature
    for ntype in g.ntypes:
        emb = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        g.nodes[ntype].data['inp'] = emb
    # g = dgl.to_homogeneous(hg)
    # g.ndata['ntype'] = g.ndata[dgl.NTYPE]
    # g.ndata['ntype'].share_memory_()
    # g.edata['etype'] = g.edata[dgl.ETYPE]
    # g.edata['etype'].share_memory_()
    # g.ndata['type_id'] = g.ndata[dgl.NID]
    # g.ndata['type_id'].share_memory_()
    # node_ids = th.arange(g.number_of_nodes())

    # # find out the target node ids
    # node_tids = g.ndata[dgl.NTYPE]
    # loc = (node_tids == category_id)
    # target_idx = node_ids[loc]
    # target_idx.share_memory_()
    # train_idx.share_memory_()
    # val_idx.share_memory_()
    # test_idx.share_memory_()
    # # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    # g.create_formats_()

    n_gpus = len(devices)
    n_cpus = mp.cpu_count()
    # cpu
    if devices[0] == -1:
        run(0, 0, n_cpus, args, ['cpu'],
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
             train_idx, val_idx, test_idx, labels), None, None)
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, n_cpus, args, devices,
            (g, node_feats, num_of_ntype, num_classes, num_rels, category,
            train_idx, val_idx, test_idx, labels), None, None)
    else:
        print("no multi-gpu")

def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=256,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--sparse-lr", type=float, default=2e-2,
            help="sparse embedding learning rate")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--fanout", type=str, default="4",
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument('--n_inp',   type=int, default=256)
    parser.add_argument('--clip',    type=int, default=1.0) 
    parser.add_argument('--max_lr',  type=float, default=1e-3) 
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=32,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--dgl-sparse", default=False, action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.add_argument('--node-feats', default=False, action='store_true',
            help='Whether use node features')
    parser.add_argument('--layer-norm', default=False, action='store_true',
            help='Use layer norm')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)
