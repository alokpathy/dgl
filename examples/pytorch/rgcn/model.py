import torch as th
import torch.nn as nn

import dgl
from dgl.nn.pytorch import RelGraphConv
from dgl import edge_subgraph

timing = True

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

_TORCH_HAS_SEARCHSORTED = getattr(th, 'searchsorted', None)

def _searchsorted(sorted_sequence, values):
    # searchsorted is introduced to PyTorch in 1.6.0
    if _TORCH_HAS_SEARCHSORTED:
        return th.searchsorted(sorted_sequence, values)
    else:
        device = values.device
        return th.from_numpy(np.searchsorted(sorted_sequence.cpu().numpy(),
                                             values.cpu().numpy())).to(device)

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm, epoch=0):
        layer_start = th.cuda.Event(enable_timing=True)
        layer_stop = th.cuda.Event(enable_timing=True)

        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(g, h, r, norm, epoch_fwd=epoch)
            else:
                # Low-mem optimization is not enabled for node ID input. When enabled,
                # it first sorts the graph based on the edge types (the sorting will not
                # change the node IDs). It then converts the etypes tensor to an integer
                # list, where each element is the number of edges of the type.
                # Sort the graph based on the etypes
                th.cuda.nvtx.range_push("nvtx-sort-edges")
                sorted_etypes, index = th.sort(r)
                pos = _searchsorted(sorted_etypes, th.arange(self.num_rels, device=g.device))
                num = th.tensor([len(r)], device=g.device)
                etypes = (th.cat([pos[1:], num]) - pos).tolist()
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-new-subgraph")
                g = edge_subgraph(g, index, preserve_nodes=True)
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-lowmem-nonemptyrels")
                nonempty_rels = th.cuda.LongTensor([i for i in range(len(etypes)) if etypes[i] != 0])
                th.cuda.nvtx.range_pop()

                th.cuda.nvtx.range_push("nvtx-lowmem-etypes")
                nonempty_etypes = th.IntTensor([i for i in etypes if i != 0])
                th.cuda.nvtx.range_pop()

                if epoch == 5: # 5th epoch and first RelGraphConv layer
                    th.cuda.profiler.cudart().cudaProfilerStart()
                    th.cuda.nvtx.range_push("nvtx-layer")

                    start_time(layer_start)
                    # h = layer(g, h, r, norm, epoch_fwd=epoch)
                    h = layer(g, h, etypes, norm, epoch_fwd=epoch, nonempty_rels=nonempty_rels, \
                                    nonempty_etypes=nonempty_etypes, index=index)
                    layer_time = stop_time(layer_start, layer_stop)

                    th.cuda.nvtx.range_pop()
                    th.cuda.profiler.cudart().cudaProfilerStop()

                    print(f"layer_time: {layer_time}")
                    exit()
                else:
                    h = layer(g, h, etypes, norm, epoch_fwd=epoch, \
                                    nonempty_rels=nonempty_rels, nonempty_etypes=nonempty_etypes, index=index)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    @property
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.dev_id)
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            loc = node_tids == ntype
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.dev_id)
            else:
                embeds[loc] = features[ntype][type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)

        return embeds
