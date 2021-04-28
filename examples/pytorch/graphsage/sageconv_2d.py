import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape
import math
import torch.distributed as dist

    #I need:
    #adj_matrix, inputs, rank, row, col, size, acc_per_rank, row_groups, col_groups, height, middim, width
    #We know self contains size, rank, row_groups/col_groups
    #uh where do I get adj_matrix, row, col from?
    #these are likely to be parameters right? agg is some sort of function?

def broad_func_2d(self, graph, adj_matrix, agg, inputs, row, col, height, middim, width):
    proc_row = proc_row_size(self.size)
    proc_col = proc_col_size(self.size)

    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col

    middim_per_proc = middim // proc_col

    #resetting height_per_proc/width_per_proc in edge case
    if row == proc_row - 1:
        height_per_proc = height - height_per_proc * (proc_row - 1)

    if col == proc_col - 1:
        width_per_proc = width - width_per_proc * (proc_col - 1)

    #setting z_loc, which is what we're returning
    z_loc = torch.FloatTensor(height_per_proc, width_per_proc).fill(0)

    #k is a column/row, I think yeehaw assuming a square grid
    for k in range(proc_col):

        row_src_rank = k + proc_col * row #counter that goes down row
        col_src_rank = k * proc_col + col #counter that goes down col

        #if last column, we change middim
        if k == proc_col - 1:
            middim_per_proc = middim - middim_per_proc * (proc_col - 1)

        #if row/col is the same as current, set acol len to specific processor specs
        if row_src_rank == self.rank:
            acol_indices_len = torch.LongTensor(
                                            [adj_matrix.indices().contiguous()[0].size(0)])
            acol_values_len = torch.LongTensor([adj_matrix.values().contiguous().size(0)])
        #if not, just set it to 0
        else:
            acol_indices_len = torch.LongTensor([0])
            acol_values_len = torch.LongTensor([0])
    
        #The Broadcast pt. 1
        dist.broadcast(acol_indices_len, row_src_rank, group=self.row_groups[row])

        #not sure yet what these do
        acol_indices_len = acol_indices_len.item()
        acol_values_len = acol_indices_len

        if row_src_rank == self.rank:
            acol_indices = adj_matrix.indices().contiguous().long()
            acol_values = adj_matrix.values().contiguous().float()
        else:
            acol_indices = torch.LongTensor(2, acol_indices_len).fill_(0)
            acol_values = torch.FloatTensor(acol_values_len).fill_(0)

        #unsplits it I think
        acol = torch.cat((acol_indices.float(), acol_values.unsqueeze(0)), dim=0).contiguous()

        dist.broadcast(acol, row_src_rank, group=self.row_groups[row])

        #wot
        if self.rank == 0:
            summa_sparse_bcast1_words[run][self.rank] += 3 * acol_values_len
        acol_indices = acol[:2].long()
        acol_values = acol[2].squeeze(0)

        if row_src_rank == self.rank:
            acol = adj_matrix
        else:
            acol = torch.sparse_coo_tensor(acol_indices, acol_values, 
                                            torch.Size([height_per_proc, middim_per_proc]))

        if col_src_rank == self.rank:
            brow = inputs
        else:
            brow = torch.FloatTensor(middim_per_proc, width_per_proc)

        brow = brow.contiguous()

        dist.broadcast(brow, col_src_rank, group=self.col_groups[col])

        z_loc = agg(graph, brow)
        #if rank == 0:
        #    summa_sparse_bcast2_words[run][rank] += brow.size(0) * brow.size(1)

    return z_loc


def broad_func_2d_loc(self, graph, mata, agg, matb, row, col, height, middim, width):
    proc_row = proc_row_size(self.size)
    proc_col = proc_col_size(self.size)

    height_per_proc = height // proc_row
    width_per_proc  = width // proc_col

    middim_per_proc = middim // proc_row

    #resetting height_per_proc/width_per_proc in edge case
    if row == proc_row - 1:
        height_per_proc = height - height_per_proc * (proc_row - 1)

    # if col == proc_col - 1:
    #     width_per_proc = width - width_per_proc * (proc_col - 1)
    #NEW
    width_per_proc = matb[rank].size(1)

    acol_tens = torch.FloatTensor(height_per_proc, middim_per_proc)
    brow_tens = torch.FloatTensor(middim_per_proc, width_per_proc)

    acol = acol_tens
    brow = brow_tens

    #setting z_loc, which is what we're returning
    z_loc = torch.FloatTensor(height_per_proc, width_per_proc).fill(0)

    #k is a column/row, I think yeehaw assuming a square grid
    for k in range(proc_col):
        
        row_src_rank = k + proc_col * row #counter that goes down row
        col_src_rank = k * proc_col + col #counter that goes down col

        #if last column, we change middim
        if k == proc_col - 1:
            #middim_per_proc = middim - middim_per_proc * (proc_col - 1)
            middim_per_proc -= proc_col * middim_per_proc - middim

        if row_src_rank == self.rank:
            acol = mata
        else:
            acol = acol_tens
            acol = torch.FloatTensor(height_per_proc, matb[col_src_rank].size(0))
        
        acol = acol.contiguous()
    
        #The Broadcast pt. 1
        dist.broadcast(acol, row_src_rank, row_groups[row])

        brow = matb[col_src_rank]

        z_loc += torch.mm(acol, brow)
        #if rank == 0:
        #    summa_sparse_bcast2_words[run][rank] += brow.size(0) * brow.size(1)

    return z_loc

class SAGEConvAgg(nn.Module):
    def __init__(self,
                in_feats,
                # out_feats,
                aggregator_type):
                
        super(SAGEConvAgg, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
       
        self._aggre_type = aggregator_type

        self._agg_loc = SAGEConvAggLoc(in_feats, aggregator_type)

    def forward(self, graphsage, graph, ampbyp, inputs):
        return SAGEConvAggFn.apply(graphsage, graph, ampbyp, self._agg_loc, inputs)


class SAGEConvAggFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, graph, adj_matrix, agg, inputs):
        rank_row = int(rank / proc_col)
        rank_col = rank % proc_col
        width = adj_matrix.shape()[1]
        z = broad_func(self, graph, adj_matrix, agg, inputs, rank_row, rank_col, node_count, node_count, width)
        ctx.save_for_backward(inputs)
        ctx.ampbyp = ampbyp
        ctx.graph = graph
        ctx.agg = agg
        ctx.self = self

        return z

    @staticmethod
    def backward(ctx, grad_output):
        graph = ctx.graph
        agg = ctx.agg
        ampbyp = ctx.ampbyp
        inputs = ctx.saved_tensors
        self = ctx.self

        dz = broad_func(self, graph, ampbyp, agg, grad_output)

        return None, None, None, None, dz


class SAGEConvAggLoc(nn.Module):
    r"""

    """
    def __init__(self,
                in_feats,
                # out_feats,
                aggregator_type):
                
        super(SAGEConvAggLoc, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
       
        self._aggre_type = aggregator_type
              
    
    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}
    
    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                # No dropout in aggregation
                # feat_src = feat_dst = self.feat_drop(feat)
                feat_src = feat_dst = feat 
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst     # same as above if homogeneous
                graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
                h_neigh = graph.dstdata['neigh']
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))
        
            return h_neigh


class SAGEConvMLP(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConvMLP, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, h_neigh):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        
        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            # rst = self.fc_neigh(h_neigh)
            rst = SAGEConvMLPFn.apply(h_neigh, self.fc_neigh.weight, self.fc_neigh.bias)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst

# Inherit from Function
# Assumes GCN aggregation type
class SAGEConvMLPFn(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
