# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True)
        return x / norm.clamp(min = self.eps) * self.g

class EquivariantUpdate(nn.Module):
    def __init__(
        self, 
        hidden_nf, 
        normalization_factor, 
        aggregation_method,
        edges_in_d=1, 
        tanh=False, 
        coords_range=10.0
    ):
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if linker_mask is not None:
            agg = agg * linker_mask

        coord = coord + agg
        return coord

    def forward(
        self, 
        h, 
        coord, 
        edge_index, 
        coord_diff, 
        edge_attr=None, 
        linker_mask=None, 
        node_mask=None, 
        edge_mask=None
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask, linker_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        nn.init.xavier_normal_(self.in_proj.weight)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        nn.init.xavier_normal_(self.out_proj.weight)


    def forward(
        self,
        query,
        attn_bias = None,
        graph_attn_weight = None,
    ):
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1) # (bs*n_head, n_nodes, head_dim)
        if attn_bias is not None:
            attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_probs = torch.tanh(graph_attn_weight) * F.softmax(attn_weights, -1)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn

class Graphormer3DEncoderLayer(nn.Module):
    """
    Implements a Graphormer-3D Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        # self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        # nn.init.xavier_normal_(self.fc1.weight)
        # self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        # nn.init.xavier_normal_(self.fc2.weight)
        self.fc1 = GLU(embedding_dim, ffn_embedding_dim, ReluSquared())
        self.fc2 = GLU(ffn_embedding_dim, embedding_dim, ReluSquared())
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x,
        attn_bias = None,
        graph_attn_weight=None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=attn_bias,
            graph_attn_weight=graph_attn_weight,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x

class GraphormerBlock(nn.Module):
    def __init__(
            self, 
            hidden_nf, 
            edge_feat_nf=2, 
            device='cpu', 
            tanh=False, 
            coords_range=15, 
            norm_constant=1, 
            normalization_factor=100, 
            aggregation_method='sum',
            ffn_embedding_dim=3072,
            attention_heads=8,
            dropout=0.1,
            activation_dropout= 0.1,
        ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention_heads = attention_heads


        self.add_module("gcl", Graphormer3DEncoderLayer(
            hidden_nf,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=attention_heads,
            dropout=dropout,
            activation_dropout= activation_dropout,
        )) 
        self.gcl_equiv = EquivariantUpdate(
            hidden_nf, 
            edges_in_d=edge_feat_nf, 
            tanh=tanh,
            coords_range=self.coords_range_layer,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method
        )
        
        self.edge_fc1 = nn.Linear(2, attention_heads)
        self.edge_fc2 = nn.Linear(2, attention_heads)
        nn.init.xavier_normal_(self.edge_fc1.weight)
        nn.init.xavier_normal_(self.edge_fc2.weight)

    def forward(
        self, h, x, 
        edge_index, 
        node_mask=None, 
        linker_mask=None, 
        edge_mask=None, 
        edge_attr=None, 
        size=None, 
        graph_attn_bias=None
    ):
        
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        
        bs, n_nodes = size        
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        graph_attn_bias = (self.edge_fc1(edge_attr)*edge_mask).view(bs, n_nodes, n_nodes, self.attention_heads).permute(0,3,1,2).reshape(bs*self.attention_heads, n_nodes, n_nodes)
        graph_attn_weight = (self.edge_fc2(edge_attr)*edge_mask).view(bs, n_nodes, n_nodes, self.attention_heads).permute(0,3,1,2).reshape(bs*self.attention_heads, n_nodes, n_nodes)
        
        h = h.view(bs, n_nodes, -1).transpose(0,1)

        h = self._modules["gcl"](h, attn_bias=graph_attn_bias, graph_attn_weight=graph_attn_weight)
        # h = self.gcl(h, attn_bias=graph_attn_bias)
        h = h.transpose(0,1)
        h = h.view(bs*n_nodes, -1) * node_mask

        x = self.gcl_equiv(
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            linker_mask=linker_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x

class Graphormer3D(nn.Module):
    def __init__(
        self, 
        in_node_nf, #11
        hidden_nf, 
        device='cpu', 
        n_layers=6, 
        tanh=False, 
        coords_range=15, 
        norm_constant=1e-6, 
        normalization_factor=100, 
        aggregation_method='sum',
        ffn_embedding_dim=3072,
        attention_heads=8,
        dropout=0.1,
        activation_dropout= 0.1,
    ):
        super().__init__()

        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, in_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, GraphormerBlock(
                hidden_nf, 
                edge_feat_nf=edge_feat_nf, 
                device=device,
                tanh=tanh,
                coords_range=coords_range, 
                norm_constant=norm_constant,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
                ffn_embedding_dim=ffn_embedding_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                activation_dropout= activation_dropout,
            ))


    def forward(self, h, x, edge_index, node_mask=None, linker_mask=None, edge_mask=None, size=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index) # (B*N*N, 1)
        output = self.embedding(h)
        
        for i in range(0, self.n_layers):
            output, x = self._modules["e_block_%d" % i](
                output, x, edge_index,
                node_mask=node_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                edge_attr=distances,
                size=size,
                graph_attn_bias=None,
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(output)
        if node_mask is not None:
            h = h * node_mask
        return h, x

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

